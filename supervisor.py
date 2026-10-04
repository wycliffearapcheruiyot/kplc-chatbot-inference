"""
supervisor.py

Always-on mode for the Kaggle session. One background thread that:

  * keeps a session running: whenever the state is idle / ended / error it
    pushes a new kernel (with a back-off, so a quota problem or a 409 doesn't
    hammer Kaggle);
  * restarts on a schedule: serve.py exits cleanly after MAX_RUNTIME_SECONDS
    and calls /session/ended, which this thread answers with a new push;
  * notices a dead model server: while READY it probes the tunnel's /health and
    flips to error after a few misses (Kaggle killed the run without a webhook);
  * re-adopts a live session after a Render restart (state is in memory, the
    Kaggle run isn't) instead of pushing a duplicate;
  * tells the account rotation (accounts.py) when launches keep failing, so a
    broken or out-of-quota account gets skipped instead of retried for hours.

Only the Kaggle session is restarted. Nothing here keeps Render awake: if the
service sleeps, serve.py's /session/ended webhook wakes it and this thread
starts again with the app.
"""

import threading
import time
from datetime import datetime, timezone

import httpx


def parse_window(spec: str):
    """"05:00-13:00" (UTC) -> (start_minute, end_minute); "" -> None (always)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        start, end = spec.split("-")
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        return sh * 60 + sm, eh * 60 + em
    except ValueError:
        print(f"WARNING: ignoring bad KEEP_ALIVE_WINDOW_UTC={spec!r}; use HH:MM-HH:MM", flush=True)
        return None


def in_window(window, now: datetime | None = None) -> bool:
    if window is None:
        return True
    now = now or datetime.now(timezone.utc)
    minute = now.hour * 60 + now.minute
    start, end = window
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end  # wraps past midnight


class Supervisor:
    def __init__(
        self,
        sessions,
        launch,                     # () -> bool ; pushes a session, raises on failure
        tunnel_url: str,
        rotation=None,
        interval_seconds: int = 60,
        retry_backoff_seconds: int = 300,
        after_end_delay_seconds: int = 30,
        window_spec: str = "",
        health_failures_before_error: int = 3,
    ):
        self.sessions = sessions
        self.launch = launch
        self.tunnel_url = tunnel_url.rstrip("/")
        self.rotation = rotation
        self.interval = interval_seconds
        self.retry_backoff = retry_backoff_seconds
        self.after_end_delay = after_end_delay_seconds
        self.window = parse_window(window_spec)
        self.max_misses = health_failures_before_error

        self._misses = 0
        self._attempt_open = False   # a launch this thread made hasn't been judged yet
        self._last_attempt = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="kaggle-supervisor", daemon=True)
        self._thread.start()
        print("Supervisor: always-on mode enabled.", flush=True)

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # never let the loop die
                print(f"Supervisor tick failed: {e!r}", flush=True)
            self._stop.wait(self.interval)

    # --- one pass ----------------------------------------------------------

    def tick(self):
        snap = self.sessions.snapshot()
        state = snap["state"]

        # Judge the launch this thread made last: it either ended in an error
        # (counts toward skipping that account) or reached ready.
        if self._attempt_open:
            if state == "error":
                self._attempt_open = False
                if self.rotation and self.rotation.record_failure(snap.get("error_detail") or ""):
                    print("Supervisor: account kept failing, moving to the next one.", flush=True)
            elif state in ("ready", "ended"):
                self._attempt_open = False

        if state == "starting":
            return  # SessionManager times this into "error" by itself

        if state == "ready":
            if self.rotation:
                self.rotation.succeeded()
            self._check_tunnel()
            return

        # idle / ended / error: make sure a session comes back.
        self._misses = 0
        if self._adopt_if_running():
            return
        if not in_window(self.window):
            return
        wait = self.after_end_delay if state == "ended" else (
            self.retry_backoff if state == "error" else 0
        )
        if time.time() - self._last_attempt < wait:
            return
        self._last_attempt = time.time()
        self._attempt_open = True
        print(f"Supervisor: state={state}, launching a new Kaggle session.", flush=True)
        try:
            self.launch()
        except Exception as e:
            print(f"Supervisor: launch failed: {e}", flush=True)

    # --- helpers -----------------------------------------------------------

    def _probe(self):
        """Returns the tunnel's /health JSON, or None if it isn't answering."""
        if not self.tunnel_url:
            return None
        try:
            r = httpx.get(f"{self.tunnel_url}/health", timeout=15)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, ValueError):
            pass
        return None

    def _adopt_if_running(self) -> bool:
        info = self._probe()
        if not info or info.get("status") != "ok":
            return False
        uptime = float(info.get("uptime_seconds") or 0)
        if self.rotation:
            self.rotation.sync(info.get("session_number"))
        self.sessions.adopt_running(started_at=time.time() - uptime)
        print(f"Supervisor: adopted a model server already up for {uptime:.0f}s.", flush=True)
        return True

    def _check_tunnel(self):
        if self._probe():
            self._misses = 0
            return
        self._misses += 1
        if self._misses >= self.max_misses:
            self._misses = 0
            self.sessions.mark_error(
                f"Model tunnel stopped answering /health ({self.max_misses} checks in a row)."
            )
