"""
main.py

Render-hosted backend for the KPLC chatbot demo.

Responsibilities:
    1. POST /session/start   - triggers `kaggle kernels push` on
                                kplc-kaggle-notebook/, which starts serve.py
                                running on a Kaggle GPU.
    2. POST /session/ready   - webhook serve.py calls once the model +
                                Cloudflare Tunnel are confirmed up.
    3. POST /session/ended   - webhook serve.py calls when it shuts itself
                                down (idle timeout or fatal error).
    4. GET  /session/status  - current state, for the frontend to poll.
    5. POST /generate,
       POST /embed           - proxied through to the Kaggle-hosted model
                                over its fixed Cloudflare Tunnel hostname,
                                only while a session is READY.

Env vars (see .env.example):
    KAGGLE_USERNAME, KAGGLE_KEY   - Kaggle API credentials (one account)
    KAGGLE_ACCOUNTS               - optional JSON of several accounts to rotate
                                     through, {"user": "key", ...}; replaces
                                     KAGGLE_USERNAME / KAGGLE_KEY (see accounts.py)
    SESSIONS_PER_ACCOUNT          - default 3. Launches an account serves in a
                                     row before the next account takes over.
    KAGGLE_KERNEL_PATH            - path to the kernel folder to push
                                     (default: ./kplc-kaggle-notebook)
    SESSION_WEBHOOK_SECRET        - shared secret; must match the Kaggle
                                     Secret of the same name used by serve.py
    MODEL_TUNNEL_URL              - fixed Cloudflare Tunnel hostname, e.g.
                                     https://model.yourdomain.com
    SESSION_START_TIMEOUT_SECONDS - optional, default 600. How long to wait
                                     for /session/ready before giving up.
    PROXY_TIMEOUT_SECONDS         - optional, default 120. Timeout for
                                     proxied /generate calls.

  Always-on mode (all optional; see .env.example):
    AUTO_KEEP_ALIVE               - "true" keeps a Kaggle session running and
                                     relaunches it after every exit.
    RESTART_EVERY_HOURS           - default 8. serve.py exits cleanly after
                                     this long; the supervisor relaunches it.
    KAGGLE_IDLE_TIMEOUT_SECONDS   - default 0 (never) when always-on, 900
                                     otherwise.
    KEEP_ALIVE_WINDOW_UTC         - e.g. "05:00-13:00": only launch inside it.

Every variable above can also be edited live from the admin panel's Environment
tab (stored in MongoDB, read through app_config.cfg, applied within ~10 s). The
real environment variable is the fallback. MONGODB_URI must be a real env var:
it is how this service finds those settings.

Run with a single worker (see README) since session state is in-memory.
"""

import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Body, FastAPI, Header, HTTPException
from pydantic import BaseModel

from accounts import Rotation, load_accounts
from kaggle_client import KagglePushError, push_kernel, push_secrets_dataset
from session_manager import SessionManager
from supervisor import Supervisor

# --- config -----------------------------------------------------------------
# Everything is read through cfg at the moment it is needed (never frozen at
# import), so the admin panel's Environment tab takes effect without a redeploy.

from app_config import cfg


def kaggle_kernel_path() -> str:
    return cfg.get_str("KAGGLE_KERNEL_PATH", "./kplc-kaggle-notebook") or "./kplc-kaggle-notebook"


def webhook_secret() -> str:
    return cfg.get_str("SESSION_WEBHOOK_SECRET")


def model_tunnel_url() -> str:
    return cfg.get_str("MODEL_TUNNEL_URL").rstrip("/")


def proxy_timeout() -> int:
    return cfg.get_int("PROXY_TIMEOUT_SECONDS", 120)


def start_timeout() -> int:
    return cfg.get_int("SESSION_START_TIMEOUT_SECONDS", 600)


# This backend's own public URL, handed to serve.py so it knows where to
# call /session/ready and /session/ended. BACKEND_URL lets you override it;
# otherwise Render auto-injects RENDER_EXTERNAL_URL, so on Render you don't
# need to set this by hand at all.
def backend_url() -> str:
    return cfg.get_str("BACKEND_URL") or os.environ.get("RENDER_EXTERNAL_URL", "").strip()


def cloudflare_tunnel_token() -> str:
    return cfg.get_str("CLOUDFLARE_TUNNEL_TOKEN")


# --- always-on mode -----------------------------------------------------------
def keep_alive() -> bool:
    return cfg.get_bool("AUTO_KEEP_ALIVE", False)


def restart_every_hours() -> float:
    return cfg.get_float("RESTART_EVERY_HOURS", 8.0)


def idle_timeout_seconds() -> int:
    # Always-on: no idle shutdown (the supervisor would only relaunch it anyway,
    # burning a model load per idle gap). Otherwise keep the original 15 min.
    return cfg.get_int("KAGGLE_IDLE_TIMEOUT_SECONDS", 0 if keep_alive() else 900)


def max_runtime_seconds() -> int:
    # Kaggle ends a GPU session at 12h, so a longer cycle can't be honoured.
    return int(min(restart_every_hours(), 11.5) * 3600) if keep_alive() else 0


rotation = Rotation([])


def sync_rotation() -> str | None:
    """Applies the current account settings to the rotation (the launch counter
    is kept). Returns an error message if KAGGLE_ACCOUNTS can't be used."""
    try:
        accounts = load_accounts(
            cfg.get_str("KAGGLE_ACCOUNTS"), cfg.get_str("KAGGLE_USERNAME"), cfg.get_str("KAGGLE_KEY")
        )
    except ValueError as e:
        return str(e)
    rotation.configure(
        accounts,
        sessions_per_account=cfg.get_int("SESSIONS_PER_ACCOUNT", 3),
        skip_after_failures=cfg.get_int("ACCOUNT_SKIP_AFTER_FAILURES", 3),
    )
    return None


if not os.environ.get("MONGODB_URI", "").strip():
    print("WARNING: MONGODB_URI is not set, so the admin panel cannot edit this service's settings "
          "(it falls back to plain environment variables).", flush=True)
_accounts_error = sync_rotation()
if _accounts_error:
    print(f"WARNING: {_accounts_error}", flush=True)

for name, value in [
    ("SESSION_WEBHOOK_SECRET", webhook_secret()),
    ("MODEL_TUNNEL_URL", model_tunnel_url()),
    ("BACKEND_URL", backend_url()),
    ("CLOUDFLARE_TUNNEL_TOKEN", cloudflare_tunnel_token()),
]:
    if not value:
        print(f"WARNING: {name} is not set (env var or admin panel). See .env.example.")

sessions = SessionManager(start_timeout_seconds=start_timeout)


def _launch_session() -> bool:
    """Pushes the secrets dataset + kernel on the account the rotation picks.
    Shared by POST /session/start and the always-on supervisor. Returns False
    if a session is already starting/ready; raises KagglePushError (state ->
    error) on failure."""
    if not sessions.try_begin_start():
        return False
    problem = sync_rotation()
    if problem:
        sessions.mark_error(problem)
        raise KagglePushError(problem)
    try:
        number, idx, account = rotation.plan()
    except RuntimeError as e:
        sessions.mark_error(str(e))
        raise KagglePushError(str(e)) from e
    label = f"account {idx + 1}/{len(rotation.accounts)}"
    try:
        secrets_dataset_ref = push_secrets_dataset(
            account["username"],
            account["key"],
            {
                "BACKEND_URL": backend_url(),
                "SESSION_WEBHOOK_SECRET": webhook_secret(),
                "CLOUDFLARE_TUNNEL_TOKEN": cloudflare_tunnel_token(),
                # strings on purpose: serve.py treats "0" as a real value
                "IDLE_TIMEOUT_SECONDS": str(idle_timeout_seconds()),
                "MAX_RUNTIME_SECONDS": str(max_runtime_seconds()),
                "SESSION_NUMBER": str(number),
            },
        )
        push_kernel(
            kaggle_kernel_path(),
            account["username"],
            account["key"],
            extra_dataset_sources=[secrets_dataset_ref],
        )
    except KagglePushError as e:
        sessions.mark_error(f"[{label}] {e}")
        raise
    rotation.launched()
    print(f"Launched session #{number} on {label}.", flush=True)
    return True


supervisor = Supervisor(
    sessions,
    launch=_launch_session,
    tunnel_url=model_tunnel_url(),
    rotation=rotation,
    # re-read every loop, so edits in the admin panel apply without a restart
    settings=lambda: {
        "enabled": keep_alive(),
        "tunnel_url": model_tunnel_url(),
        "interval_seconds": cfg.get_int("SUPERVISOR_INTERVAL_SECONDS", 60),
        "retry_backoff_seconds": cfg.get_int("RESTART_RETRY_BACKOFF_SECONDS", 300),
        "window_spec": cfg.get_str("KEEP_ALIVE_WINDOW_UTC"),
    },
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    supervisor.start()  # idle unless AUTO_KEEP_ALIVE is true (re-checked every loop)
    yield
    supervisor.stop()


app = FastAPI(title="KPLC Chatbot Backend", lifespan=lifespan)


# --- auth helper for webhooks called by serve.py -----------------------------

def _check_webhook_secret(x_webhook_secret: str | None):
    expected = webhook_secret()
    if not expected or x_webhook_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing webhook secret.")


# --- session lifecycle ---------------------------------------------------------

@app.post("/session/start")
def start_session():
    """Triggers a new Kaggle run. Idempotent: if a session is already
    starting or ready, just returns the current status instead of pushing
    again (avoids spamming `kaggle kernels push` from double-clicks).

    Fully automated — no Kaggle UI steps required. Before pushing the
    kernel, this pushes BACKEND_URL / SESSION_WEBHOOK_SECRET /
    CLOUDFLARE_TUNNEL_TOKEN to a private Kaggle Dataset (see
    kaggle_client.py for why: Kaggle Secrets' UI binding doesn't survive
    CLI pushes, dataset attachments do) and attaches that dataset to the
    kernel so serve.py can read the same config every run via
    /kaggle/input/.../secrets.json.
    """
    try:
        triggered = _launch_session()
    except KagglePushError as e:
        raise HTTPException(status_code=502, detail=f"Failed to start Kaggle session: {e}") from e

    return {"triggered": triggered, **sessions.snapshot()}


@app.post("/session/ready")
def session_ready(
    payload: dict | None = Body(default=None),
    x_webhook_secret: str | None = Header(default=None),
):
    """Called by serve.py once the model is loaded and the tunnel is up."""
    _check_webhook_secret(x_webhook_secret)
    rotation.sync((payload or {}).get("session_number"))
    rotation.succeeded()
    sessions.mark_ready()
    return {"ok": True}


@app.post("/session/ended")
def session_ended(
    payload: dict | None = Body(default=None),
    x_webhook_secret: str | None = Header(default=None),
):
    """Called by serve.py when it shuts itself down (scheduled restart, idle
    timeout or fatal startup error). Its session_number re-syncs the account
    rotation, so a Render service that slept or restarted meanwhile still
    picks the right account for the next launch."""
    _check_webhook_secret(x_webhook_secret)
    rotation.sync((payload or {}).get("session_number"))
    sessions.mark_ended()
    return {"ok": True}


@app.get("/session/status")
def session_status():
    snap = sessions.snapshot()
    sync_rotation()  # so rotation.describe() reflects settings edited in the admin panel
    restart_in = None
    if keep_alive() and snap["state"] == "ready" and snap["started_at"]:
        restart_in = max(0, int(snap["started_at"] + max_runtime_seconds() - time.time()))
    return {
        **snap,
        "keep_alive": keep_alive(),
        "restart_in_seconds": restart_in,
        "rotation": rotation.describe(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# --- proxy to the Kaggle-hosted model -------------------------------------------

class GenerateRequest(BaseModel):
    system_prompt: str
    question: str


class EmbedRequest(BaseModel):
    texts: list[str]


def _require_ready():
    if not sessions.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No model session is ready. POST /session/start first, "
                   "then poll /session/status until state is 'ready'.",
        )


@app.post("/generate")
def generate(req: GenerateRequest):
    _require_ready()
    try:
        resp = httpx.post(
            f"{model_tunnel_url()}/generate",
            json=req.model_dump(),
            timeout=proxy_timeout(),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Model server error: {e.response.text}") from e
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach model server: {e}") from e


@app.post("/embed")
def embed(req: EmbedRequest):
    _require_ready()
    try:
        resp = httpx.post(
            f"{model_tunnel_url()}/embed",
            json=req.model_dump(),
            timeout=proxy_timeout(),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Model server error: {e.response.text}") from e
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach model server: {e}") from e
