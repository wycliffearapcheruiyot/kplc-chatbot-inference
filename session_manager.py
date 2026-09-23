"""
session_manager.py

In-memory state machine for the Kaggle GPU session lifecycle.

States:
    idle      - no session running, nothing pushed to Kaggle
    starting  - kernels push has been issued, waiting for serve.py's
                /session/ready webhook
    ready     - serve.py confirmed the model + tunnel are up; /generate and
                /embed can be proxied
    ended     - serve.py shut itself down (idle timeout) and told us
    error     - the push failed, or serve.py reported an early failure, or
                we gave up waiting for /session/ready

NOTE: this is process-local state. Render must run this service with a
single worker process (see README) or the state won't be shared correctly
across workers. For anything beyond a single-instance demo, swap this for
a small external store (Redis, a database row) instead.
"""

import threading
import time
from enum import Enum
from typing import Optional


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    ENDED = "ended"
    ERROR = "error"


class SessionManager:
    def __init__(self, start_timeout_seconds: int = 600):
        self._lock = threading.Lock()
        self._state = SessionState.IDLE
        self._started_at: Optional[float] = None
        self._ready_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._error_detail: Optional[str] = None
        self.start_timeout_seconds = start_timeout_seconds

    # --- transitions -----------------------------------------------------

    def try_begin_start(self) -> bool:
        """Claims the right to start a session. Returns False if a session
        is already starting or ready (caller should not push again)."""
        with self._lock:
            if self._state in (SessionState.STARTING, SessionState.READY):
                return False
            self._state = SessionState.STARTING
            self._started_at = time.time()
            self._ready_at = None
            self._ended_at = None
            self._error_detail = None
            return True

    def mark_ready(self):
        with self._lock:
            self._state = SessionState.READY
            self._ready_at = time.time()

    def mark_ended(self):
        with self._lock:
            self._state = SessionState.ENDED
            self._ended_at = time.time()

    def mark_error(self, detail: str):
        with self._lock:
            self._state = SessionState.ERROR
            self._error_detail = detail

    def reset_to_idle(self):
        with self._lock:
            self._state = SessionState.IDLE
            self._error_detail = None

    # --- reads -------------------------------------------------------------

    def _maybe_timeout(self):
        """If we've been STARTING too long with no /session/ready, flip to
        error so callers (and /generate) stop waiting on a dead session."""
        if self._state == SessionState.STARTING and self._started_at:
            if time.time() - self._started_at > self.start_timeout_seconds:
                self._state = SessionState.ERROR
                self._error_detail = (
                    f"Timed out waiting for /session/ready after "
                    f"{self.start_timeout_seconds}s"
                )

    def snapshot(self) -> dict:
        with self._lock:
            self._maybe_timeout()
            return {
                "state": self._state.value,
                "started_at": self._started_at,
                "ready_at": self._ready_at,
                "ended_at": self._ended_at,
                "error_detail": self._error_detail,
            }

    def is_ready(self) -> bool:
        with self._lock:
            self._maybe_timeout()
            return self._state == SessionState.READY
