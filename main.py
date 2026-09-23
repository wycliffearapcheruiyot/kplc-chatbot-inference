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
    KAGGLE_USERNAME, KAGGLE_KEY   - Kaggle API credentials
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

Run with a single worker (see README) since session state is in-memory.
"""

import os

import httpx
from fastapi import Body, FastAPI, Header, HTTPException
from pydantic import BaseModel

from kaggle_client import KagglePushError, push_kernel, push_secrets_dataset
from session_manager import SessionManager

# --- config -----------------------------------------------------------------

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")
KAGGLE_KERNEL_PATH = os.environ.get("KAGGLE_KERNEL_PATH", "./kplc-kaggle-notebook")
SESSION_WEBHOOK_SECRET = os.environ.get("SESSION_WEBHOOK_SECRET", "")
MODEL_TUNNEL_URL = os.environ.get("MODEL_TUNNEL_URL", "").rstrip("/")
SESSION_START_TIMEOUT_SECONDS = int(os.environ.get("SESSION_START_TIMEOUT_SECONDS", "600"))
PROXY_TIMEOUT_SECONDS = int(os.environ.get("PROXY_TIMEOUT_SECONDS", "120"))

# This backend's own public URL, handed to serve.py so it knows where to
# call /session/ready and /session/ended. BACKEND_URL lets you override it;
# otherwise Render auto-injects RENDER_EXTERNAL_URL, so on Render you don't
# need to set this by hand at all.
BACKEND_URL = os.environ.get("BACKEND_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
CLOUDFLARE_TUNNEL_TOKEN = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")

for name, value in [
    ("SESSION_WEBHOOK_SECRET", SESSION_WEBHOOK_SECRET),
    ("MODEL_TUNNEL_URL", MODEL_TUNNEL_URL),
    ("BACKEND_URL", BACKEND_URL),
    ("CLOUDFLARE_TUNNEL_TOKEN", CLOUDFLARE_TUNNEL_TOKEN),
]:
    if not value:
        print(f"WARNING: env var {name} is not set. See .env.example.")

app = FastAPI(title="KPLC Chatbot Backend")
sessions = SessionManager(start_timeout_seconds=SESSION_START_TIMEOUT_SECONDS)


# --- auth helper for webhooks called by serve.py -----------------------------

def _check_webhook_secret(x_webhook_secret: str | None):
    if not SESSION_WEBHOOK_SECRET or x_webhook_secret != SESSION_WEBHOOK_SECRET:
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
    claimed = sessions.try_begin_start()
    if not claimed:
        return {"triggered": False, **sessions.snapshot()}

    try:
        secrets_dataset_ref = push_secrets_dataset(
            KAGGLE_USERNAME,
            KAGGLE_KEY,
            {
                "BACKEND_URL": BACKEND_URL,
                "SESSION_WEBHOOK_SECRET": SESSION_WEBHOOK_SECRET,
                "CLOUDFLARE_TUNNEL_TOKEN": CLOUDFLARE_TUNNEL_TOKEN,
            },
        )
        push_kernel(
            KAGGLE_KERNEL_PATH,
            KAGGLE_USERNAME,
            KAGGLE_KEY,
            extra_dataset_sources=[secrets_dataset_ref],
        )
    except KagglePushError as e:
        sessions.mark_error(str(e))
        raise HTTPException(status_code=502, detail=f"Failed to start Kaggle session: {e}") from e

    return {"triggered": True, **sessions.snapshot()}


@app.post("/session/ready")
def session_ready(x_webhook_secret: str | None = Header(default=None)):
    """Called by serve.py once the model is loaded and the tunnel is up."""
    _check_webhook_secret(x_webhook_secret)
    sessions.mark_ready()
    return {"ok": True}


@app.post("/session/ended")
def session_ended(x_webhook_secret: str | None = Header(default=None)):
    """Called by serve.py when it shuts itself down (idle timeout or fatal
    startup error)."""
    _check_webhook_secret(x_webhook_secret)
    sessions.mark_ended()
    return {"ok": True}


@app.get("/session/status")
def session_status():
    return sessions.snapshot()


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
            f"{MODEL_TUNNEL_URL}/generate",
            json=req.model_dump(),
            timeout=PROXY_TIMEOUT_SECONDS,
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
            f"{MODEL_TUNNEL_URL}/embed",
            json=req.model_dump(),
            timeout=PROXY_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Model server error: {e.response.text}") from e
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach model server: {e}") from e
