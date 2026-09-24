"""
serve.py

Section 4 of the deployment guide: the Kaggle serving script pushed via
`kaggle kernels push -p kaggle_notebook` (called by session_manager.py's
start_session()). Runs as a Kaggle "Save & Run All" batch execution.

On every run, in order:
    1. Loads Qwen3-4B (generation) and a small sentence-embedding model
       (retrieval) from the attached Kaggle Dataset / cache.
    2. Starts a local FastAPI server on localhost:8000 with POST /generate
       and POST /embed, matching the shapes model_client.py (backend) expects.
    3. Opens a named Cloudflare Tunnel (cloudflared, token-based) pointed at
       localhost:8000, so it comes up on the same fixed public hostname
       every run (hostname routing is configured once in the Cloudflare
       dashboard, not here).
    4. Once the local server and tunnel are both confirmed up, POSTs to the
       backend's /session/ready webhook (with the shared secret header).
    5. Runs an idle-watchdog: any /generate or /embed call resets the idle
       clock. After IDLE_TIMEOUT_SECONDS with no calls, it stops cloudflared,
       POSTs /session/ended, and exits — ending this Kaggle session and
       freeing the weekly GPU quota.

Required Kaggle Secrets (Add-ons -> Secrets, in the Kaggle notebook editor):
    BACKEND_URL              e.g. https://kplc-chatbot-backend.onrender.com
    SESSION_WEBHOOK_SECRET   same value as Render's SESSION_WEBHOOK_SECRET
    CLOUDFLARE_TUNNEL_TOKEN  token for a named tunnel already routed (in the
                              Cloudflare Zero Trust dashboard) to
                              http://localhost:8000

See README.md in this folder for the one-time setup of the Kaggle Dataset,
the named tunnel, and these secrets.
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# sentence-transformers isn't in Kaggle's base image — install it on first
# run rather than requiring a separate setup step.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"],
        check=True,
    )
    from sentence_transformers import SentenceTransformer

# --- config -----------------------------------------------------------------

LOCAL_PORT = 8000
IDLE_TIMEOUT_SECONDS = 15 * 60
IDLE_CHECK_INTERVAL_SECONDS = 30
READY_CONFIRM_GRACE_SECONDS = 10
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 512
CLOUDFLARED_LOG = "/kaggle/working/cloudflared.log"

# --- config source: private dataset (primary), Kaggle Secrets/env (fallback) --
#
# The Render backend pushes BACKEND_URL / SESSION_WEBHOOK_SECRET /
# CLOUDFLARE_TUNNEL_TOKEN into a private Kaggle Dataset on every
# /session/start call and attaches it to this kernel's dataset_sources
# (see kaggle_client.py). That's the PRIMARY source, because it's fully
# automated and survives CLI pushes. Kaggle Secrets (Add-ons -> Secrets)
# are kept as a manual fallback only, since their kernel binding is NOT
# reliably preserved across CLI/API pushes.

def _load_secrets_from_dataset() -> dict:
    """Searches /kaggle/input for a secrets.json written by the backend's
    push_secrets_dataset(). Returns {} if not found (falls through to the
    Kaggle Secrets / env var path below)."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return {}
    for match in input_root.rglob("secrets.json"):
        try:
            return json.loads(match.read_text())
        except Exception as e:
            print(f"Found {match} but couldn't parse it: {e}", file=sys.stderr)
    return {}


try:
    from kaggle_secrets import UserSecretsClient

    _secrets_client = UserSecretsClient()

    def _get_kaggle_secret(name: str) -> str:
        try:
            return _secrets_client.get_secret(name)
        except Exception:
            return ""

except ImportError:
    def _get_kaggle_secret(name: str) -> str:
        return ""


_dataset_secrets = _load_secrets_from_dataset()


def get_secret(name: str) -> str:
    # Priority: dataset file (automated) > Kaggle Secrets UI (manual,
    # unreliable across pushes) > plain env var (local testing).
    return _dataset_secrets.get(name) or _get_kaggle_secret(name) or os.environ.get(name, "")


BACKEND_URL = get_secret("BACKEND_URL").rstrip("/")
SESSION_WEBHOOK_SECRET = get_secret("SESSION_WEBHOOK_SECRET")
_raw_token = get_secret("CLOUDFLARE_TUNNEL_TOKEN").strip().strip("'\"")
# Tolerate pasting Cloudflare's whole install command
# ("cloudflared.exe service install eyJ..."): the token is the last word.
CLOUDFLARE_TUNNEL_TOKEN = _raw_token.split()[-1].strip("'\"") if _raw_token else ""
if CLOUDFLARE_TUNNEL_TOKEN:
    # Safe diagnostic: never prints the token itself.
    print(
        f"Tunnel token: {len(CLOUDFLARE_TUNNEL_TOKEN)} chars, "
        f"starts with eyJ: {CLOUDFLARE_TUNNEL_TOKEN.startswith('eyJ')}",
        flush=True,
    )

for name, value in [
    ("BACKEND_URL", BACKEND_URL),
    ("SESSION_WEBHOOK_SECRET", SESSION_WEBHOOK_SECRET),
    ("CLOUDFLARE_TUNNEL_TOKEN", CLOUDFLARE_TUNNEL_TOKEN),
]:
    if not value:
        print(
            f"FATAL: '{name}' not found in the secrets dataset, Kaggle "
            f"Secrets, or env vars. See README.md.",
            file=sys.stderr,
        )
        sys.exit(1)

# --- find the attached model dataset -----------------------------------------

def find_model_dir() -> str:
    """The Kaggle Dataset from the one-time setup mounts read-only under
    /kaggle/input/. The mount layout can differ between Kaggle versions, so
    instead of assuming a path we search for the folder that holds config.json
    next to .safetensors weights."""
    input_root = Path("/kaggle/input")
    configs = sorted(input_root.rglob("config.json")) if input_root.exists() else []
    candidates = [c.parent for c in configs if any(c.parent.glob("*.safetensors"))]
    if not candidates:
        raise RuntimeError(
            "No model weights found under /kaggle/input/. Make sure the weights "
            "Dataset is listed in kernel-metadata.json's dataset_sources."
        )
    if len(candidates) > 1:
        print(f"Multiple model folders found, using the first: {candidates}", file=sys.stderr)
    return str(candidates[0])


# --- global state -------------------------------------------------------------

_last_activity_lock = threading.Lock()
_last_activity_time = time.time()
_shutting_down = threading.Event()


def _touch_activity():
    global _last_activity_time
    with _last_activity_lock:
        _last_activity_time = time.time()


def _seconds_since_activity() -> float:
    with _last_activity_lock:
        return time.time() - _last_activity_time


# --- load models ---------------------------------------------------------------
# Loaded inside main() (not at import time) so that any failure here -- bad
# dataset path, out of memory -- goes through the handler that tells the
# backend the session ended, instead of leaving the dashboard on "starting".

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
embedder = None


def load_models():
    global tokenizer, model, embedder
    print("Loading models...", flush=True)
    load_start = time.time()

    model_dir = find_model_dir()
    print(f"Loading Qwen3-4B from {model_dir} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # float16, not bfloat16: Kaggle's T4/P100 GPUs have no native bf16 support.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME}) ...", flush=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f"Models loaded in {time.time() - load_start:.1f}s", flush=True)


# --- FastAPI app ----------------------------------------------------------------

app = FastAPI(title="KPLC Chatbot Model Server")


class GenerateRequest(BaseModel):
    system_prompt: str
    question: str


class GenerateResponse(BaseModel):
    answer: str


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    _touch_activity()
    try:
        messages = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise HTTPException(status_code=500, detail="Model produced an empty response.")
        return GenerateResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    _touch_activity()
    if not req.texts:
        return EmbedResponse(embeddings=[])
    try:
        vectors = embedder.encode(req.texts, normalize_embeddings=True)
        return EmbedResponse(embeddings=[v.tolist() for v in vectors])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}") from e


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=LOCAL_PORT, log_level="info")


# --- cloudflared tunnel -----------------------------------------------------------

def ensure_cloudflared() -> str:
    """Downloads the cloudflared binary if it isn't already present."""
    binary = Path("/kaggle/working/cloudflared")
    if not binary.exists():
        print("Downloading cloudflared...", flush=True)
        subprocess.run(
            [
                "wget", "-q",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "-O", str(binary),
            ],
            check=True,
        )
        binary.chmod(0o755)
    return str(binary)


def start_tunnel(cloudflared_path: str) -> subprocess.Popen:
    print("Starting Cloudflare Tunnel...", flush=True)
    # Log to a file, NOT a pipe: nobody reads the pipe, so once its buffer
    # filled up cloudflared would block and the tunnel would stall.
    log = open(CLOUDFLARED_LOG, "w")
    proc = subprocess.Popen(
        [
            cloudflared_path, "tunnel",
            "--no-autoupdate",
            "run",
            "--token", CLOUDFLARE_TUNNEL_TOKEN,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def print_cloudflared_log_tail(lines: int = 20):
    try:
        tail = Path(CLOUDFLARED_LOG).read_text().splitlines()[-lines:]
        print("--- cloudflared log (tail) ---\n" + "\n".join(tail), file=sys.stderr, flush=True)
    except Exception:
        pass


# --- webhooks back to the Render backend --------------------------------------

def call_webhook(path: str, extra: dict | None = None):
    # The Render backend may be asleep (free tier cold start takes 30-60s),
    # so use a generous timeout and retry a few times.
    url = f"{BACKEND_URL}{path}"
    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                url,
                json=extra or {},
                headers={"X-Webhook-Secret": SESSION_WEBHOOK_SECRET},
                timeout=90,
            )
            print(f"POST {path} -> {resp.status_code}", flush=True)
            return
        except requests.exceptions.RequestException as e:
            print(
                f"Failed to call webhook {path} (attempt {attempt}/{attempts}): {e}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < attempts:
                time.sleep(5)


# --- idle watchdog -----------------------------------------------------------------

def idle_watchdog(tunnel_proc: subprocess.Popen):
    while not _shutting_down.is_set():
        time.sleep(IDLE_CHECK_INTERVAL_SECONDS)
        idle_for = _seconds_since_activity()
        if tunnel_proc.poll() is not None:
            print("cloudflared exited unexpectedly, shutting down.", flush=True)
            break
        if idle_for >= IDLE_TIMEOUT_SECONDS:
            print(f"Idle for {idle_for:.0f}s, shutting down.", flush=True)
            break
    shutdown(tunnel_proc)


def shutdown(tunnel_proc: subprocess.Popen):
    if _shutting_down.is_set():
        return
    _shutting_down.set()
    print("Stopping tunnel and reporting session ended...", flush=True)
    try:
        tunnel_proc.terminate()
        tunnel_proc.wait(timeout=10)
    except Exception:
        tunnel_proc.kill()
    call_webhook("/session/ended")
    print("Session ended, exiting.", flush=True)
    os._exit(0)


# --- main --------------------------------------------------------------------------

def main():
    load_models()
    cloudflared_path = ensure_cloudflared()
    tunnel_proc = start_tunnel(cloudflared_path)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server + tunnel a moment to come up, then sanity-check both
    # before telling the backend we're ready.
    time.sleep(READY_CONFIRM_GRACE_SECONDS)

    if tunnel_proc.poll() is not None:
        print("cloudflared failed to start — check CLOUDFLARE_TUNNEL_TOKEN.", file=sys.stderr)
        print_cloudflared_log_tail()
        call_webhook("/session/ended")
        sys.exit(1)

    try:
        r = requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Local server health check failed: {e}", file=sys.stderr)
        tunnel_proc.terminate()
        call_webhook("/session/ended")
        sys.exit(1)

    _touch_activity()  # start the idle clock from "now", not from model-load time
    call_webhook("/session/ready")
    print("Session ready. Serving until idle timeout or manual stop.", flush=True)

    idle_watchdog(tunnel_proc)  # blocks until idle timeout or tunnel death


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        call_webhook("/session/ended")
        sys.exit(1)