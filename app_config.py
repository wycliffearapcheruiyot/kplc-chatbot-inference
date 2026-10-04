"""
app_config.py

This service's single RuntimeConfig. Settings are read through `cfg` at the
moment they are needed (admin-panel override -> real env var -> default), so
edits made in the admin panel's Environment tab apply without a redeploy.
See runtime_config.py. MONGODB_URI (real env var) is how it finds them.
"""

from runtime_config import make

# Variables reported to the admin panel so it can show what this service started with.
NAMES = [
    "MONGODB_URI", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_ACCOUNTS", "SESSIONS_PER_ACCOUNT",
    "ACCOUNT_SKIP_AFTER_FAILURES", "KAGGLE_KERNEL_PATH", "SESSION_WEBHOOK_SECRET",
    "CLOUDFLARE_TUNNEL_TOKEN", "MODEL_TUNNEL_URL", "BACKEND_URL", "SESSION_START_TIMEOUT_SECONDS",
    "PROXY_TIMEOUT_SECONDS", "AUTO_KEEP_ALIVE", "RESTART_EVERY_HOURS", "KAGGLE_IDLE_TIMEOUT_SECONDS",
    "KEEP_ALIVE_WINDOW_UTC", "SUPERVISOR_INTERVAL_SECONDS", "RESTART_RETRY_BACKOFF_SECONDS", "PYTHON_VERSION",
]

cfg = make("session-backend", NAMES)
