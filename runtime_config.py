"""
runtime_config.py

Lets the admin panel change a service's environment variables without a
redeploy. IDENTICAL COPY in every Python service (gateway, session backend,
dataset backend, kb-builder) -- edit one, copy to the rest.

How it works
  * Overrides live in MongoDB Atlas: database `kplc_chatbot`, collection
    `service_settings`, one document per service:
        {"_id": "gateway",
         "values": {"UPSTREAM_TIMEOUT_SECONDS": "120", ...},   <- written by the admin panel
         "env": {"MONGODB_URI": "...", ...},                   <- reported by the service itself
         "env_reported_at": <datetime>, "updated_at": <datetime>}
  * cfg.get("NAME", default) returns, in order:
        1. the override stored by the admin panel   (applies within ~10 s)
        2. the real environment variable            (Render / .env)
        3. the default passed in
  * The service also REPORTS its real environment into `env`, so the admin
    panel can show what each service started with. Only names listed in
    `names` are reported.
  * MONGODB_URI is the one variable that can never live in the database (it is
    how the service finds the database), so it always comes from the real
    environment.

Fail-safe: if MongoDB is slow or down, the last good snapshot (or plain
os.environ) is used and the lookup is retried after a short back-off. A
settings problem can slow a refresh but never breaks a request.

To undo an override by hand, delete its key from `values` in Atlas, or press
"Reset to env" in the admin panel.
"""

import os
import threading
import time
from datetime import datetime, timezone

SETTINGS_DB = "kplc_chatbot"
SETTINGS_COLLECTION = "service_settings"

TTL_SECONDS = 10                 # how stale an override may be
RETRY_AFTER_FAILURE_SECONDS = 30  # back-off after a failed Mongo read
TRUE_VALUES = ("1", "true", "yes", "on")


class RuntimeConfig:
    def __init__(self, service: str, names=()):
        self.service = service
        self.names = tuple(names)
        self._db = None            # set by bind() when the app already has a database handle
        self._client = None        # lazily created from MONGODB_URI otherwise
        self._overrides: dict[str, str] = {}
        self._fetched_at = 0.0     # monotonic time of the last successful read
        self._failed_at = None     # monotonic time of the last failed read
        self._reported = None      # last env snapshot written to Mongo
        self._lock = threading.Lock()

    # --- wiring --------------------------------------------------------------

    def bind(self, db) -> None:
        """Use an existing pymongo Database (the gateway already has one)."""
        self._db = db
        self.invalidate()

    def invalidate(self) -> None:
        """Forget the cached overrides so the next read goes to MongoDB."""
        self._fetched_at = 0.0
        self._failed_at = None

    def _collection(self):
        if self._db is not None:
            return self._db[SETTINGS_COLLECTION]
        uri = os.environ.get("MONGODB_URI", "").strip()
        if not uri:
            return None
        if self._client is None:
            from pymongo import MongoClient  # local import: only needed here

            self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        return self._client[SETTINGS_DB][SETTINGS_COLLECTION]

    # --- refresh -------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force:
            if self._fetched_at and now - self._fetched_at < TTL_SECONDS:
                return
            if self._failed_at is not None and now - self._failed_at < RETRY_AFTER_FAILURE_SECONDS:
                return
        # One thread refreshes; the others keep using the current snapshot.
        if not self._lock.acquire(blocking=False):
            return
        try:
            col = self._collection()
            if col is None:
                self._fetched_at = time.monotonic()
                return
            doc = col.find_one({"_id": self.service}) or {}
            values = doc.get("values") or {}
            self._overrides = {str(k): "" if v is None else str(v) for k, v in values.items()}
            self._fetched_at = time.monotonic()
            self._failed_at = None
            self._report_env(col, doc)
        except Exception as e:  # noqa: BLE001 -- never let settings break a request
            self._failed_at = time.monotonic()
            print(f"runtime_config[{self.service}]: could not read settings ({e.__class__.__name__}); "
                  "using the last known values.", flush=True)
        finally:
            self._lock.release()

    def _report_env(self, col, doc) -> None:
        """Tells the admin panel what this service's real environment holds."""
        if not self.names:
            return
        snapshot = {n: os.environ[n] for n in self.names if n in os.environ}
        if snapshot == (doc.get("env") or {}):
            self._reported = snapshot
            return
        col.update_one(
            {"_id": self.service},
            {"$set": {"env": snapshot, "env_reported_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        self._reported = snapshot

    # --- reads ---------------------------------------------------------------

    def overrides(self) -> dict[str, str]:
        self.refresh()
        return dict(self._overrides)

    def get(self, name: str, default=None):
        self.refresh()
        if name in self._overrides:
            return self._overrides[name]
        return os.environ.get(name, default)

    def get_str(self, name: str, default: str = "") -> str:
        value = self.get(name, default)
        return default if value is None else str(value).strip()

    def get_int(self, name: str, default: int) -> int:
        try:
            return int(str(self.get(name, "")).strip())
        except ValueError:
            return default

    def get_float(self, name: str, default: float) -> float:
        try:
            return float(str(self.get(name, "")).strip())
        except ValueError:
            return default

    def get_bool(self, name: str, default: bool = False) -> bool:
        value = self.get(name)
        if value is None or str(value).strip() == "":
            return default
        return str(value).strip().lower() in TRUE_VALUES

    def environ(self) -> dict[str, str]:
        """os.environ with the overrides applied -- for subprocess env=."""
        self.refresh()
        return {**os.environ, **self._overrides}


def make(service: str, names=()) -> RuntimeConfig:
    return RuntimeConfig(service, names)
