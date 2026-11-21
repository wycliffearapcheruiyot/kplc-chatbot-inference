"""
accounts.py

Several Kaggle accounts, used one after another.

KAGGLE_ACCOUNTS holds the credentials, either as an object
    {"username_one": "key_one", "username_two": "key_two"}
or as a list
    [{"username": "username_one", "key": "key_one"}, ...]
(if it is unset, the single KAGGLE_USERNAME / KAGGLE_KEY pair is used).

Rotation: every launch gets a number n = 0, 1, 2, ...  and runs on account
    (n // SESSIONS_PER_ACCOUNT) % len(accounts)
so each account serves SESSIONS_PER_ACCOUNT launches in a row, then the next
account takes over, and after the last one it wraps to the first.

The backend keeps no database, and a free Render service restarts and sleeps,
so the counter is not stored on Render. It travels with the Kaggle run:
serve.py is told its SESSION_NUMBER at launch and reports it back in /health
and in its /session/ready and /session/ended webhooks, which re-sync the
counter. If every one of those is lost the counter restarts at 0 (first
account), which only costs some fairness.

Skip rule: if a launch attempt ends in an error SKIP_AFTER_FAILURES times in a
row (bad key, quota used up, ...), the rotation jumps to the next account
instead of retrying the broken one for hours. A 409 (a run is still going on
that account) is never counted, since moving to another account then would
start a second GPU run.
"""

import json
import threading


def load_accounts(raw: str, fallback_username: str = "", fallback_key: str = "") -> list[dict]:
    raw = (raw or "").strip()
    accounts: list[dict] = []
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"KAGGLE_ACCOUNTS is not valid JSON: {e.msg}") from None
        if isinstance(data, dict):
            data = [{"username": u, "key": k} for u, k in data.items()]
        if not isinstance(data, list):
            raise ValueError("KAGGLE_ACCOUNTS must be a JSON object or list")
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                raise ValueError(f"KAGGLE_ACCOUNTS entry {i} must be an object")
            user = str(item.get("username", "")).strip()
            key = str(item.get("key", "")).strip()
            if not user or not key:
                raise ValueError(f"KAGGLE_ACCOUNTS entry {i} needs a username and a key")
            accounts.append({"username": user, "key": key})
    elif fallback_username and fallback_key:
        accounts.append({"username": fallback_username, "key": fallback_key})

    names = [a["username"].lower() for a in accounts]
    if len(set(names)) != len(names):
        raise ValueError("KAGGLE_ACCOUNTS lists the same username twice")
    return accounts


class Rotation:
    def __init__(self, accounts: list[dict], sessions_per_account: int = 3, skip_after_failures: int = 3):
        self.accounts = accounts
        self.sessions_per_account = max(1, int(sessions_per_account))
        self.skip_after_failures = max(1, int(skip_after_failures))
        self._n = 0            # number of the NEXT launch
        self._fail_streak = 0
        self._lock = threading.Lock()

    def configure(self, accounts: list[dict], sessions_per_account: int, skip_after_failures: int) -> None:
        """Applies settings edited in the admin panel. The launch counter and
        failure streak are kept, so a change never resets the rotation."""
        with self._lock:
            self.accounts = accounts
            self.sessions_per_account = max(1, int(sessions_per_account))
            self.skip_after_failures = max(1, int(skip_after_failures))

    def plan(self) -> tuple[int, int, dict]:
        """(launch number, account index, account) for the next launch."""
        with self._lock:
            if not self.accounts:
                raise RuntimeError("No Kaggle accounts configured.")
            idx = (self._n // self.sessions_per_account) % len(self.accounts)
            return self._n, idx, self.accounts[idx]

    def launched(self):
        with self._lock:
            self._n += 1

    def succeeded(self):
        with self._lock:
            self._fail_streak = 0

    def sync(self, session_number) -> None:
        """A run reported its SESSION_NUMBER, so the next launch is one more."""
        try:
            k = int(session_number)
        except (TypeError, ValueError):
            return
        if k < 0:
            return
        with self._lock:
            self._n = max(self._n, k + 1)

    def record_failure(self, detail: str = "") -> bool:
        """One launch attempt ended in an error. Returns True if this moved the
        rotation on to the next account."""
        if "409" in detail or "Conflict" in detail:
            return False
        with self._lock:
            self._fail_streak += 1
            if self._fail_streak < self.skip_after_failures or len(self.accounts) < 2:
                return False
            self._n = (self._n // self.sessions_per_account + 1) * self.sessions_per_account
            self._fail_streak = 0
            return True

    def describe(self) -> dict:
        """Safe to expose: no usernames or keys."""
        with self._lock:
            idx = (self._n // self.sessions_per_account) % max(1, len(self.accounts))
            return {
                "accounts": len(self.accounts),
                "sessions_per_account": self.sessions_per_account,
                "launches_so_far": self._n,
                "next_launch_account": idx + 1,
            }
