"""Session backend picks up admin-panel overrides (stored in MongoDB) at runtime."""
import os
import sys

import mongomock
import pymongo
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["MONGODB_URI"] = "mongodb://fake"
os.environ["SESSION_WEBHOOK_SECRET"] = "env-secret"
os.environ["MODEL_TUNNEL_URL"] = "https://model.env.example/"
os.environ["KAGGLE_USERNAME"] = "env-user"
os.environ["KAGGLE_KEY"] = "env-key"
_mock = mongomock.MongoClient()
pymongo.MongoClient = lambda *a, **k: _mock

import main  # noqa: E402
from app_config import cfg  # noqa: E402

COL = _mock["kplc_chatbot"]["service_settings"]


def override(**values):
    COL.update_one({"_id": "session-backend"}, {"$set": {f"values.{k}": v for k, v in values.items()}}, upsert=True)
    cfg.invalidate()


@pytest.fixture(autouse=True)
def clean():
    COL.delete_many({})
    cfg.invalidate()
    yield


def test_env_is_used_when_there_is_no_override():
    assert main.webhook_secret() == "env-secret"
    assert main.model_tunnel_url() == "https://model.env.example"
    assert main.keep_alive() is False
    assert main.idle_timeout_seconds() == 900


def test_overrides_win_and_apply_without_restart():
    override(SESSION_WEBHOOK_SECRET="db-secret", MODEL_TUNNEL_URL="https://model.db.example///", PROXY_TIMEOUT_SECONDS="7")
    assert main.webhook_secret() == "db-secret"
    assert main.model_tunnel_url() == "https://model.db.example"
    assert main.proxy_timeout() == 7


def test_webhook_secret_check_follows_the_override():
    from fastapi import HTTPException
    main._check_webhook_secret("env-secret")
    override(SESSION_WEBHOOK_SECRET="db-secret")
    with pytest.raises(HTTPException):
        main._check_webhook_secret("env-secret")
    main._check_webhook_secret("db-secret")


def test_keep_alive_derived_values():
    override(AUTO_KEEP_ALIVE="true", RESTART_EVERY_HOURS="20")
    assert main.keep_alive() is True
    assert main.idle_timeout_seconds() == 0                 # default flips with keep-alive
    assert main.max_runtime_seconds() == int(11.5 * 3600)   # clamped to Kaggle's 12h limit
    override(KAGGLE_IDLE_TIMEOUT_SECONDS="120")
    assert main.idle_timeout_seconds() == 120
    override(KAGGLE_IDLE_TIMEOUT_SECONDS="")                 # empty falls back to the default
    assert main.idle_timeout_seconds() == 0


def test_status_endpoint_reflects_keep_alive_and_rotation():
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    assert c.get("/session/status").json()["keep_alive"] is False
    override(AUTO_KEEP_ALIVE="true", SESSIONS_PER_ACCOUNT="5",
             KAGGLE_ACCOUNTS='{"a":"ka","b":"kb"}')
    body = c.get("/session/status").json()
    assert body["keep_alive"] is True
    assert body["rotation"]["accounts"] == 2 and body["rotation"]["sessions_per_account"] == 5


def test_rotation_keeps_its_counter_when_settings_change():
    override(KAGGLE_ACCOUNTS='{"a":"ka","b":"kb"}', SESSIONS_PER_ACCOUNT="1")
    assert main.sync_rotation() is None
    main.rotation.launched()
    main.rotation.launched()
    override(KAGGLE_ACCOUNTS='{"a":"ka","b":"kb","c":"kc"}')
    main.sync_rotation()
    assert main.rotation.describe()["launches_so_far"] == 2
    assert main.rotation.describe()["accounts"] == 3


def test_bad_kaggle_accounts_is_reported_not_raised():
    override(KAGGLE_ACCOUNTS="{not json")
    assert "not valid JSON" in main.sync_rotation()


def test_launch_with_bad_accounts_marks_session_error(monkeypatch):
    override(KAGGLE_ACCOUNTS="{not json")
    main.sessions.reset_to_idle()
    with pytest.raises(main.KagglePushError):
        main._launch_session()
    assert main.sessions.snapshot()["state"] == "error"
    main.sessions.reset_to_idle()


def test_launch_pushes_the_current_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "push_secrets_dataset", lambda user, key, secrets: seen.update(user=user, key=key, secrets=secrets) or "u/ds")
    monkeypatch.setattr(main, "push_kernel", lambda path, user, key, extra_dataset_sources: seen.update(path=path))
    override(CLOUDFLARE_TUNNEL_TOKEN="tok-db", KAGGLE_KERNEL_PATH="./other", BACKEND_URL="https://me.example",
             KAGGLE_USERNAME="db-user", KAGGLE_KEY="db-key")
    main.sessions.reset_to_idle()
    assert main._launch_session() is True
    assert seen["user"] == "db-user" and seen["key"] == "db-key" and seen["path"] == "./other"
    assert seen["secrets"]["CLOUDFLARE_TUNNEL_TOKEN"] == "tok-db"
    assert seen["secrets"]["BACKEND_URL"] == "https://me.example"
    assert seen["secrets"]["IDLE_TIMEOUT_SECONDS"] == "900"
    main.sessions.reset_to_idle()


def test_supervisor_picks_up_settings_each_loop():
    sup = main.supervisor
    override(AUTO_KEEP_ALIVE="true", SUPERVISOR_INTERVAL_SECONDS="45", RESTART_RETRY_BACKOFF_SECONDS="99",
             KEEP_ALIVE_WINDOW_UTC="05:00-13:00", MODEL_TUNNEL_URL="https://t.example/")
    sup._apply()
    assert sup.enabled is True and sup.interval == 45 and sup.retry_backoff == 99
    assert sup.window == (300, 780) and sup.tunnel_url == "https://t.example"
    override(AUTO_KEEP_ALIVE="false")
    sup._apply()
    assert sup.enabled is False


def test_session_start_timeout_is_dynamic():
    override(SESSION_START_TIMEOUT_SECONDS="5")
    assert main.sessions.start_timeout_seconds == 5


def test_service_reports_its_real_env_to_the_panel():
    cfg.refresh(force=True)
    env = COL.find_one({"_id": "session-backend"})["env"]
    assert env["SESSION_WEBHOOK_SECRET"] == "env-secret"
    assert "AUTO_KEEP_ALIVE" not in env          # not set in the real env, so not reported
