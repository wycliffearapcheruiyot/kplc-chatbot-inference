"""Don't push a second Kaggle run while one is already running or queued."""
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

import kaggle_client  # noqa: E402
import main  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ('acct/kplc-chatbot-serving has status "KernelWorkerStatus.RUNNING"', "running"),
    ('acct/kplc-chatbot-serving has status "RUNNING"', "running"),
    ('has status "KernelWorkerStatus.QUEUED"', "queued"),
    ('has status "KernelWorkerStatus.COMPLETE"', "complete"),
    ('has status "KernelWorkerStatus.ERROR"', "error"),
    ('has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelled"),
    ("", "unknown"),
    ("404 Not Found", "unknown"),
])
def test_parse_run_state(text, expected):
    assert kaggle_client.parse_run_state(text) == expected


@pytest.fixture()
def fresh(monkeypatch):
    main.sessions.reset_to_idle()
    main.cfg.invalidate()
    main.sync_rotation()
    pushed = []
    monkeypatch.setattr(main, "push_secrets_dataset", lambda *a, **k: pushed.append("secrets") or "acct/ds")
    monkeypatch.setattr(main, "push_kernel", lambda *a, **k: pushed.append("kernel"))
    return pushed


def test_running_notebook_is_not_pushed_again(fresh, monkeypatch):
    monkeypatch.setattr(main, "find_active_run", lambda *a, **k: ("acct", "running"))
    assert main._launch_session() is True
    assert fresh == []                                # nothing pushed
    assert main.sessions.snapshot()["state"] == "starting"  # waits for /session/ready


def test_idle_notebook_is_pushed(fresh, monkeypatch):
    monkeypatch.setattr(main, "find_active_run", lambda *a, **k: None)
    assert main._launch_session() is True
    assert fresh == ["secrets", "kernel"]


def test_check_can_be_switched_off(fresh, monkeypatch):
    monkeypatch.setenv("SKIP_PUSH_IF_KAGGLE_RUNNING", "false")
    main.cfg.invalidate()
    monkeypatch.setattr(main, "find_active_run", lambda *a, **k: ("acct", "running"))
    assert main._launch_session() is True
    assert fresh == ["secrets", "kernel"]


def test_late_ready_from_the_running_notebook_completes_the_session(fresh, monkeypatch):
    monkeypatch.setattr(main, "find_active_run", lambda *a, **k: ("acct", "running"))
    main._launch_session()
    main.sessions.mark_ready()  # what POST /session/ready does
    assert main.sessions.snapshot()["state"] == "ready"
