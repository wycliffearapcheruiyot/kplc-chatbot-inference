import os, sys
os.environ.pop("MONGODB_URI", None)
os.environ["GATEWAY_API_KEY"] = "gw-key"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


def _client():
    import main
    return TestClient(main.app)


def test_protected_endpoints_reject_missing_or_wrong_key():
    c = _client()
    for path, body in [("/session/start", None), ("/generate", {"system_prompt": "s", "question": "q"}),
                       ("/embed", {"texts": ["a"]})]:
        assert c.post(path, json=body).status_code == 401
        assert c.post(path, json=body, headers={"X-Gateway-Key": "nope"}).status_code == 401


def test_correct_key_passes_auth():
    c = _client()
    # no session is running, so /generate gets past auth and then answers 503
    r = c.post("/generate", json={"system_prompt": "s", "question": "q"}, headers={"X-Gateway-Key": "gw-key"})
    assert r.status_code == 503


def test_status_and_health_stay_open():
    c = _client()
    assert c.get("/health").status_code == 200
    assert c.get("/session/status").status_code == 200
