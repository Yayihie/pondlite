import base64

import pytest
from fastapi.testclient import TestClient

from pondlite import serve, web
from pondlite.auth import auth_enabled, require_production_config


def test_production_requires_complete_auth_config(monkeypatch):
    monkeypatch.setenv("PONDLITE_ENV", "production")
    monkeypatch.delenv("PONDLITE_USERNAME", raising=False)
    monkeypatch.delenv("PONDLITE_PASSWORD", raising=False)
    monkeypatch.delenv("PONDLITE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PONDLITE_USERNAME"):
        require_production_config()
    monkeypatch.setenv("PONDLITE_USERNAME", "founder")
    monkeypatch.setenv("PONDLITE_PASSWORD", "a-long-test-password")
    monkeypatch.setenv("PONDLITE_SECRET_KEY", "a-long-test-session-secret")
    require_production_config()
    assert auth_enabled()


def test_flask_login_logout_and_csrf(monkeypatch):
    monkeypatch.setenv("PONDLITE_USERNAME", "founder")
    monkeypatch.setenv("PONDLITE_PASSWORD", "a-long-test-password")
    monkeypatch.setenv("PONDLITE_SECRET_KEY", "a-long-test-session-secret")
    client = web.app.test_client()
    response = client.get("/ingest")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    assert client.get("/login").status_code == 200
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    assert client.post("/login", data={
        "username": "founder", "password": "wrong", "_csrf_token": token,
    }).status_code == 401
    response = client.post("/login", data={
        "username": "founder", "password": "a-long-test-password", "_csrf_token": token,
    })
    assert response.status_code == 302
    assert client.get("/ingest").status_code == 200
    assert client.post("/ingest", data={"name": "demo", "data": "{}"}).status_code == 400
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    assert client.post("/logout", data={"_csrf_token": token}).status_code == 302
    assert client.get("/ingest").status_code == 302


def test_api_auth_and_public_health(monkeypatch):
    monkeypatch.setenv("PONDLITE_USERNAME", "founder")
    monkeypatch.setenv("PONDLITE_PASSWORD", "a-long-test-password")
    client = TestClient(serve.app)
    assert client.get("/health").status_code == 200
    assert client.post("/v1/runs", json={"run_dir": "/tmp"}).status_code == 401
    credentials = base64.b64encode(b"founder:a-long-test-password").decode()
    response = client.post(
        "/v1/runs", json={"run_dir": "/tmp"},
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 400


def test_security_headers_present():
    response = web.app.test_client().get("/ingest")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
