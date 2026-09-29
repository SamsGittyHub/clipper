"""The optional API token (PRD 12: keys must not be reachable by anyone).

Off by default, because the app is one user on localhost. Once `API_TOKEN` is
set the whole API closes, which is what makes it safe to bind anywhere else --
the settings panel holds a model API key and `/api/quit` stops the server.
"""

from __future__ import annotations

import pytest

from app.api import auth
from app.config import settings


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "s3cret-token-value")
    return "s3cret-token-value"


def test_the_api_is_open_when_no_token_is_configured(client):
    assert client.get("/api/health").status_code == 200
    state = client.get("/api/auth").json()
    assert state == {"required": False, "authenticated": True}


def test_every_api_route_is_closed_once_a_token_is_set(client, token):
    for path in ("/api/health", "/api/config", "/api/jobs", "/api/settings/ai"):
        response = client.get(path)
        assert response.status_code == 401, path
        assert response.json()["error_code"] == "ERROR_UNAUTHORISED"
    assert client.get("/api/health").headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer s3cret-token-value"},
        {"X-API-Key": "s3cret-token-value"},
    ],
)
def test_the_token_is_accepted_from_either_header(client, token, headers):
    assert client.get("/api/health", headers=headers).status_code == 200


def test_a_wrong_token_is_refused(client, token):
    assert client.get("/api/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/auth", json={"token": "wrong"}).status_code == 401


def test_signing_in_sets_a_cookie_that_unlocks_the_browser(client, token):
    assert client.get("/api/health").status_code == 401

    body = client.post("/api/auth", json={"token": token}).json()
    assert body == {"required": True, "authenticated": True}

    # The TestClient keeps the cookie, exactly as a browser would.
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/auth").json()["authenticated"] is True

    client.delete("/api/auth")
    assert client.get("/api/health").status_code == 401


def test_the_state_endpoint_stays_reachable_so_the_ui_can_ask(client, token):
    """Asking whether a token is needed must not itself need one."""
    body = client.get("/api/auth").json()
    assert body == {"required": True, "authenticated": False}


def test_the_tiktok_callback_stays_open(client, token):
    """TikTok redirects the browser here and cannot send our header.

    It carries its own one-time `state`, which is what actually authorises it.
    """
    assert "/api/accounts/tiktok/callback" in auth.EXEMPT_PATHS
    response = client.get("/api/accounts/tiktok/callback", follow_redirects=False)
    assert response.status_code != 401


def test_the_frontend_itself_is_not_gated(client, token):
    """A 401 on the page would leave nowhere to type the token."""
    assert client.get("/").status_code == 200


def test_health_reports_whether_a_token_is_required(client, token):
    body = client.get("/api/health", headers={"X-API-Key": token}).json()
    assert body["auth_required"] is True
