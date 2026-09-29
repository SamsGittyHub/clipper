"""The desktop launcher's port selection.

Ordinarily any free port will do. Once TikTok publishing is configured it
will not: TikTok matches the redirect URI against the one registered for the
app exactly, and sends the browser back to whatever is listening on that
port. Drifting onto a spare port means handing a live authorization code to
some unrelated program.
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

from app.config import settings

LAUNCHER = Path(__file__).resolve().parents[1] / "packaging" / "launch.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("clipz_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_tiktok(monkeypatch):
    monkeypatch.setattr(settings, "tiktok_client_key", "a-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "a-secret")
    monkeypatch.setattr(
        settings, "tiktok_redirect_uri", "http://localhost:8000/api/accounts/tiktok/callback"
    )


def test_no_port_is_pinned_when_publishing_is_not_configured(launcher, monkeypatch):
    monkeypatch.setattr(settings, "tiktok_client_key", None)
    monkeypatch.setattr(settings, "tiktok_client_secret", None)
    assert launcher.required_oauth_port() is None


def test_a_configured_app_pins_the_redirect_uri_port(launcher, configured_tiktok):
    assert launcher.required_oauth_port() == 8000


def test_half_configured_credentials_do_not_pin(launcher, monkeypatch):
    """A key without a secret cannot complete OAuth, so nothing is gained."""
    monkeypatch.setattr(settings, "tiktok_client_key", "a-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", None)
    assert launcher.required_oauth_port() is None


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("http://localhost:9000/api/accounts/tiktok/callback", 9000),
        ("http://localhost/api/accounts/tiktok/callback", 80),
        ("https://localhost/api/accounts/tiktok/callback", 443),
        ("http://127.0.0.1:8080/api/accounts/tiktok/callback", 8080),
    ],
)
def test_a_local_redirect_uri_pins_its_own_port(launcher, monkeypatch, uri, expected):
    monkeypatch.setattr(settings, "tiktok_client_key", "a-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "a-secret")
    monkeypatch.setattr(settings, "tiktok_redirect_uri", uri)
    assert launcher.required_oauth_port() == expected


@pytest.mark.parametrize(
    "uri",
    [
        "https://clips.example.com/api/accounts/tiktok/callback",
        "https://abc-123.trycloudflare.com/api/accounts/tiktok/callback",
    ],
)
def test_a_tunnelled_redirect_uri_pins_nothing(launcher, monkeypatch, uri):
    """TikTok refuses localhost, so the redirect URI is a public hostname.

    Its port describes the tunnel's front door, not this machine. Reading 443
    off it would make CLIPZ try to bind a privileged port for no reason.
    """
    monkeypatch.setattr(settings, "tiktok_client_key", "a-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "a-secret")
    monkeypatch.setattr(settings, "tiktok_redirect_uri", uri)
    assert launcher.required_oauth_port() is None


def test_port_is_free_detects_a_taken_port(launcher):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert launcher.port_is_free(taken) is False
    assert launcher.port_is_free(taken) is True


def test_free_port_still_falls_back_when_nothing_is_pinned(launcher):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        chosen = launcher.free_port(taken)
    assert chosen != taken and chosen > 0
