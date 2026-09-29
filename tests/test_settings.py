"""Runtime AI configuration and the settings endpoints."""

from __future__ import annotations

import json
import stat

import pytest

from app.services import runtime_config


@pytest.fixture(autouse=True)
def clean_runtime_config():
    runtime_config.clear()
    runtime_config.clear_publishing()
    yield
    runtime_config.clear()
    runtime_config.clear_publishing()


# --------------------------------------------------------------- resolution


def test_unconfigured_state_falls_back_to_the_heuristic(client):
    body = client.get("/api/settings/ai").json()
    assert body["source"]["base_url"] == "default"
    assert body["source"]["provider"] == "env"  # AI_PROVIDER is set for tests
    assert body["configured"] is False
    assert body["key_hint"] is None
    assert body["effective_provider"] == "heuristic"
    assert any(preset["id"] == "openai" for preset in body["presets"])


def test_saved_key_is_masked_and_never_returned(client):
    secret = "sk-test-1234567890-abcdefghijklmnop"
    body = client.put("/api/settings/ai", json={"api_key": secret, "model": "some-model"}).json()

    assert body["configured"] is True
    assert body["key_hint"] == "sk-tes...mnop"
    assert body["source"]["api_key"] == "runtime"
    assert secret not in json.dumps(body)
    # Nor on any other endpoint.
    assert secret not in json.dumps(client.get("/api/settings/ai").json())
    assert secret not in json.dumps(client.get("/api/health").json())
    assert secret not in json.dumps(client.get("/api/config").json())


def test_saving_without_retyping_the_key_keeps_it(client):
    client.put("/api/settings/ai", json={"api_key": "sk-keep-this-one-please"})
    body = client.put("/api/settings/ai", json={"model": "another-model"}).json()
    assert body["configured"] is True
    assert body["model"] == "another-model"


def test_empty_string_clears_a_single_override(client):
    client.put("/api/settings/ai", json={"api_key": "sk-remove-me-now-ok", "model": "m"})
    body = client.put("/api/settings/ai", json={"api_key": ""}).json()
    assert body["configured"] is False
    assert body["model"] == "m", "clearing the key should not touch other fields"


def test_delete_reverts_everything_to_the_environment(client):
    client.put(
        "/api/settings/ai",
        json={"api_key": "sk-abc-def-ghi-jkl", "base_url": "https://example.test/v1", "model": "m"},
    )
    body = client.delete("/api/settings/ai").json()
    assert body["configured"] is False
    assert body["base_url"] == "https://api.ox.ai/v1"  # the built-in default
    assert body["source"]["base_url"] == "default"


def test_heuristic_toggle_wins_over_a_saved_key(client):
    """The toggle overrides both a saved key and AI_PROVIDER from the env."""
    active = client.put(
        "/api/settings/ai",
        json={"api_key": "sk-still-saved-here", "provider": "openai_compatible"},
    ).json()
    assert active["effective_provider"] == "openai_compatible"

    body = client.put("/api/settings/ai", json={"provider": "heuristic"}).json()
    assert body["configured"] is True, "the key stays saved, it is just unused"
    assert body["effective_provider"] == "heuristic"
    assert client.get("/api/health").json()["ai_provider"] == "heuristic"


def test_invalid_base_url_is_rejected(client):
    assert client.put("/api/settings/ai", json={"base_url": "ftp://nope"}).status_code == 422
    assert client.put("/api/settings/ai", json={"provider": "magic"}).status_code == 422


# ------------------------------------------------------------ file storage


def test_the_key_file_is_owner_readable_only(client):
    client.put("/api/settings/ai", json={"api_key": "sk-on-disk-secret-value"})
    path = runtime_config.config_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_corrupt_config_file_is_ignored_not_fatal(client):
    runtime_config.config_path().write_text("{ not json", encoding="utf-8")
    runtime_config.reset_cache()
    assert client.get("/api/settings/ai").json()["configured"] is False


# --------------------------------------------------------- provider wiring


def test_get_provider_uses_the_runtime_settings(client, monkeypatch):
    from app.services import ai_highlights

    built = {}

    class FakeClient:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    client.put(
        "/api/settings/ai",
        json={
            "api_key": "sk-runtime-key-value-here",
            "base_url": "https://runtime.test/v1",
            "model": "runtime-model",
            "provider": "openai_compatible",
        },
    )

    provider = ai_highlights.get_provider()
    assert isinstance(provider, ai_highlights.OpenAICompatibleProvider)
    assert provider.model == "runtime-model"
    assert built["api_key"] == "sk-runtime-key-value-here"
    assert built["base_url"] == "https://runtime.test/v1"


# ------------------------------------------------------- connection testing


def test_connection_test_requires_a_key(client):
    response = client.post("/api/settings/ai/test", json={})
    assert response.status_code == 400
    assert "key" in response.json()["detail"].lower()


def test_connection_test_reports_success(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.settings.check_connection",
        lambda **kwargs: (True, f"Connected to {kwargs['model']}."),
    )
    body = client.post(
        "/api/settings/ai/test",
        json={"api_key": "sk-unsaved-key-value", "model": "trial-model"},
    ).json()
    assert body["ok"] is True
    assert "trial-model" in body["message"]


def test_connection_test_reports_failure_without_leaking_the_key(client):
    secret = "sk-bad-key-that-should-not-leak"
    body = client.post(
        "/api/settings/ai/test",
        json={"api_key": secret, "base_url": "http://127.0.0.1:9/v1", "model": "m"},
    ).json()
    assert body["ok"] is False
    assert secret not in json.dumps(body)


def test_check_connection_scrubs_the_key_from_errors():
    from app.services.ai_highlights import check_connection

    secret = "sk-leaky-key-abcdefghijklmnop"
    ok, message = check_connection(
        api_key=secret, base_url="http://127.0.0.1:9/v1", model="m", timeout=2.0
    )
    assert ok is False
    assert secret not in message


# ----------------------------------------------------- publishing settings


def test_publishing_settings_default_to_the_environment(client):
    body = client.get("/api/settings/publishing").json()
    assert body["auto_publish"] is False
    assert body["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert body["source"]["auto_publish"] == "default"


def test_publishing_settings_round_trip(client):
    body = client.put(
        "/api/settings/publishing",
        json={"auto_publish": True, "privacy_level": "SELF_ONLY"},
    ).json()
    assert body["auto_publish"] is True
    assert body["privacy_level"] == "SELF_ONLY"
    assert body["source"]["privacy_level"] == "runtime"

    # Omitted fields are left untouched, and a False is a value, not an unset
    # one: turning the toggle off must survive the round trip.
    body = client.put("/api/settings/publishing", json={"auto_publish": False}).json()
    assert body["auto_publish"] is False
    assert body["privacy_level"] == "SELF_ONLY"
    assert client.get("/api/settings/publishing").json()["auto_publish"] is False


def test_publishing_settings_reject_an_unknown_privacy_level(client):
    response = client.put("/api/settings/publishing", json={"privacy_level": "FRIENDS_ONLY"})
    assert response.status_code == 422


def test_draft_mode_round_trip(client):
    body = client.put("/api/settings/publishing", json={"publish_as_draft": True}).json()
    assert body["publish_as_draft"] is True
    assert client.get("/api/settings/publishing").json()["publish_as_draft"] is True

    body = client.put("/api/settings/publishing", json={"publish_as_draft": False}).json()
    assert body["publish_as_draft"] is False
    assert body["source"]["publish_as_draft"] == "runtime"


def test_clearing_ai_settings_keeps_publishing_settings(client):
    client.put("/api/settings/publishing", json={"auto_publish": True})
    client.put("/api/settings/ai", json={"api_key": "sk-cleared-anyway-123456"})
    assert client.get("/api/settings/ai").json()["configured"] is True

    client.delete("/api/settings/ai")
    assert client.get("/api/settings/ai").json()["configured"] is False
    assert client.get("/api/settings/publishing").json()["auto_publish"] is True
