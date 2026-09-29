"""AI configuration endpoints backing the settings panel.

The API key is write-only over HTTP: it can be set and cleared, and its
presence is reported as a masked hint, but no endpoint returns it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..services import runtime_config
from ..services.ai_highlights import check_connection

router = APIRouter(prefix="/api/settings", tags=["settings"])
# Base URLs for endpoints that speak the OpenAI chat-completions protocol.
# The model field is always editable; these are starting points, not a
# supported-provider list.
PRESETS = [
    {"id": "ox", "label": "Ox Alpha", "base_url": "https://api.ox.ai/v1", "model": "ox-alpha"},
    {"id": "openai", "label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    {"id": "openrouter", "label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": ""},
    {"id": "groq", "label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": ""},
    {"id": "together", "label": "Together", "base_url": "https://api.together.xyz/v1", "model": ""},
    {"id": "ollama", "label": "Ollama (local)", "base_url": "http://localhost:11434/v1", "model": ""},
    {"id": "custom", "label": "Custom", "base_url": "", "model": ""},
]


class AISettingsUpdate(BaseModel):
    """Every field is optional: omit one to leave it untouched."""

    api_key: str | None = Field(default=None, max_length=400)
    base_url: str | None = Field(default=None, max_length=400)
    model: str | None = Field(default=None, max_length=200)
    provider: Literal["openai_compatible", "heuristic"] | None = None

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value


class ConnectionTest(BaseModel):
    """Test values before saving them; omitted fields use what is saved."""

    api_key: str | None = Field(default=None, max_length=400)
    base_url: str | None = Field(default=None, max_length=400)
    model: str | None = Field(default=None, max_length=200)


@router.get("/ai")
def get_ai_settings() -> dict:
    return {**runtime_config.current().public(), "presets": PRESETS}


@router.put("/ai")
def update_ai_settings(payload: AISettingsUpdate) -> dict:
    """Save runtime overrides. An empty string clears one back to the env."""
    runtime_config.update(**payload.model_dump(exclude_unset=True))
    return {**runtime_config.current().public(), "presets": PRESETS}


@router.delete("/ai")
def clear_ai_settings() -> dict:
    """Forget everything entered here and fall back to the environment."""
    return {**runtime_config.clear().public(), "presets": PRESETS}


@router.post("/ai/test")
def test_ai_settings(payload: ConnectionTest) -> dict:
    """Make one tiny live call so the user finds out now, not mid-job."""
    saved = runtime_config.current()
    api_key = (payload.api_key or "").strip() or saved.api_key
    base_url = (payload.base_url or "").strip() or saved.base_url
    model = (payload.model or "").strip() or saved.model

    if not api_key:
        raise HTTPException(status_code=400, detail="No API key to test. Enter one first.")
    if not model:
        raise HTTPException(status_code=400, detail="No model configured. Enter a model name.")

    ok, message = check_connection(api_key=api_key, base_url=base_url, model=model)
    return {"ok": ok, "message": message, "model": model, "base_url": base_url}


# ------------------------------------------------------- publishing (8.10)


class PublishingSettingsUpdate(BaseModel):
    """Every field is optional: omit one to leave it untouched."""

    auto_publish: bool | None = None
    privacy_level: str | None = None
    publish_as_draft: bool | None = None

    @field_validator("privacy_level")
    @classmethod
    def _known_privacy(cls, value: str | None) -> str | None:
        # One shared constant governs this endpoint and the publish endpoints,
        # so the accepted levels can never drift apart.
        if value and value not in runtime_config.PRIVACY_LEVELS:
            raise ValueError(
                f"privacy_level must be one of {sorted(runtime_config.PRIVACY_LEVELS)}"
            )
        return value


@router.get("/publishing")
def get_publishing_settings() -> dict:
    return runtime_config.publishing().public()


@router.put("/publishing")
def update_publishing_settings(payload: PublishingSettingsUpdate) -> dict:
    """Save the auto-post toggle and default privacy level."""
    changes = payload.model_dump(exclude_unset=True)
    if "privacy_level" in changes:
        changes["publish_privacy_level"] = changes.pop("privacy_level")
    runtime_config.update(**changes)
    return runtime_config.publishing().public()
