"""Runtime configuration set from the web UI.

The PRD keeps credentials in environment variables, and that stays the default.
This adds a second, higher-priority layer the user can edit while the app is
running: values live in `storage/runtime_config.json`, written 0600. Secrets
(the model key) are never sent back to the browser -- only a masked hint.

Precedence: runtime file > environment > built-in default.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger("clipz.runtime_config")

PROVIDERS = ("openai_compatible", "heuristic")
PRIVACY_LEVELS = (
    "PUBLIC_TO_EVERYONE",
    "MUTUAL_FOLLOW_FRIENDS",
    "FOLLOWER_OF_CREATOR",
    "SELF_ONLY",
)

_AI_KEYS = ("api_key", "base_url", "model", "provider")
_PUBLISH_KEYS = ("auto_publish", "publish_privacy_level", "publish_as_draft")
_ALLOWED = _AI_KEYS + _PUBLISH_KEYS

_lock = threading.RLock()
_cache: dict[str, Any] | None = None


def config_path() -> Path:
    return settings.storage_dir / "runtime_config.json"


@dataclass
class AISettings:
    """The resolved AI configuration, plus where each value came from."""

    api_key: str | None
    base_url: str
    model: str
    provider: str
    source: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def effective_provider(self) -> str:
        """What `get_provider()` will actually build."""
        if self.provider == "heuristic" or not self.api_key:
            return "heuristic"
        return "openai_compatible"

    def public(self) -> dict[str, Any]:
        """Safe to hand to the frontend: no key, only a hint."""
        return {
            "configured": self.configured,
            "key_hint": mask(self.api_key),
            "base_url": self.base_url,
            "model": self.model,
            "provider": self.provider,
            "effective_provider": self.effective_provider,
            "source": self.source,
            "env_key_present": bool(settings.ox_api_key),
        }


@dataclass
class PublishSettings:
    """The resolved publishing configuration, plus where each value came from."""

    auto_publish: bool
    privacy_level: str
    publish_as_draft: bool
    source: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "auto_publish": self.auto_publish,
            "privacy_level": self.privacy_level,
            "publish_as_draft": self.publish_as_draft,
            "source": self.source,
        }


def mask(key: str | None) -> str | None:
    """`sk-abcd...wxyz` -- enough to recognise, not enough to use."""
    if not key:
        return None
    if len(key) <= 10:
        return f"{key[:2]}{'*' * 6}"
    return f"{key[:6]}...{key[-4:]}"


def _load() -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        path = config_path()
        if not path.exists():
            _cache = {}
            return _cache
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _cache = {key: raw[key] for key in _ALLOWED if key in raw and raw[key] is not None}
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Ignoring unreadable runtime config at %s: %s", path, exc)
            _cache = {}
        return _cache


def _store(values: dict[str, Any]) -> None:
    """Write atomically, owner-readable only."""
    global _cache
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)
    _cache = values


# Which `Settings` field backs each editable value.
_ENV_FIELDS = {
    "api_key": "ox_api_key",
    "base_url": "ox_base_url",
    "model": "ox_model",
    "provider": "ai_provider",
    "auto_publish": "auto_publish",
    "publish_privacy_level": "default_privacy_level",
    "publish_as_draft": "publish_as_draft",
}


def current() -> AISettings:
    """Resolve runtime overrides over environment values over defaults."""
    overrides = _load()
    # `model_fields_set` holds the fields the environment (or .env) actually
    # supplied, which is what separates "env" from "default" in the UI.
    from_env = settings.model_fields_set

    def pick(key: str) -> tuple[Any, str]:
        fallback = getattr(settings, _ENV_FIELDS[key])
        if key in overrides and overrides[key] not in (None, ""):
            return overrides[key], "runtime"
        if fallback in (None, ""):
            return fallback, "default"
        return fallback, "env" if _ENV_FIELDS[key] in from_env else "default"

    api_key, key_source = pick("api_key")
    base_url, base_source = pick("base_url")
    model, model_source = pick("model")
    provider, provider_source = pick("provider")

    return AISettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=provider if provider in PROVIDERS else "openai_compatible",
        source={
            "api_key": key_source if api_key else "unset",
            "base_url": base_source,
            "model": model_source,
            "provider": provider_source,
        },
    )


def publishing() -> PublishSettings:
    """Resolve publishing settings with the runtime > environment > default order.

    Unlike the AI settings, a boolean must survive its own falseness: an
    explicit runtime `false` is an override, not an unset value, so it is
    checked with `is not None` rather than truthiness.
    """
    overrides = _load()
    from_env = settings.model_fields_set

    def pick(key: str) -> tuple[Any, str]:
        fallback = getattr(settings, _ENV_FIELDS[key])
        if key in overrides and overrides[key] is not None:
            return overrides[key], "runtime"
        if fallback is None or (isinstance(fallback, str) and fallback == ""):
            return fallback, "default"
        return fallback, "env" if _ENV_FIELDS[key] in from_env else "default"

    auto_value, auto_source = pick("auto_publish")
    privacy_value, privacy_source = pick("publish_privacy_level")
    draft_value, draft_source = pick("publish_as_draft")
    return PublishSettings(
        auto_publish=bool(auto_value),
        privacy_level=privacy_value if privacy_value in PRIVACY_LEVELS else settings.default_privacy_level,
        publish_as_draft=bool(draft_value),
        source={
            "auto_publish": auto_source,
            "privacy_level": privacy_source,
            "publish_as_draft": draft_source,
        },
    )


def update(**changes: Any) -> None:
    """Apply changes from the settings panel.

    An empty string clears an override and falls back to the environment; a
    field that is absent is left untouched, so saving the panel without
    retyping the key keeps the stored key. Bools pass through untouched --
    `False` is a value, not an unset one.
    """
    with _lock:
        values = dict(_load())
        for key, value in changes.items():
            if key not in _ALLOWED or value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
            if value == "":
                values.pop(key, None)
            else:
                values[key] = value
        _store(values)
    log.info("Runtime config updated (%s)", ", ".join(sorted(changes)) or "no changes")


def clear() -> AISettings:
    """Drop the AI overrides and revert to the environment.

    Publishing settings are deliberately kept: Clear in the settings panel is
    about the model key, and silently unsetting the auto-post toggle as a
    side effect would surprise.
    """
    global _cache
    with _lock:
        values = {key: value for key, value in _load().items() if key not in _AI_KEYS}
        _store(values)
    log.info("Runtime AI config cleared")
    return current()


def clear_publishing() -> PublishSettings:
    """Drop the publishing overrides and revert to the environment."""
    with _lock:
        values = {key: value for key, value in _load().items() if key not in _PUBLISH_KEYS}
        _store(values)
    log.info("Runtime publishing config cleared")
    return publishing()


def reset_cache() -> None:
    """Force the next read to hit disk (used by tests)."""
    global _cache
    with _lock:
        _cache = None
