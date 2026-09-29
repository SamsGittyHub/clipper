"""TikTok Content Posting API client (PRD 8.10).

Implements the Direct Post flow with FILE_UPLOAD:

    creator_info/query  ->  video/init  ->  PUT chunks  ->  status/fetch

Endpoint hosts are configurable so the test suite can point the whole client
at a local stub instead of the live API.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import settings

log = logging.getLogger("clipz.tiktok")

# The default; `TIKTOK_SCOPES` overrides it. See `scopes()`.
SCOPES = ("user.info.basic", "video.publish", "video.upload")


def scopes() -> tuple[str, ...]:
    """The scopes to request, from configuration."""
    raw = (settings.tiktok_scopes or "").replace(" ", "")
    return tuple(part for part in raw.split(",") if part) or SCOPES

# TikTok accepts chunks of 5-64 MB; a file at or below the ceiling goes in one
# piece, and the final chunk of a larger file absorbs the remainder.
MIN_CHUNK = 5 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024


class TikTokError(RuntimeError):
    """An API call failed. `retryable` drives the publisher's backoff."""

    def __init__(self, message: str, *, code: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RateLimiter:
    """Per-token sliding window. TikTok allows 6 posts/min per access token."""

    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._calls: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget the window (used by tests)."""
        with self._lock:
            self._calls.clear()

    def acquire(self, key: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                calls = [t for t in self._calls.get(key, []) if now - t < self.per_seconds]
                if len(calls) < self.max_calls:
                    calls.append(now)
                    self._calls[key] = calls
                    return
                wait = self.per_seconds - (now - calls[0]) + 0.05
            log.info("Rate limit reached for account %s; waiting %.1fs", key[:12], wait)
            time.sleep(max(0.0, wait))


_post_limiter = RateLimiter(max_calls=6, per_seconds=60.0)


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None
    open_id: str
    scopes: str
    expires_at: datetime | None
    refresh_expires_at: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expiry(seconds: Any) -> datetime | None:
    try:
        return _now() + timedelta(seconds=int(seconds))
    except (TypeError, ValueError):
        return None


def is_configured() -> bool:
    return bool(settings.tiktok_client_key and settings.tiktok_client_secret)


# --------------------------------------------------------------------- OAuth


def authorize_url(*, state: str, code_challenge: str | None = None) -> str:
    params = {
        "client_key": settings.tiktok_client_key or "",
        "scope": ",".join(scopes()),
        "response_type": "code",
        "redirect_uri": settings.tiktok_redirect_uri,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{settings.tiktok_auth_base}/v2/auth/authorize/?{urlencode(params)}"


def _token_request(form: dict[str, str]) -> TokenSet:
    url = f"{settings.tiktok_api_base}/v2/oauth/token/"
    try:
        response = httpx.post(
            url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=settings.tiktok_timeout_s,
        )
    except httpx.HTTPError as exc:
        raise TikTokError(f"Could not reach TikTok: {exc}", retryable=True) from exc

    payload = _json(response)
    if response.status_code >= 400 or payload.get("error"):
        raise TikTokError(
            payload.get("error_description") or payload.get("error") or f"HTTP {response.status_code}",
            code=str(payload.get("error")),
            retryable=response.status_code >= 500,
        )

    return TokenSet(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        open_id=payload.get("open_id", ""),
        scopes=payload.get("scope", ""),
        expires_at=_expiry(payload.get("expires_in")),
        refresh_expires_at=_expiry(payload.get("refresh_expires_in")),
        raw=payload,
    )


def exchange_code(code: str, *, code_verifier: str | None = None) -> TokenSet:
    form = {
        "client_key": settings.tiktok_client_key or "",
        "client_secret": settings.tiktok_client_secret or "",
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.tiktok_redirect_uri,
    }
    if code_verifier:
        form["code_verifier"] = code_verifier
    return _token_request(form)


def refresh_token(token: str) -> TokenSet:
    """Refresh an access token. The new refresh token must be stored too."""
    return _token_request(
        {
            "client_key": settings.tiktok_client_key or "",
            "client_secret": settings.tiktok_client_secret or "",
            "grant_type": "refresh_token",
            "refresh_token": token,
        }
    )


# ---------------------------------------------------------------- API calls


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"_raw": response.text[:400]}


def _call(path: str, access_token: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{settings.tiktok_api_base}{path}"
    try:
        response = httpx.post(
            url,
            json=body or {},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            timeout=settings.tiktok_timeout_s,
        )
    except httpx.HTTPError as exc:
        raise TikTokError(f"Could not reach TikTok: {exc}", retryable=True) from exc

    payload = _json(response)
    error = payload.get("error") or {}
    code = error.get("code") if isinstance(error, dict) else str(error)

    if response.status_code == 429 or code == "rate_limit_exceeded":
        raise TikTokError("TikTok rate limit reached", code="rate_limit_exceeded", retryable=True)
    if response.status_code >= 500:
        raise TikTokError(f"TikTok server error (HTTP {response.status_code})", retryable=True)
    if code and code != "ok":
        raise TikTokError(
            error.get("message") or code,
            code=code,
            # An expired token is retryable: the publisher refreshes and retries.
            retryable=code in ("access_token_invalid", "token_expired", "internal_error"),
        )
    if response.status_code >= 400:
        raise TikTokError(f"HTTP {response.status_code}: {str(payload)[:200]}")
    return payload.get("data") or {}


def creator_info(access_token: str) -> dict[str, Any]:
    """Required before a direct post; also tells us the allowed privacy levels."""
    return _call("/v2/post/publish/creator_info/query/", access_token)


def user_info(access_token: str) -> dict[str, Any]:
    """Display name and avatar, so the UI can show who is connected."""
    url = f"{settings.tiktok_api_base}/v2/user/info/"
    try:
        response = httpx.get(
            url,
            params={"fields": "open_id,avatar_url,display_name,username"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=settings.tiktok_timeout_s,
        )
    except httpx.HTTPError as exc:
        raise TikTokError(f"Could not reach TikTok: {exc}", retryable=True) from exc
    payload = _json(response)
    return (payload.get("data") or {}).get("user") or {}


def plan_chunks(size: int) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) within TikTok's limits."""
    if size <= MAX_CHUNK:
        return size, 1
    chunk = MIN_CHUNK * 2  # 10 MB
    count = max(1, math.floor(size / chunk))
    return chunk, count


def init_direct_post(
    access_token: str,
    *,
    title: str,
    size: int,
    privacy_level: str | None = None,
    disable_comment: bool = False,
    disable_duet: bool = False,
    disable_stitch: bool = False,
    draft: bool = False,
) -> dict[str, Any]:
    """Reserve the post and get an upload URL. Rate limited per token.

    With `draft=True` the video lands in the account's TikTok drafts instead
    of being posted: the mode TikTok allows before an app is audited. Drafts
    take no privacy or interaction settings -- the creator picks those when
    they post from the TikTok app.
    """
    _post_limiter.acquire(access_token)
    chunk_size, total_chunks = plan_chunks(size)
    post_info: dict[str, Any] = {
        # TikTok counts UTF-16 runes; 2200 is the documented ceiling.
        "title": title[:2200],
    }
    if draft:
        post_info["draft"] = True
    else:
        post_info.update(
            privacy_level=privacy_level,
            disable_comment=disable_comment,
            disable_duet=disable_duet,
            disable_stitch=disable_stitch,
        )
    return _call(
        "/v2/post/publish/video/init/",
        access_token,
        {
            "post_info": post_info,
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunks,
            },
        },
    )


def upload_file(upload_url: str, path: Path) -> None:
    """PUT the video in chunks, with the final chunk absorbing the remainder."""
    size = path.stat().st_size
    chunk_size, total_chunks = plan_chunks(size)

    with path.open("rb") as handle:
        for index in range(total_chunks):
            first = index * chunk_size
            # The last chunk runs to the end of the file, however long that is.
            last = size - 1 if index == total_chunks - 1 else first + chunk_size - 1
            handle.seek(first)
            data = handle.read(last - first + 1)
            try:
                response = httpx.put(
                    upload_url,
                    content=data,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(len(data)),
                        "Content-Range": f"bytes {first}-{last}/{size}",
                    },
                    timeout=settings.tiktok_upload_timeout_s,
                )
            except httpx.HTTPError as exc:
                raise TikTokError(f"Upload failed: {exc}", retryable=True) from exc

            if response.status_code >= 500 or response.status_code == 429:
                raise TikTokError(
                    f"Upload chunk {index + 1}/{total_chunks} rejected (HTTP {response.status_code})",
                    retryable=True,
                )
            if response.status_code >= 400:
                raise TikTokError(
                    f"Upload chunk {index + 1}/{total_chunks} failed "
                    f"(HTTP {response.status_code}): {response.text[:200]}"
                )
            log.info("Uploaded chunk %d/%d (%d bytes)", index + 1, total_chunks, len(data))


def publish_status(access_token: str, publish_id: str) -> dict[str, Any]:
    return _call(
        "/v2/post/publish/status/fetch/",
        access_token,
        {"publish_id": publish_id},
    )


def revoke(access_token: str) -> None:
    """Best effort: tell TikTok the token is no longer needed."""
    try:
        httpx.post(
            f"{settings.tiktok_api_base}/v2/oauth/revoke/",
            data={
                "client_key": settings.tiktok_client_key or "",
                "client_secret": settings.tiktok_client_secret or "",
                "token": access_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=settings.tiktok_timeout_s,
        )
    except httpx.HTTPError as exc:  # pragma: no cover - network dependent
        log.warning("Token revocation failed: %s", exc)
