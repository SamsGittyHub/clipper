"""Connecting TikTok accounts, and posting clips to all of them (PRD 8.10)."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets as pysecrets
import threading
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_session
from ..models import Account, Clip, ClipStatus, PublishingJob, utcnow
from ..services import publishing, runtime_config, secrets, tiktok

log = logging.getLogger("clipz.accounts")

router = APIRouter(prefix="/api/accounts", tags=["accounts"])

# Pending OAuth handshakes: state -> (code_verifier, created_at). In-memory by
# design; an interrupted connect just needs starting again.
_pending: dict[str, tuple[str, datetime]] = {}
_pending_lock = threading.Lock()
STATE_TTL_S = 600


class AccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    platform: str
    username: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    enabled: bool
    last_error: str | None = None
    token_expires_at: datetime | None = None
    created_at: datetime


class PublishAllRequest(BaseModel):
    """Post clips to every connected account at once."""

    clip_ids: list[str] | None = None      # default: every rendered clip in the job
    account_ids: list[str] | None = None   # default: every connected account
    privacy_level: str | None = None

    @field_validator("privacy_level")
    @classmethod
    def _known_privacy(cls, value: str | None) -> str | None:
        if value and value not in runtime_config.PRIVACY_LEVELS:
            raise ValueError(f"privacy_level must be one of {sorted(runtime_config.PRIVACY_LEVELS)}")
        return value


class PublishJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    clip_id: str
    account_id: str | None = None
    platform: str
    status: str
    caption: str | None = None
    privacy_level: str | None = None
    draft: bool = False
    attempts: int = 0
    external_post_id: str | None = None
    published_url: str | None = None
    error: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _enum_to_str(cls, value):
        return getattr(value, "value", value)


def _prune_states() -> None:
    now = utcnow()
    with _pending_lock:
        for state, (_, created) in list(_pending.items()):
            created_aware = created if created.tzinfo else created.replace(tzinfo=now.tzinfo)
            if (now - created_aware).total_seconds() > STATE_TTL_S:
                _pending.pop(state, None)


# ------------------------------------------------------------------- OAuth


@router.get("/tiktok/connect")
def connect_tiktok() -> dict:
    """Start the OAuth handshake; the UI opens the returned URL."""
    if not tiktok.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "TikTok is not configured. Set TIKTOK_CLIENT_KEY and "
                "TIKTOK_CLIENT_SECRET (from developers.tiktok.com) and restart."
            ),
        )
    _prune_states()
    state = pysecrets.token_urlsafe(24)
    verifier = pysecrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        if settings.tiktok_use_pkce
        else None
    )
    with _pending_lock:
        _pending[state] = (verifier, utcnow())
    return {"authorize_url": tiktok.authorize_url(state=state, code_challenge=challenge)}


@router.get("/tiktok/callback", include_in_schema=False)
def tiktok_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    """Where TikTok sends the user back. Exchanges the code and stores tokens."""
    if error:
        return _closing_page(f"TikTok declined the connection: {error_description or error}", ok=False)
    if not code or not state:
        return _closing_page("TikTok did not return an authorization code.", ok=False)

    _prune_states()
    with _pending_lock:
        pending = _pending.pop(state, None)
    if pending is None:
        # Unknown state: either expired or not initiated here. Refuse it.
        return _closing_page("This connection attempt expired. Try connecting again.", ok=False)

    verifier = pending[0] if settings.tiktok_use_pkce else None
    try:
        tokens = tiktok.exchange_code(code, code_verifier=verifier)
    except tiktok.TikTokError as exc:
        return _closing_page(f"Could not complete the connection: {exc}", ok=False)

    profile = {}
    try:
        profile = tiktok.user_info(tokens.access_token)
    except tiktok.TikTokError as exc:  # non-fatal: the account still works
        log.warning("Could not fetch TikTok profile: %s", exc)

    external_id = tokens.open_id or profile.get("open_id") or ""
    if not external_id:
        return _closing_page("TikTok did not identify the account.", ok=False)

    account = session.scalar(
        select(Account).where(Account.platform == "tiktok", Account.external_id == external_id)
    )
    if account is None:
        account = Account(platform="tiktok", external_id=external_id)
        session.add(account)

    account.access_token_enc = secrets.encrypt(tokens.access_token)
    account.refresh_token_enc = secrets.encrypt(tokens.refresh_token) if tokens.refresh_token else None
    account.scopes = tokens.scopes
    account.token_expires_at = tokens.expires_at
    account.refresh_expires_at = tokens.refresh_expires_at
    account.display_name = profile.get("display_name") or account.display_name
    account.username = profile.get("username") or account.username
    account.avatar_url = profile.get("avatar_url") or account.avatar_url
    account.enabled = True
    account.last_error = None
    session.commit()

    name = account.username or account.display_name or "your account"
    return _closing_page(f"Connected {name}. You can close this window.", ok=True)


def _closing_page(message: str, *, ok: bool) -> HTMLResponse:
    """A tiny page that reports the outcome and tells the opener to refresh.

    The success page offers "Connect another account" -- connecting several
    accounts is a loop of approve-and-repeat, so the next click is right
    there. The fetch sends the session cookie, which satisfies the API token
    gate for callers that use one.
    """
    colour = "#15803d" if ok else "#d0343e"
    next_button = (
        """<button id="another" style="margin-top:14px;padding:10px 18px;font:600 13.5px system-ui,sans-serif;
cursor:pointer;border-radius:10px;border:1px solid rgba(15,45,54,.2);background:#fff;color:#0c161a">
Connect another account</button>"""
        if ok
        else ""
    )
    return HTMLResponse(
        f"""<!doctype html><meta charset="utf-8"><title>CLIPZ</title>
<body style="margin:0;display:grid;place-items:center;height:100vh;background:#f2f6f7;
color:#0c161a;font:15px/1.6 system-ui,sans-serif">
<div style="text-align:center;max-width:32rem;padding:2rem">
<p style="color:{colour};font-weight:600">{message}</p>
<p style="color:#71828b;font-size:13px">This window can be closed.</p>
{next_button}
<p style="margin-top:14px"><a href="/" style="color:#0b7f96;font-size:13px">Back to CLIPZ</a></p>
</div>
<script>
  try {{ if (window.opener) window.opener.postMessage({{clipz: "accounts-changed"}}, "*"); }} catch (e) {{}}
  setTimeout(() => window.close(), {2500 if ok else 6000});
  const another = document.getElementById("another");
  if (another) another.addEventListener("click", async () => {{
    another.disabled = true;
    try {{
      const response = await fetch("/api/accounts/tiktok/connect");
      const body = await response.json();
      if (body.authorize_url) location.href = body.authorize_url;
      else another.disabled = false;
    }} catch (e) {{
      another.disabled = false;
    }}
  }});
</script></body>"""
    )


# ---------------------------------------------------------------- accounts


@router.get("", response_model=list[AccountOut])
def list_accounts(session: Session = Depends(get_session)) -> list[AccountOut]:
    accounts = session.scalars(select(Account).order_by(Account.created_at)).all()
    return [AccountOut.model_validate(account) for account in accounts]


@router.delete("/{account_id}")
def disconnect(account_id: str, session: Session = Depends(get_session)) -> dict:
    account = session.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    try:
        tiktok.revoke(secrets.decrypt(account.access_token_enc))
    except ValueError:  # undecryptable token: nothing to revoke
        pass
    session.delete(account)
    session.commit()
    return {"disconnected": account_id}


# --------------------------------------------------------------- publishing


@router.post("/publish/job/{job_id}", response_model=list[PublishJobOut], status_code=202)
def publish_job_everywhere(
    job_id: str,
    payload: PublishAllRequest | None = None,
    session: Session = Depends(get_session),
) -> list[PublishJobOut]:
    """Post every rendered clip in a job to every connected account at once."""
    payload = payload or PublishAllRequest()
    clip_ids = payload.clip_ids or [
        clip_id
        for clip_id in session.scalars(
            select(Clip.id).where(Clip.job_id == job_id, Clip.status == ClipStatus.READY).order_by(Clip.rank)
        )
    ]
    if not clip_ids:
        raise HTTPException(status_code=409, detail="This job has no rendered clips to post")

    try:
        result = publishing.fan_out(
            clip_ids=clip_ids,
            account_ids=payload.account_ids,
            privacy_level=payload.privacy_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    rows = session.scalars(
        select(PublishingJob).where(PublishingJob.id.in_(result.created + result.skipped))
    ).all()
    return [PublishJobOut.model_validate(row) for row in rows]


@router.get("/publish/job/{job_id}", response_model=list[PublishJobOut])
def publish_status_for_job(job_id: str, session: Session = Depends(get_session)) -> list[PublishJobOut]:
    """Live status of every post created for a job's clips."""
    rows = session.scalars(
        select(PublishingJob)
        .join(Clip, Clip.id == PublishingJob.clip_id)
        .where(Clip.job_id == job_id)
        .order_by(PublishingJob.created_at)
    ).all()
    return [PublishJobOut.model_validate(row) for row in rows]
