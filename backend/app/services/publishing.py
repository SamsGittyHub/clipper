"""Fan a clip out to every connected account at once (PRD 8.10).

One `PublishingJob` per (clip, account) pair, executed on a thread pool so all
accounts post in parallel. The unique constraint on that pair is the
idempotency guarantee: re-running a fan-out reuses the existing row instead of
posting the same clip to the same account twice.

Per-account limits still apply -- TikTok allows 6 post requests per minute per
token, which `tiktok.RateLimiter` enforces, so "simultaneously" means every
account starts at once, not that any single account is flooded.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update as sa_update
from sqlalchemy.exc import IntegrityError

from ..config import settings
from ..db import session_scope
from ..models import Account, Clip, ClipStatus, PublishingJob, PublishStatus, utcnow
from . import runtime_config, secrets, tiktok

log = logging.getLogger("clipz.publishing")

# Refresh a little before expiry rather than waiting for a 401 mid-upload.
REFRESH_MARGIN = timedelta(minutes=5)


@dataclass
class FanOutResult:
    created: list[str]
    skipped: list[str]

    @property
    def total(self) -> int:
        return len(self.created)


def caption_for(clip: Clip) -> str:
    """Title plus hashtags -- TikTok takes a single text field."""
    parts = [(clip.title or "").strip()]
    tags = " ".join(clip.hashtags or [])
    if tags:
        parts.append(tags)
    return "\n\n".join(part for part in parts if part).strip() or "New clip"


def connected_accounts(session, platform: str = "tiktok") -> list[Account]:
    return list(
        session.scalars(
            select(Account).where(Account.platform == platform, Account.enabled.is_(True))
        )
    )


def fan_out(
    *,
    clip_ids: list[str],
    account_ids: list[str] | None = None,
    privacy_level: str | None = None,
) -> FanOutResult:
    """Create one publishing job per clip per account, then dispatch them all."""
    # An omitted privacy level falls back to the settings-panel default (which
    # itself falls back to the environment), so manual posting and auto-posting
    # honour the same privacy setting.
    publish = runtime_config.publishing()
    privacy = privacy_level or publish.privacy_level
    # The draft decision is read once per fan-out, so every post in one batch
    # agrees; it applies to both new rows and revived FAILED rows.
    draft = publish.publish_as_draft
    created: list[str] = []
    skipped: list[str] = []

    with session_scope() as session:
        accounts = connected_accounts(session)
        if account_ids:
            accounts = [account for account in accounts if account.id in account_ids]
        if not accounts:
            raise ValueError("No connected TikTok accounts. Connect one in Settings first.")

        clips = list(session.scalars(select(Clip).where(Clip.id.in_(clip_ids))))
        ready = [clip for clip in clips if clip.status == ClipStatus.READY and clip.output_path]
        if not ready:
            raise ValueError("None of those clips have been rendered yet.")

        for clip in ready:
            for account in accounts:
                existing = session.scalar(
                    select(PublishingJob).where(
                        PublishingJob.clip_id == clip.id,
                        PublishingJob.account_id == account.id,
                    )
                )
                if existing and existing.status != PublishStatus.FAILED:
                    # Already posted or in flight -- never post it twice.
                    skipped.append(existing.id)
                    continue
                if existing:
                    existing.status = PublishStatus.PENDING
                    existing.error = None
                    created.append(existing.id)
                    continue

                job = PublishingJob(
                    clip_id=clip.id,
                    account_id=account.id,
                    platform=account.platform,
                    status=PublishStatus.PENDING,
                    caption=caption_for(clip),
                    privacy_level=None if draft else privacy,
                    draft=draft,
                )
                session.add(job)
                try:
                    session.flush()
                except IntegrityError:  # pragma: no cover - concurrent fan-out
                    session.rollback()
                    skipped.append(f"{clip.id}:{account.id}")
                    continue
                created.append(job.id)

    if created:
        dispatch(created)
    return FanOutResult(created=created, skipped=skipped)


def dispatch(publish_ids: list[str]) -> None:
    """Run the publishing jobs in parallel, one thread per account job."""
    workers = max(1, min(settings.publish_concurrency, len(publish_ids)))
    log.info("Publishing %d posts across %d worker(s)", len(publish_ids), workers)

    def run() -> None:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="publish") as pool:
            futures = {pool.submit(publish_one, pid): pid for pid in publish_ids}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:  # a failed post must not stop the others
                    log.exception("Publishing job %s crashed", futures[future])

    import threading

    threading.Thread(target=run, name="publish-fanout", daemon=True).start()


# --------------------------------------------------------------------------
# One post
# --------------------------------------------------------------------------


def _fresh_access_token(account_id: str) -> str:
    """Return a usable access token, refreshing it first if it is near expiry."""
    with session_scope() as session:
        account = session.get(Account, account_id)
        if account is None:
            raise ValueError("Account no longer exists")
        expires_at = account.token_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        needs_refresh = expires_at is not None and expires_at - REFRESH_MARGIN <= datetime.now(timezone.utc)
        if not needs_refresh:
            return secrets.decrypt(account.access_token_enc)
        if not account.refresh_token_enc:
            raise ValueError("Access token expired and no refresh token is stored; reconnect the account")
        refresh_value = secrets.decrypt(account.refresh_token_enc)

    tokens = tiktok.refresh_token(refresh_value)
    with session_scope() as session:
        account = session.get(Account, account_id)
        if account is None:
            raise ValueError("Account no longer exists")
        account.access_token_enc = secrets.encrypt(tokens.access_token)
        if tokens.refresh_token:
            account.refresh_token_enc = secrets.encrypt(tokens.refresh_token)
        account.token_expires_at = tokens.expires_at
        account.refresh_expires_at = tokens.refresh_expires_at
        log.info("Refreshed access token for account %s", account_id)
    return tokens.access_token


def _set_status(publish_id: str, status: PublishStatus, **fields) -> None:
    with session_scope() as session:
        job = session.get(PublishingJob, publish_id)
        if job is None:
            return
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        if status.is_terminal:
            job.completed_at = utcnow()


def _claim(publish_id: str, attempt: int) -> bool:
    """Atomically claim a publishing job for this attempt.

    The claim is a conditional UPDATE rather than read-then-write: two threads
    dispatched for the same id (a retry racing another fan-out) must not both
    upload. A row already PUBLISHING belongs to another thread and is left
    alone; PUBLISHED/UPLOADED rows are equally unclaimable.
    """
    with session_scope() as session:
        result = session.execute(
            sa_update(PublishingJob)
            .where(
                PublishingJob.id == publish_id,
                PublishingJob.status.in_((PublishStatus.PENDING, PublishStatus.FAILED)),
            )
            .values(status=PublishStatus.PUBLISHING, attempts=attempt)
        )
        return bool(result.rowcount)


def publish_one(publish_id: str) -> None:
    """Publish a single clip to a single account, with retries."""
    attempts = max(1, settings.publish_max_attempts)

    for attempt in range(1, attempts + 1):
        if not _claim(publish_id, attempt):
            return  # another thread is already publishing this exact post
        with session_scope() as session:
            job = session.get(PublishingJob, publish_id)
            if job is None:
                return
            clip = session.get(Clip, job.clip_id)
            account_id = job.account_id
            caption = job.caption or ""
            privacy = job.privacy_level or runtime_config.publishing().privacy_level
            draft = bool(job.draft)
            output_path = clip.output_path if clip else None

        if not output_path or not Path(output_path).exists():
            _set_status(publish_id, PublishStatus.FAILED, error="The rendered clip is no longer on disk")
            return
        if not account_id:
            _set_status(publish_id, PublishStatus.FAILED, error="No account is attached to this post")
            return

        try:
            _do_publish(publish_id, account_id, Path(output_path), caption, privacy, draft)
            return
        except tiktok.TikTokError as exc:
            last = attempt >= attempts
            log.warning(
                "Publish %s attempt %d/%d failed: %s (retryable=%s)",
                publish_id, attempt, attempts, exc, exc.retryable,
            )
            if last or not exc.retryable:
                _set_status(publish_id, PublishStatus.FAILED, error=str(exc)[:500])
                _note_account_error(account_id, str(exc))
                return
            # Release the claim while backing off, so the next attempt can
            # take it again through the same conditional update.
            _set_status(publish_id, PublishStatus.PENDING)
            time.sleep(min(30.0, 2.0 ** attempt))
        except Exception as exc:
            log.exception("Publish %s failed unexpectedly", publish_id)
            _set_status(publish_id, PublishStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:500])
            return


def _note_account_error(account_id: str, message: str) -> None:
    with session_scope() as session:
        account = session.get(Account, account_id)
        if account:
            account.last_error = message[:500]


def _do_publish(
    publish_id: str, account_id: str, path: Path, caption: str, privacy: str, draft: bool
) -> None:
    access_token = _fresh_access_token(account_id)

    # Direct posts require creator_info first: it validates the token and
    # tells us which privacy levels this account actually allows -- a private
    # account cannot post publicly. Draft uploads need none of that.
    if not draft:
        info = tiktok.creator_info(access_token)
        allowed = info.get("privacy_level_options") or []
        if allowed and privacy not in allowed:
            fallback = "SELF_ONLY" if "SELF_ONLY" in allowed else allowed[0]
            log.info("Account %s does not allow %s; using %s", account_id, privacy, fallback)
            privacy = fallback

    size = path.stat().st_size
    init_kwargs: dict[str, Any] = {}
    if not draft:
        init_kwargs.update(
            privacy_level=privacy,
            disable_comment=bool(info.get("comment_disabled")),
            disable_duet=bool(info.get("duet_disabled")),
            disable_stitch=bool(info.get("stitch_disabled")),
        )
    init = tiktok.init_direct_post(
        access_token,
        title=caption,
        size=size,
        draft=draft,
        **init_kwargs,
    )
    tiktok_publish_id = init.get("publish_id")
    upload_url = init.get("upload_url")
    if not tiktok_publish_id or not upload_url:
        raise tiktok.TikTokError("TikTok did not return an upload URL")

    _set_status(publish_id, PublishStatus.PUBLISHING, external_post_id=tiktok_publish_id, privacy_level=privacy)
    tiktok.upload_file(upload_url, path)

    if draft:
        # There is no post to poll: the video is sitting in the account's
        # TikTok drafts, waiting for the creator to finish it there.
        _set_status(publish_id, PublishStatus.PUBLISHED, error=None)
        log.info("Uploaded %s to TikTok drafts", publish_id)
        return

    _set_status(publish_id, PublishStatus.UPLOADED)

    _await_publish(publish_id, access_token, tiktok_publish_id)


def _await_publish(publish_id: str, access_token: str, tiktok_publish_id: str) -> None:
    """Poll until TikTok finishes processing, or give up and leave it UPLOADED.

    A timeout is not a failure: the bytes are with TikTok and the post usually
    appears shortly after, so the row stays UPLOADED rather than FAILED.
    """
    deadline = time.time() + settings.publish_status_timeout_s

    while time.time() < deadline:
        try:
            status = tiktok.publish_status(access_token, tiktok_publish_id)
        except tiktok.TikTokError as exc:
            if not exc.retryable:
                raise
            time.sleep(settings.publish_status_poll_s)
            continue

        state = (status.get("status") or "").upper()
        if state in ("PUBLISH_COMPLETE", "SUCCESS", "PUBLISHED"):
            url = None
            ids = status.get("publicaly_available_post_id") or status.get("publicly_available_post_id") or []
            if ids:
                url = f"https://www.tiktok.com/video/{ids[0]}"
            _set_status(publish_id, PublishStatus.PUBLISHED, published_url=url, error=None)
            log.info("Published %s to TikTok (%s)", publish_id, url or tiktok_publish_id)
            return
        if state in ("FAILED", "PUBLISH_FAILED"):
            reason = status.get("fail_reason") or "TikTok rejected the post"
            _set_status(publish_id, PublishStatus.FAILED, error=str(reason)[:500])
            return
        time.sleep(settings.publish_status_poll_s)

    log.info("Publish %s still processing after timeout; leaving as UPLOADED", publish_id)
