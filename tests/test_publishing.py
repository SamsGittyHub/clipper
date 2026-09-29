"""Connecting accounts and fanning clips out to all of them (PRD 8.10).

Everything runs against a local stub that speaks TikTok's protocol: the real
`httpx` client, the real chunking, the real OAuth exchange -- only the far end
is replaced.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from app.db import session_scope
from app.models import Account, Clip, ClipStatus, Job, JobStatus, PublishingJob, PublishStatus
from app.services import publishing, runtime_config, secrets, tiktok
from app.workers.pipeline import JobReporter, _auto_publish_if_enabled


class StubTikTok:
    """Records what it receives and replays canned responses."""

    def __init__(self):
        self.calls: list[dict] = []
        self.uploads: list[dict] = []
        self.publish_state = "PUBLISH_COMPLETE"
        self.init_error: dict | None = None
        self.privacy_options = ["PUBLIC_TO_EVERYONE", "SELF_ONLY"]
        self.token_counter = 0


@pytest.fixture
def stub_tiktok(monkeypatch):
    stub = StubTikTok()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            stub.calls.append({"path": path, "auth": self.headers.get("authorization")})
            if path == "/v2/user/info/":
                self._send({"data": {"user": {
                    "open_id": "openid-1", "display_name": "Clip Channel",
                    "username": "clipchannel", "avatar_url": "https://cdn.test/a.jpg"}}})
            else:
                self._send({}, 404)

        def do_PUT(self):
            length = int(self.headers.get("content-length", 0))
            data = self.rfile.read(length)
            stub.uploads.append({
                "range": self.headers.get("content-range"),
                "type": self.headers.get("content-type"),
                "bytes": len(data),
            })
            self.send_response(201)
            self.send_header("content-length", "0")
            self.end_headers()

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw.startswith(b"{") else parse_qs(raw.decode())
            except (ValueError, UnicodeDecodeError):
                body = {}
            stub.calls.append({"path": path, "body": body, "auth": self.headers.get("authorization")})

            if path == "/v2/oauth/token/":
                stub.token_counter += 1
                self._send({
                    "access_token": f"act.token{stub.token_counter}",
                    "refresh_token": f"rft.token{stub.token_counter}",
                    "open_id": "openid-1",
                    "scope": "user.info.basic,video.publish",
                    "expires_in": 86400,
                    "refresh_expires_in": 31536000,
                    "token_type": "Bearer",
                })
            elif path == "/v2/post/publish/creator_info/query/":
                self._send({"data": {
                    "creator_nickname": "Clip Channel",
                    "privacy_level_options": stub.privacy_options,
                    "comment_disabled": False, "duet_disabled": True, "stitch_disabled": False,
                    "max_video_post_duration_sec": 600,
                }, "error": {"code": "ok"}})
            elif path == "/v2/post/publish/video/init/":
                if stub.init_error:
                    self._send({"error": stub.init_error}, stub.init_error.pop("_http", 200))
                else:
                    self._send({"data": {
                        "publish_id": "pub-xyz",
                        "upload_url": f"http://127.0.0.1:{self.server.server_port}/upload",
                    }, "error": {"code": "ok"}})
            elif path == "/v2/post/publish/status/fetch/":
                self._send({"data": {
                    "status": stub.publish_state,
                    "publicaly_available_post_id": ["7311111111"],
                }, "error": {"code": "ok"}})
            elif path == "/v2/oauth/revoke/":
                self._send({})
            else:
                self._send({}, 404)

        def log_message(self, *args):
            pass

    # Threaded: accounts publish concurrently, as they do against the real API.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    from app.config import settings

    monkeypatch.setattr(settings, "tiktok_api_base", base)
    monkeypatch.setattr(settings, "tiktok_auth_base", base)
    monkeypatch.setattr(settings, "tiktok_client_key", "test-key")
    monkeypatch.setattr(settings, "tiktok_client_secret", "test-secret")
    monkeypatch.setattr(settings, "publish_status_poll_s", 0)
    stub.base = base
    yield stub
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_accounts():
    # The limiter is a per-token sliding window shared process-wide; without a
    # reset, tokens reused across tests would queue behind earlier tests.
    tiktok._post_limiter.reset()
    with session_scope() as session:
        for row in session.query(PublishingJob).all():
            session.delete(row)
        for row in session.query(Account).all():
            session.delete(row)
    yield


@pytest.fixture(autouse=True)
def clean_publish_settings():
    """The auto-post toggle persists in runtime_config.json; tests must not
    leak it into each other (or into the pipeline tests)."""
    runtime_config.clear_publishing()
    yield
    runtime_config.clear_publishing()


def make_account(name="clipchannel", external_id="openid-1") -> str:
    """Each account gets its own token, as it would after its own OAuth."""
    with session_scope() as session:
        account = Account(
            platform="tiktok",
            external_id=external_id,
            username=name,
            display_name=name,
            access_token_enc=secrets.encrypt(f"act.{external_id}"),
            refresh_token_enc=secrets.encrypt("rft.existing"),
        )
        session.add(account)
        session.flush()
        return account.id


def make_clip(tmp_path, rank=1, title="A clip") -> str:
    with session_scope() as session:
        job = Job(source_url="https://youtube.com/watch?v=x", status=JobStatus.COMPLETED)
        session.add(job)
        session.flush()
        path = tmp_path / f"clip_{rank}.mp4"
        path.write_bytes(b"\x00\x01\x02\x03" * 3000)
        clip = Clip(
            job_id=job.id, rank=rank, start_time=0, end_time=30, duration=30,
            title=title, hashtags=["#one", "#two"],
            status=ClipStatus.READY, output_path=str(path),
        )
        session.add(clip)
        session.flush()
        return clip.id


# ------------------------------------------------------------------- OAuth


def test_connect_requires_configuration(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tiktok_client_key", None)
    response = client.get("/api/accounts/tiktok/connect")
    assert response.status_code == 503
    assert "TIKTOK_CLIENT_KEY" in response.json()["detail"]


def test_connect_returns_an_authorize_url_with_pkce(client, stub_tiktok):
    body = client.get("/api/accounts/tiktok/connect").json()
    parsed = urlparse(body["authorize_url"])
    params = parse_qs(parsed.query)
    assert parsed.path == "/v2/auth/authorize/"
    assert params["client_key"] == ["test-key"]
    assert "video.publish" in params["scope"][0]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"][0]


def test_callback_stores_an_encrypted_token(client, stub_tiktok):
    state = parse_qs(urlparse(client.get("/api/accounts/tiktok/connect").json()["authorize_url"]).query)["state"][0]
    page = client.get(f"/api/accounts/tiktok/callback?code=abc&state={state}")
    assert page.status_code == 200
    assert "Connected clipchannel" in page.text

    [account] = client.get("/api/accounts").json()
    assert account["username"] == "clipchannel"
    # The token is never returned by the API, and is ciphertext at rest.
    assert "act.token" not in json.dumps(account)
    with session_scope() as session:
        stored = session.query(Account).one()
        assert stored.access_token_enc.startswith("gAAAA")
        assert secrets.decrypt(stored.access_token_enc) == "act.token1"


def test_callback_rejects_an_unknown_state(client, stub_tiktok):
    """A code arriving with a state we never issued is a forged callback."""
    page = client.get("/api/accounts/tiktok/callback?code=abc&state=not-ours")
    assert "expired" in page.text
    assert client.get("/api/accounts").json() == []


def test_reconnecting_updates_the_same_account(client, stub_tiktok):
    for _ in range(2):
        state = parse_qs(urlparse(client.get("/api/accounts/tiktok/connect").json()["authorize_url"]).query)["state"][0]
        client.get(f"/api/accounts/tiktok/callback?code=abc&state={state}")
    assert len(client.get("/api/accounts").json()) == 1, "same open_id must not duplicate"


# ------------------------------------------------------------- publishing


def test_one_clip_posts_to_every_connected_account(client, stub_tiktok, tmp_path):
    accounts = [make_account(f"chan{i}", f"openid-{i}") for i in range(3)]
    clip_id = make_clip(tmp_path)

    result = publishing.fan_out(clip_ids=[clip_id])
    assert result.total == 3, "one post per connected account"

    _wait_for_terminal(3)
    with session_scope() as session:
        rows = session.query(PublishingJob).all()
        assert {row.account_id for row in rows} == set(accounts)
        assert all(row.status == PublishStatus.PUBLISHED for row in rows), [
            (row.status, row.error) for row in rows
        ]
        assert all(row.published_url == "https://www.tiktok.com/video/7311111111" for row in rows)

    inits = [c for c in stub_tiktok.calls if c["path"].endswith("/video/init/")]
    assert len(inits) == 3
    # Caption carries the clip title and its hashtags.
    assert "A clip" in inits[0]["body"]["post_info"]["title"]
    assert "#one #two" in inits[0]["body"]["post_info"]["title"]
    # Interaction settings follow what the creator allows.
    assert inits[0]["body"]["post_info"]["disable_duet"] is True
    assert len(stub_tiktok.uploads) == 3


def test_every_clip_goes_to_every_account(client, stub_tiktok, tmp_path):
    for i in range(2):
        make_account(f"chan{i}", f"openid-{i}")
    clips = [make_clip(tmp_path, rank=r, title=f"Clip {r}") for r in (1, 2, 3)]

    result = publishing.fan_out(clip_ids=clips)
    assert result.total == 6, "3 clips x 2 accounts"
    _wait_for_terminal(6)

    with session_scope() as session:
        rows = session.query(PublishingJob).all()
        assert len({(r.clip_id, r.account_id) for r in rows}) == 6
        assert all(r.status == PublishStatus.PUBLISHED for r in rows), [
            (r.status, r.error) for r in rows
        ]


def test_a_repeated_fan_out_does_not_post_twice(client, stub_tiktok, tmp_path):
    make_account()
    clip_id = make_clip(tmp_path)

    first = publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(1)
    uploads_after_first = len(stub_tiktok.uploads)

    second = publishing.fan_out(clip_ids=[clip_id])
    assert second.created == [] and len(second.skipped) == 1
    assert len(stub_tiktok.uploads) == uploads_after_first, "the clip was uploaded again"


def test_the_video_is_uploaded_with_a_correct_content_range(client, stub_tiktok, tmp_path):
    make_account()
    clip_id = make_clip(tmp_path)
    with session_scope() as session:
        size = Path(session.get(Clip, clip_id).output_path).stat().st_size

    publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(1)

    [upload] = stub_tiktok.uploads
    assert upload["range"] == f"bytes 0-{size - 1}/{size}"
    assert upload["type"] == "video/mp4"
    assert upload["bytes"] == size


def test_privacy_falls_back_when_the_account_disallows_it(client, stub_tiktok, tmp_path):
    stub_tiktok.privacy_options = ["SELF_ONLY"]  # e.g. a private account
    make_account()
    clip_id = make_clip(tmp_path)

    publishing.fan_out(clip_ids=[clip_id], privacy_level="PUBLIC_TO_EVERYONE")
    _wait_for_terminal(1)

    init = next(c for c in stub_tiktok.calls if c["path"].endswith("/video/init/"))
    assert init["body"]["post_info"]["privacy_level"] == "SELF_ONLY"


def test_a_failing_account_does_not_stop_the_others(client, stub_tiktok, tmp_path, monkeypatch):
    make_account("good", "openid-good")
    bad_id = make_account("bad", "openid-bad")
    clip_id = make_clip(tmp_path)

    real_init = tiktok.init_direct_post

    def flaky_init(access_token, **kwargs):
        if access_token == "act.bad":
            raise tiktok.TikTokError("spam_risk_too_many_posts", code="spam_risk", retryable=False)
        return real_init(access_token, **kwargs)

    with session_scope() as session:
        session.get(Account, bad_id).access_token_enc = secrets.encrypt("act.bad")
    monkeypatch.setattr("app.services.publishing.tiktok.init_direct_post", flaky_init)

    publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(2)

    with session_scope() as session:
        rows = {r.account_id: r for r in session.query(PublishingJob).all()}
        assert rows[bad_id].status == PublishStatus.FAILED
        assert "spam_risk" in rows[bad_id].error
        others = [r for aid, r in rows.items() if aid != bad_id]
        assert all(r.status == PublishStatus.PUBLISHED for r in others)


def test_an_expired_token_is_refreshed_before_posting(client, stub_tiktok, tmp_path):
    from datetime import datetime, timedelta, timezone

    account_id = make_account()
    with session_scope() as session:
        session.get(Account, account_id).token_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    clip_id = make_clip(tmp_path)

    publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(1)

    assert any(c["path"] == "/v2/oauth/token/" for c in stub_tiktok.calls), "no refresh was attempted"
    with session_scope() as session:
        account = session.get(Account, account_id)
        assert secrets.decrypt(account.access_token_enc) == "act.token1", "new token not stored"
        assert session.query(PublishingJob).one().status == PublishStatus.PUBLISHED


def test_fan_out_without_accounts_is_a_clear_error(client, tmp_path):
    clip_id = make_clip(tmp_path)
    with pytest.raises(ValueError, match="No connected TikTok accounts"):
        publishing.fan_out(clip_ids=[clip_id])


def test_publish_endpoint_posts_a_whole_job(client, stub_tiktok, tmp_path):
    make_account("chan-a", "openid-a")
    make_account("chan-b", "openid-b")
    clip_id = make_clip(tmp_path)
    with session_scope() as session:
        job_id = session.get(Clip, clip_id).job_id

    response = client.post(f"/api/accounts/publish/job/{job_id}")
    assert response.status_code == 202
    assert len(response.json()) == 2

    _wait_for_terminal(2)
    statuses = [row["status"] for row in client.get(f"/api/accounts/publish/job/{job_id}").json()]
    assert statuses == ["PUBLISHED", "PUBLISHED"]


def test_publishing_a_job_with_no_rendered_clips_is_rejected(client, stub_tiktok):
    make_account()
    with session_scope() as session:
        job = Job(source_url="https://youtube.com/watch?v=x", status=JobStatus.COMPLETED)
        session.add(job)
        session.flush()
        job_id = job.id
    assert client.post(f"/api/accounts/publish/job/{job_id}").status_code == 409


def test_disconnecting_removes_the_account(client, stub_tiktok):
    account_id = make_account()
    assert client.delete(f"/api/accounts/{account_id}").status_code == 200
    assert client.get("/api/accounts").json() == []


# ------------------------------------------------------------- auto-posting


def make_completed_job_with_clips(tmp_path, n=2) -> str:
    """A job already at COMPLETED with n rendered clips, ready to auto-post."""
    with session_scope() as session:
        job = Job(source_url="https://youtube.com/watch?v=x", status=JobStatus.COMPLETED)
        session.add(job)
        session.flush()
        for rank in range(1, n + 1):
            path = tmp_path / f"clip_{rank}.mp4"
            path.write_bytes(b"\x00\x01\x02\x03" * 2000)
            session.add(
                Clip(
                    job_id=job.id, rank=rank, start_time=0, end_time=30, duration=30,
                    title=f"Clip {rank}", status=ClipStatus.READY, output_path=str(path),
                )
            )
        return job.id


def test_auto_post_disabled_does_nothing(client, tmp_path):
    make_account()
    job_id = make_completed_job_with_clips(tmp_path, n=1)

    _auto_publish_if_enabled(job_id, JobReporter(job_id))

    with session_scope() as session:
        assert session.query(PublishingJob).count() == 0


def test_auto_post_enabled_fans_out_every_clip_to_every_account(client, stub_tiktok, tmp_path):
    make_account("chan-a", "openid-a")
    make_account("chan-b", "openid-b")
    job_id = make_completed_job_with_clips(tmp_path, n=2)
    runtime_config.update(auto_publish=True)

    _auto_publish_if_enabled(job_id, JobReporter(job_id))
    _wait_for_terminal(4)  # 2 clips x 2 accounts

    with session_scope() as session:
        rows = session.query(PublishingJob).all()
        assert len(rows) == 4
        assert all(r.status == PublishStatus.PUBLISHED for r in rows), [
            (r.status, r.error) for r in rows
        ]


def test_auto_post_without_accounts_leaves_the_job_completed(client, stub_tiktok, tmp_path):
    """No connected accounts must not raise out of the completion path."""
    job_id = make_completed_job_with_clips(tmp_path, n=1)
    runtime_config.update(auto_publish=True)

    _auto_publish_if_enabled(job_id, JobReporter(job_id))

    with session_scope() as session:
        assert session.query(PublishingJob).count() == 0
        assert session.get(Job, job_id).status == JobStatus.COMPLETED


def test_auto_post_respects_the_clip_cap(client, stub_tiktok, tmp_path, monkeypatch):
    from app.config import settings

    make_account()
    job_id = make_completed_job_with_clips(tmp_path, n=3)
    runtime_config.update(auto_publish=True)
    monkeypatch.setattr(settings, "auto_publish_max_clips", 1)

    _auto_publish_if_enabled(job_id, JobReporter(job_id))
    _wait_for_terminal(1)

    with session_scope() as session:
        [row] = session.query(PublishingJob).all()
        assert session.get(Clip, row.clip_id).rank == 1, "the cap keeps the top-ranked clip"


def test_auto_post_uses_the_runtime_privacy_level(client, stub_tiktok, tmp_path):
    make_account()
    job_id = make_completed_job_with_clips(tmp_path, n=1)
    runtime_config.update(auto_publish=True, publish_privacy_level="SELF_ONLY")

    _auto_publish_if_enabled(job_id, JobReporter(job_id))
    _wait_for_terminal(1)

    init = next(c for c in stub_tiktok.calls if c["path"].endswith("/video/init/"))
    assert init["body"]["post_info"]["privacy_level"] == "SELF_ONLY"


def test_manual_fan_out_uses_the_runtime_privacy_level(client, stub_tiktok, tmp_path):
    """The settings-panel privacy default must govern manual posts too."""
    make_account()
    clip_id = make_clip(tmp_path)
    runtime_config.update(publish_privacy_level="SELF_ONLY")

    publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(1)

    init = next(c for c in stub_tiktok.calls if c["path"].endswith("/video/init/"))
    assert init["body"]["post_info"]["privacy_level"] == "SELF_ONLY"


def test_draft_uploads_skip_posting_and_privacy(client, stub_tiktok, tmp_path):
    """Draft mode lands the video in TikTok drafts: no privacy negotiation,
    no status polling -- the creator finishes the post in the TikTok app."""
    make_account()
    clip_id = make_clip(tmp_path)
    runtime_config.update(publish_as_draft=True)

    publishing.fan_out(clip_ids=[clip_id])
    _wait_for_terminal(1)

    init = next(c for c in stub_tiktok.calls if c["path"].endswith("/video/init/"))
    assert init["body"]["post_info"]["draft"] is True
    assert "privacy_level" not in init["body"]["post_info"]
    assert not any(c["path"].endswith("/status/fetch/") for c in stub_tiktok.calls)
    with session_scope() as session:
        row = session.query(PublishingJob).one()
        assert row.status == PublishStatus.PUBLISHED
        assert row.draft is True
        assert row.published_url is None


def test_draft_mode_is_read_fresh_per_fan_out(client, stub_tiktok, tmp_path):
    make_account()
    first = make_clip(tmp_path, rank=1)
    second = make_clip(tmp_path, rank=2)

    publishing.fan_out(clip_ids=[first])
    _wait_for_terminal(1)
    runtime_config.update(publish_as_draft=True)
    publishing.fan_out(clip_ids=[second])
    _wait_for_terminal(2)

    with session_scope() as session:
        rows = {r.clip_id: r for r in session.query(PublishingJob).all()}
        assert rows[first].draft is False
        assert rows[second].draft is True


def _wait_for_terminal(expected: int, timeout: float = 20.0) -> None:
    """Publishing runs on background threads; wait for them to settle.

    UPLOADED is deliberately not accepted here: it means TikTok has the bytes
    but is still processing, so treating it as finished would assert against a
    half-done post and pass or fail depending on timing.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        with session_scope() as session:
            rows = session.query(PublishingJob).all()
            if len(rows) >= expected and all(row.status.is_terminal for row in rows):
                return
        time.sleep(0.1)
    with session_scope() as session:
        states = [(r.status.value, r.error) for r in session.query(PublishingJob).all()]
    raise AssertionError(f"publishing jobs did not finish in time: {states}")


# ------------------------------------------------------------------- scopes


def test_requested_scopes_come_from_configuration(monkeypatch):
    """An app may only request scopes it has been granted.

    Asking for one it does not have fails the login with an error naming
    `client_key`, which sends you looking at the wrong thing entirely, so the
    list has to be narrowable without a code change.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "tiktok_scopes", "user.info.basic")
    assert tiktok.scopes() == ("user.info.basic",)

    monkeypatch.setattr(settings, "tiktok_scopes", "user.info.basic, video.publish")
    assert tiktok.scopes() == ("user.info.basic", "video.publish")


def test_an_empty_scope_setting_falls_back_to_the_default(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tiktok_scopes", "")
    assert tiktok.scopes() == tiktok.SCOPES


def test_the_authorize_url_carries_the_configured_scopes(monkeypatch):
    import urllib.parse as up

    from app.config import settings

    monkeypatch.setattr(settings, "tiktok_client_key", "a-key")
    monkeypatch.setattr(settings, "tiktok_scopes", "user.info.basic")
    query = up.parse_qs(up.urlparse(tiktok.authorize_url(state="s")).query)
    assert query["scope"] == ["user.info.basic"]
