"""HTTP surface (PRD 10)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_worker(monkeypatch):
    """Accept jobs without running the pipeline."""
    enqueued: list[str] = []
    monkeypatch.setattr(
        "app.api.jobs.get_queue",
        lambda: type("Q", (), {"enqueue": staticmethod(enqueued.append), "depth": 0})(),
    )
    return enqueued


def create_job(client, **overrides):
    payload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "clip_count": 3,
        "max_duration": 45,
        "caption_style": "dynamic",
        **overrides,
    }
    return client.post("/api/jobs", json=payload)


# -------------------------------------------------------------------- jobs


def test_create_job_returns_a_queued_status_payload(client, no_real_worker):
    response = create_job(client)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["progress"] == 0
    assert body["job_id"].startswith("job_")
    assert no_real_worker == [body["job_id"]]


def test_status_endpoint_matches_the_documented_shape(client):
    job_id = create_job(client).json()["job_id"]
    body = client.get(f"/api/jobs/{job_id}").json()
    for field in ("job_id", "status", "progress", "message"):
        assert field in body


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"url": "not-a-url"}, "url"),
        ({"url": "https://vimeo.com/123"}, "url"),
        ({"clip_count": 0}, "clip_count"),
        ({"clip_count": 99}, "clip_count"),
    ],
)
def test_invalid_requests_are_rejected(client, payload, field):
    response = create_job(client, **payload)
    assert response.status_code == 422


def test_max_duration_is_capped_at_the_configured_limit(client):
    body = create_job(client, max_duration=6000).json()
    assert body["max_duration"] == 90.0


def test_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/job_missing").status_code == 404
    assert client.get("/api/jobs/job_missing/clips").status_code == 404


def test_job_listing_is_newest_first(client):
    first = create_job(client).json()["job_id"]
    second = create_job(client).json()["job_id"]
    listed = [job["job_id"] for job in client.get("/api/jobs?limit=5").json()]
    assert listed.index(second) < listed.index(first)


# ------------------------------------------------------------------- clips


@pytest.fixture
def rendered_clip(client, tmp_path):
    """A job with one clip whose output file exists on disk."""
    from app.db import session_scope
    from app.models import Clip, ClipStatus

    job_id = create_job(client).json()["job_id"]
    output = tmp_path / "clip_01.mp4"
    output.write_bytes(b"\x00\x01" * 4096)

    with session_scope() as session:
        clip = Clip(
            job_id=job_id,
            rank=1,
            start_time=10.0,
            end_time=40.0,
            duration=30.0,
            ai_score=88.0,
            final_score=71.5,
            title="A title",
            hashtags=["#one"],
            status=ClipStatus.READY,
            output_path=str(output),
            score_breakdown={"signals": {"ai_score": 0.88}},
        )
        session.add(clip)
        session.flush()
        return {"job_id": job_id, "clip_id": clip.id, "path": output}


def test_clip_listing_exposes_download_and_preview_links(client, rendered_clip):
    [clip] = client.get(f"/api/jobs/{rendered_clip['job_id']}/clips").json()
    assert clip["has_output"] is True
    assert clip["download_url"] == f"/api/clips/{clip['id']}/download"
    assert clip["preview_url"] == f"/api/clips/{clip['id']}/preview"


def test_metadata_edits_are_saved_and_hashtags_normalised(client, rendered_clip):
    response = client.patch(
        f"/api/clips/{rendered_clip['clip_id']}",
        json={"title": "Edited", "hashtags": ["podcast", "#mindset"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Edited"
    assert body["hashtags"] == ["#podcast", "#mindset"]
    # Persisted, not just echoed.
    assert client.get(f"/api/clips/{rendered_clip['clip_id']}").json()["title"] == "Edited"


def test_download_sends_the_file_as_an_attachment(client, rendered_clip):
    response = client.get(f"/api/clips/{rendered_clip['clip_id']}/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert len(response.content) == rendered_clip["path"].stat().st_size


def test_preview_supports_range_requests_so_the_player_can_seek(client, rendered_clip):
    size = rendered_clip["path"].stat().st_size
    response = client.get(
        f"/api/clips/{rendered_clip['clip_id']}/preview", headers={"Range": "bytes=100-199"}
    )
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 100-199/{size}"
    assert len(response.content) == 100

    unsatisfiable = client.get(
        f"/api/clips/{rendered_clip['clip_id']}/preview",
        headers={"Range": f"bytes={size + 10}-"},
    )
    assert unsatisfiable.status_code == 416


def test_download_of_an_unrendered_clip_is_a_conflict(client, rendered_clip):
    from app.db import session_scope
    from app.models import Clip

    with session_scope() as session:
        session.get(Clip, rendered_clip["clip_id"]).output_path = None
    assert client.get(f"/api/clips/{rendered_clip['clip_id']}/download").status_code == 409


def test_publishing_is_declared_unimplemented(client, rendered_clip):
    response = client.post(
        f"/api/clips/{rendered_clip['clip_id']}/publish", json={"platform": "tiktok"}
    )
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


# ------------------------------------------------------------------ system


def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert body["ffmpeg"] is True
    assert body["encoder"]
    assert body["ai_provider"] == "heuristic"  # no key configured in tests


def test_config_exposes_no_secrets(client):
    body = client.get("/api/config").json()
    assert "dynamic" in body["caption_styles"]
    assert sum(body["ranking_weights"].values()) == pytest.approx(1.0)
    assert "key" not in str(body).lower()


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "CLIPZ" in response.text


def test_health_flags_a_missing_js_runtime(client, monkeypatch):
    """Downloads get flaky without one, so the UI has to be able to say so."""
    from app.services import downloader

    downloader.detect_js_runtimes.cache_clear()
    monkeypatch.setattr(downloader.shutil, "which", lambda _binary: None)
    assert client.get("/api/health").json()["js_runtime"] is None

    downloader.detect_js_runtimes.cache_clear()
    monkeypatch.setattr(downloader.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr(downloader, "_binary_major_version", lambda _binary: 22)
    assert client.get("/api/health").json()["js_runtime"] == "deno"
    downloader.detect_js_runtimes.cache_clear()


def test_old_node_is_not_offered_to_ytdlp(monkeypatch):
    """yt-dlp rejects Node below 20; offering it would just add a warning."""
    from app.services import downloader

    downloader.detect_js_runtimes.cache_clear()
    monkeypatch.setattr(
        downloader.shutil, "which", lambda binary: "/usr/bin/node" if binary == "node" else None
    )
    monkeypatch.setattr(downloader, "_binary_major_version", lambda _binary: 18)
    assert downloader.detect_js_runtimes() == ()

    monkeypatch.setattr(downloader, "_binary_major_version", lambda _binary: 20)
    downloader.detect_js_runtimes.cache_clear()
    assert downloader.detect_js_runtimes() == ("node",)
    downloader.detect_js_runtimes.cache_clear()


def test_quit_is_absent_outside_the_packaged_build(client, monkeypatch):
    """A source checkout runs in a terminal; remote shutdown would be a footgun."""
    monkeypatch.delenv("CLIPZ_PACKAGED", raising=False)
    response = client.post("/api/quit")
    assert response.status_code == 404
    assert client.get("/api/health").json()["packaged"] is False


def test_quit_signals_the_process_in_the_packaged_build(client, monkeypatch):
    monkeypatch.setenv("CLIPZ_PACKAGED", "1")
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr("app.api.system.os.kill", lambda pid, sig: signalled.append((pid, sig)))

    response = client.post("/api/quit")
    assert response.status_code == 202
    assert client.get("/api/health").json()["packaged"] is True

    import os
    import signal

    assert signalled == [(os.getpid(), signal.SIGINT)]


# --------------------------------------------------------------------- batch


def test_batch_creates_one_job_per_url(client, no_real_worker):
    urls = [
        "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "https://www.youtube.com/watch?v=bbbbbbbbbbb",
        "https://www.youtube.com/watch?v=ccccccccccc",
    ]
    response = client.post("/api/jobs/batch", json={"urls": urls, "clip_count": 3})
    assert response.status_code == 201
    results = response.json()
    assert len(results) == 3
    assert all(r["job"] is not None and r["error"] is None for r in results)
    assert {r["job"]["source_url"] for r in results} == set(urls)
    assert sorted(no_real_worker) == sorted(r["job"]["job_id"] for r in results)


def test_batch_reports_a_bad_line_without_sinking_the_others(client, no_real_worker):
    response = client.post(
        "/api/jobs/batch",
        json={
            "urls": [
                "https://www.youtube.com/watch?v=good1",
                "not a url at all",
                "https://vimeo.com/12345",
                "https://www.youtube.com/watch?v=good2",
            ]
        },
    )
    assert response.status_code == 201
    results = response.json()
    by_url = {r["url"]: r for r in results}
    assert by_url["https://www.youtube.com/watch?v=good1"]["job"] is not None
    assert by_url["https://www.youtube.com/watch?v=good2"]["job"] is not None
    assert "http(s)" in by_url["not a url at all"]["error"]
    assert "YouTube" in by_url["https://vimeo.com/12345"]["error"]
    assert len(no_real_worker) == 2, "only the two valid URLs should have been queued"


def test_batch_drops_blank_lines_and_flags_duplicates(client, no_real_worker):
    response = client.post(
        "/api/jobs/batch",
        json={
            "urls": [
                "https://www.youtube.com/watch?v=dup",
                "   ",
                "",
                "https://www.youtube.com/watch?v=dup",
            ]
        },
    )
    results = response.json()
    assert len(results) == 2  # blanks silently dropped, duplicate flagged
    assert results[0]["job"] is not None
    assert results[1]["error"] == "Duplicate in this batch; skipped"
    assert len(no_real_worker) == 1


def test_batch_of_entirely_invalid_urls_is_rejected(client):
    response = client.post("/api/jobs/batch", json={"urls": ["not a url", "also not a url"]})
    assert response.status_code == 422
    assert "No valid YouTube URLs" in response.json()["detail"]


def test_batch_respects_the_size_limit(client):
    response = client.post("/api/jobs/batch", json={"urls": ["https://youtube.com/x"] * 51})
    assert response.status_code == 422


def test_batch_requires_at_least_one_url(client):
    assert client.post("/api/jobs/batch", json={"urls": []}).status_code == 422


def test_batch_max_duration_is_capped_like_the_single_endpoint(client, no_real_worker):
    response = client.post(
        "/api/jobs/batch",
        json={"urls": ["https://www.youtube.com/watch?v=x"], "max_duration": 6000},
    )
    assert response.json()[0]["job"]["max_duration"] == 90.0
