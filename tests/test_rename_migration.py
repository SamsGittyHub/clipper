"""Upgrading from the app's previous name.

CLIPZ was called ClipForge AI. Anyone upgrading has a populated data
directory and database under the old name, and a database full of absolute
paths pointing into it. Starting empty would strand finished clips and the
connected TikTok accounts whose refresh tokens live in that database.
"""

from __future__ import annotations

import sqlite3

import pytest

from app import config as config_module
from app.config import adopt_legacy_database, default_storage_dir


@pytest.fixture(autouse=True)
def reset_adoption_state():
    config_module.adopted_from = None
    yield
    config_module.adopted_from = None


def legacy_install(root, *, name="ClipForgeAI"):
    """A data directory as the previous release would have left it."""
    directory = root / name
    (directory / "clips" / "job_old").mkdir(parents=True)
    (directory / "clips" / "job_old" / "clip_01.mp4").write_bytes(b"\x00" * 32)
    database = directory / "clipforge.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE marker (note TEXT)")
    connection.execute("INSERT INTO marker VALUES ('real user data')")
    connection.commit()
    connection.close()
    return directory


# ----------------------------------------------------------- the directory


def test_the_old_data_directory_is_adopted_on_first_run(tmp_path, monkeypatch):
    legacy = legacy_install(tmp_path)
    monkeypatch.setenv("CLIPZ_PACKAGED", "1")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    storage = default_storage_dir()

    assert storage == tmp_path / "CLIPZ"
    assert storage.is_dir() and not legacy.exists()
    assert (storage / "clips" / "job_old" / "clip_01.mp4").exists(), "clips came across"
    assert config_module.adopted_from == legacy, "the old root is recorded for path repair"


def test_an_existing_new_directory_is_never_overwritten(tmp_path, monkeypatch):
    """Two installs must not clobber each other; the new one wins."""
    legacy = legacy_install(tmp_path)
    current = tmp_path / "CLIPZ"
    current.mkdir()
    (current / "keep.txt").write_text("newer data")

    monkeypatch.setenv("CLIPZ_PACKAGED", "1")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    storage = default_storage_dir()

    assert (storage / "keep.txt").read_text() == "newer data"
    assert legacy.exists(), "the old directory is left alone rather than merged"
    assert config_module.adopted_from is None


def test_a_source_checkout_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPZ_PACKAGED", raising=False)
    assert default_storage_dir().name == "storage"
    assert config_module.adopted_from is None


# ------------------------------------------------------------ the database


def test_the_old_database_file_is_adopted(tmp_path):
    directory = legacy_install(tmp_path)

    assert adopt_legacy_database(directory) is True

    assert (directory / "clipz.db").exists()
    assert not (directory / "clipforge.db").exists()
    connection = sqlite3.connect(directory / "clipz.db")
    assert connection.execute("SELECT note FROM marker").fetchone() == ("real user data",)
    connection.close()


def test_data_still_in_the_write_ahead_log_survives(tmp_path):
    """Renaming the main file alone would orphan `-wal` and lose it."""
    directory = tmp_path / "store"
    directory.mkdir()
    legacy = directory / "clipforge.db"
    connection = sqlite3.connect(legacy)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE marker (note TEXT)")
    connection.execute("INSERT INTO marker VALUES ('written into the wal')")
    connection.commit()
    connection.close()

    assert adopt_legacy_database(directory) is True

    connection = sqlite3.connect(directory / "clipz.db")
    assert connection.execute("SELECT note FROM marker").fetchone() == ("written into the wal",)
    connection.close()
    assert not (directory / "clipforge.db-wal").exists()


def test_adoption_does_not_touch_an_existing_database(tmp_path):
    directory = legacy_install(tmp_path)
    (directory / "clipz.db").write_bytes(b"newer")

    assert adopt_legacy_database(directory) is False
    assert (directory / "clipz.db").read_bytes() == b"newer"


def test_adoption_is_a_no_op_when_there_is_nothing_to_adopt(tmp_path):
    assert adopt_legacy_database(tmp_path) is False


# --------------------------------------------------------------- the paths


def test_stored_paths_are_repointed_at_the_new_directory(tmp_path):
    """Clips record absolute paths, so a moved directory breaks every one.

    Without this the job survives the upgrade but every preview, download and
    thumbnail points somewhere that no longer exists.
    """
    from app.db import relocate_stored_paths, session_scope
    from app.models import Clip, ClipStatus, Job, JobStatus, Video

    old_root = tmp_path / "ClipForgeAI"
    new_root = tmp_path / "CLIPZ"

    with session_scope() as session:
        video = Video(
            source_url="https://example.invalid/v",
            local_path=f"{old_root}/downloads/job_x/source.mp4",
            transcript_path=f"{old_root}/transcripts/job_x.json",
        )
        session.add(video)
        job = Job(source_url="https://example.invalid/v", status=JobStatus.COMPLETED)
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            rank=1,
            start_time=0.0,
            end_time=20.0,
            duration=20.0,
            status=ClipStatus.READY,
            output_path=f"{old_root}/clips/job_x/clip_01.mp4",
            subtitle_path=f"{old_root}/work/job_x/clip_01.ass",
        )
        session.add(clip)
        session.flush()
        clip_id, video_id = clip.id, video.id

    changed = relocate_stored_paths(old_root, new_root)

    assert changed == 4
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        video = session.get(Video, video_id)
        assert clip.output_path == f"{new_root}/clips/job_x/clip_01.mp4"
        assert clip.subtitle_path == f"{new_root}/work/job_x/clip_01.ass"
        assert video.local_path == f"{new_root}/downloads/job_x/source.mp4"
        assert video.transcript_path == f"{new_root}/transcripts/job_x.json"


def test_paths_outside_the_old_directory_are_left_alone(tmp_path):
    """A custom STORAGE_DIR must not have its paths rewritten by accident."""
    from app.db import relocate_stored_paths, session_scope
    from app.models import Clip, ClipStatus, Job, JobStatus

    unrelated = "/mnt/nas/exports/clip_01.mp4"
    with session_scope() as session:
        job = Job(source_url="https://example.invalid/v", status=JobStatus.COMPLETED)
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id, rank=1, start_time=0.0, end_time=10.0, duration=10.0,
            status=ClipStatus.READY, output_path=unrelated,
        )
        session.add(clip)
        session.flush()
        clip_id = clip.id

    relocate_stored_paths(tmp_path / "ClipForgeAI", tmp_path / "CLIPZ")

    with session_scope() as session:
        assert session.get(Clip, clip_id).output_path == unrelated
