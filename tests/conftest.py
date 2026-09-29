"""Shared fixtures. Every test runs against a throwaway storage tree.

The environment is configured at import time, not in a fixture: pytest imports
test modules (and through them `app.config`) before any fixture runs, and
`Settings` is read once at import. Setting these later would silently leave the
tests writing into the real `storage/` tree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_STORAGE = Path(tempfile.mkdtemp(prefix="clipz-tests-"))
os.environ["STORAGE_DIR"] = str(_STORAGE)
# Settings also read `$XDG_CONFIG_HOME/clipz/config.env`, which on a real
# machine holds the developer's own credentials and overrides. Pointing it at
# an empty directory keeps the suite testing the defaults it asserts against,
# rather than whatever happens to be installed on the machine running it.
os.environ["XDG_CONFIG_HOME"] = str(_STORAGE / "config")
os.environ["DATABASE_URL"] = f"sqlite:///{_STORAGE / 'test.db'}"
os.environ["KEEP_INTERMEDIATE_FILES"] = "true"
os.environ["AI_PROVIDER"] = "heuristic"
os.environ.pop("OX_API_KEY", None)
sys.path.insert(0, str(REPO_ROOT / "backend"))


@pytest.fixture(scope="session", autouse=True)
def isolated_storage():
    """Guard the isolation, create the schema, then clean up afterwards."""
    from app.config import settings
    from app.db import init_db

    assert settings.storage_dir == _STORAGE, "tests must not write to the real storage tree"
    # Created here rather than relying on a TestClient having started earlier,
    # so any test file can run on its own.
    init_db()
    yield _STORAGE
    shutil.rmtree(_STORAGE, ignore_errors=True)


@pytest.fixture
def transcript():
    """A small, realistic word-timed transcript."""
    from app.services.transcript import Segment, Transcript, Word

    script = [
        "Nobody talks about this part of the business.",
        "The truth is most people quit in the first year.",
        "I lost forty thousand dollars before anything worked.",
        "What changed was measuring one number every single day.",
        "That habit turned the whole thing around in six months.",
    ]
    segments = []
    clock = 5.0
    for line in script:
        words = []
        for token in line.split():
            words.append(Word(start=clock, end=clock + 0.34, text=token, probability=0.93))
            clock += 0.42
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=line, words=words, confidence=0.9)
        )
        clock += 0.8  # pause between sentences

    return Transcript(
        text=" ".join(script),
        language="en",
        segments=segments,
        duration=clock,
        language_probability=0.99,
    )


@pytest.fixture
def long_transcript():
    """A 90-second transcript with enough distinct moments for several clips."""
    from app.services.transcript import Segment, Transcript, Word

    script = [
        "Nobody talks about this part of running a business.",
        "The truth is most people quit somewhere in the first year.",
        "I lost forty thousand dollars before a single thing worked.",
        "My co-founder walked out on a Tuesday and never replied again.",
        "What changed was measuring exactly one number every single day.",
        "That habit turned the entire company around inside six months.",
        "Here is the part that still makes me genuinely angry about it.",
        "Every consultant we hired sold us software we never once opened.",
        "Ninety percent of that budget bought absolutely nothing at all.",
        "So we cancelled everything and rebuilt the process on paper.",
        "Paper beat the expensive dashboard for almost two whole years.",
        "If you are starting today, please just begin with the paper.",
    ]
    segments = []
    clock = 4.0
    for index, line in enumerate(script):
        words = []
        for token in line.split():
            words.append(Word(start=clock, end=clock + 0.36, text=token, probability=0.91))
            clock += 0.45
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=line, words=words, confidence=0.88)
        )
        clock += 0.7 + (index % 3) * 0.4  # varied pauses between sentences

    return Transcript(
        text=" ".join(script),
        language="en",
        segments=segments,
        duration=clock,
        language_probability=0.99,
    )


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """A two-minute 16:9 clip with audio, built with FFmpeg."""
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24:duration=120",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=120",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def client(isolated_storage):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def stub_openai_server():
    """A local server speaking the OpenAI chat-completions protocol.

    Exercises the real `openai` client, our prompt and our parser end to end
    without a network call or a key. `reply["content"]` can be reassigned by a
    test to change what the "model" returns.
    """
    import json as jsonlib
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[dict] = []
    reply = {
        "content": jsonlib.dumps(
            {
                "clips": [
                    {
                        "start": 6,
                        "end": 30,
                        "viral_score": 91,
                        "hook_score": 88,
                        "emotion_score": 74,
                        "reason": "Opens on a contrarian claim.",
                        "title": "Nobody talks about this",
                        "hashtags": ["podcast", "#mindset"],
                    }
                ]
            }
        )
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            received.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("authorization"),
                    "body": jsonlib.loads(body),
                }
            )
            payload = jsonlib.dumps(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "model": "stub-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply["content"]},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # keep the test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "received": received,
        "reply": reply,
    }
    server.shutdown()
