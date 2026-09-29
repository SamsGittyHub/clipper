"""CLIPZ -- application entry point.

Run with:  uvicorn app.main:app --app-dir backend --reload
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import accounts, auth, clips, jobs, settings as settings_api, system, videos
from .config import REPO_ROOT, settings
from .db import init_db
from .errors import ErrorCode, PipelineError
from .services.media import ffmpeg_available
from .workers.queue import get_queue, recover_interrupted_jobs

log = logging.getLogger("clipz")

FRONTEND_DIR = REPO_ROOT / "frontend"


def configure_logging() -> None:
    """Console logging always; a rotating file when there is somewhere to put it.

    The packaged desktop build is launched from a menu entry and has no
    terminal, so without a file there is no way to find out why a job failed.
    """
    root = logging.getLogger()
    if root.handlers:  # uvicorn --reload re-imports this module
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if not settings.log_to_file:
        return
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            settings.logs_dir / "clipz.log",
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backups,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except OSError as exc:  # pragma: no cover - read-only storage
        log.warning("Could not open the log file: %s", exc)


def prune_old_logs() -> int:
    """Delete FFmpeg logs past their retention window.

    One file per FFmpeg invocation adds up quickly -- a few hundred per busy
    day -- and they are only useful while the render they describe is recent.
    """
    days = settings.log_retention_days
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        entries = list(settings.logs_dir.glob("*.log"))
    except OSError:  # pragma: no cover
        return 0
    for path in entries:
        if path.name.startswith("clipz.log"):
            continue  # the application log rotates on its own
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:  # pragma: no cover - raced with another cleanup
            continue
    if removed:
        log.info("Removed %d log file(s) older than %d days", removed, days)
    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_dirs()
    init_db()
    if not ffmpeg_available():
        log.error("FFmpeg/ffprobe not found on PATH -- rendering will fail")
    prune_old_logs()
    # Nothing survives a restart mid-job, so reconcile before accepting work:
    # re-queue what never started, fail what was running.
    if settings.recover_jobs_on_startup:
        recover_interrupted_jobs()
    get_queue().start()
    log.info(
        "CLIPZ %s ready; storage at %s%s",
        __version__,
        settings.storage_dir,
        " (API token required)" if settings.api_token else "",
    )
    yield
    get_queue().stop()


app = FastAPI(
    title="CLIPZ",
    description="Turns long-form video into ranked, captioned, publish-ready vertical clips.",
    version=__version__,
    lifespan=lifespan,
)

# The API is local-only in v1, but the UI may be served from a dev port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Gate `/api/*` when `API_TOKEN` is set. A no-op when it is not."""
    if request.url.path.startswith("/api/") and not auth.authorised(request):
        return JSONResponse(
            status_code=401,
            content={
                "error_code": "ERROR_UNAUTHORISED",
                "detail": "This server requires an API token.",
            },
            headers={"www-authenticate": "Bearer"},
        )
    return await call_next(request)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_request: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error_code": exc.code, "detail": exc.message, "context": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the documented error shape instead of a bare 500 body.

    The frontend reads `detail` on every failure; without this an unexpected
    server error surfaces there as an unhelpful "Request failed (500)".
    """
    log.exception("Unhandled error serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": ErrorCode.INTERNAL,
            "detail": f"{type(exc).__name__}: {exc}",
        },
    )


app.include_router(auth.router)
app.include_router(system.router)
app.include_router(settings_api.router)
app.include_router(accounts.router)
app.include_router(jobs.router)
app.include_router(clips.router)
app.include_router(videos.router)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
