#!/usr/bin/env python3
"""Desktop launcher for the packaged CLIPZ build.

Starts the local server on a free port, waits until it answers, then opens the
UI. Launched from a desktop menu there is no terminal to read, so problems are
reported in a dialog where one is available, and the UI carries its own Quit
button (there is no console to press Ctrl+C in).
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_NAME = "CLIPZ"


def fail(message: str, *, detail: str = "") -> None:
    """Report a fatal problem, using a dialog when one is available."""
    text = f"{message}\n\n{detail}".strip()
    print(f"{APP_NAME}: {text}", file=sys.stderr)
    for tool, args in (
        ("zenity", ["--error", "--width=460", f"--title={APP_NAME}", f"--text={text}"]),
        ("kdialog", ["--error", text, "--title", APP_NAME]),
        ("notify-send", [APP_NAME, text]),
    ):
        if shutil.which(tool):
            subprocess.run([tool, *args], check=False)
            break
    raise SystemExit(1)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    fail(
        "FFmpeg is required but was not found.",
        detail="Install it with:\n\n    sudo apt install ffmpeg\n\nthen start CLIPZ again.",
    )


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_port(preferred: int = 8000) -> int:
    """Use the preferred port when it is free, otherwise any open one."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    fail("Could not find a free network port to listen on.")
    raise AssertionError("unreachable")


def required_oauth_port() -> int | None:
    """The port publishing needs, when an OAuth provider is configured.

    TikTok matches the redirect URI against the one registered for the app,
    character for character. Drifting onto a spare port therefore does not
    merely inconvenience the user: the browser is sent back to whatever is
    listening on the registered port, which is some other program, carrying
    a live authorization code. So a configured redirect URI pins the port,
    and a clash is reported rather than worked around.
    """
    from app.config import settings

    if not (settings.tiktok_client_key and settings.tiktok_client_secret):
        return None
    parsed = urllib.parse.urlparse(settings.tiktok_redirect_uri)
    # Only a redirect URI pointing at this machine says anything about the
    # port to listen on. Behind a tunnel or a reverse proxy the public URL is
    # someone else's hostname on port 443, and the local port it forwards to
    # is the tunnel's business -- inferring 443 from it would have CLIPZ try
    # to bind a privileged port it has no reason to want.
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        return None
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def wait_until_ready(port: int, *, timeout: float = 90.0) -> bool:
    """Poll /api/health until the server answers."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


def open_ui(url: str) -> None:
    """Open a chromeless window when a Chromium-family browser exists.

    `--app=` gives something that looks like a desktop window rather than a
    browser tab. Firefox has no equivalent, so it gets a normal tab.
    """
    for browser in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "brave-browser", "microsoft-edge"):
        path = shutil.which(browser)
        if path:
            try:
                subprocess.Popen(
                    [path, f"--app={url}", "--window-size=1280,900"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return
            except OSError:
                break
    webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(prog="clipz", description=f"{APP_NAME} — long-form video into vertical clips")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLIPZ_PORT", 8000)))
    parser.add_argument("--no-browser", action="store_true", help="start the server without opening a window")
    parser.add_argument("--storage", type=Path, default=None, help="where clips and the database are kept")
    args = parser.parse_args()
    args.port_was_given = any(
        argument == "--port" or argument.startswith("--port=") for argument in sys.argv[1:]
    )

    os.environ.setdefault("CLIPZ_PACKAGED", "1")
    if args.storage:
        os.environ["STORAGE_DIR"] = str(args.storage)

    check_ffmpeg()
    sys.path.insert(0, str(HERE / "backend"))

    from app.config import settings

    # A configured TikTok app pins the port to its registered redirect URI.
    pinned = required_oauth_port()
    if pinned is not None and not args.port_was_given:
        if not port_is_free(pinned):
            fail(
                f"Port {pinned} is already in use.",
                detail=(
                    "TikTok publishing is configured, and its redirect URI\n"
                    f"    {settings.tiktok_redirect_uri}\n"
                    f"requires CLIPZ to listen on port {pinned}. Connecting an account\n"
                    "would otherwise send your login back to whatever else is on that\n"
                    "port.\n\nStop the other program, or start CLIPZ with --port to run\n"
                    "without publishing."
                ),
            )
        port = pinned
    else:
        port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"  # imported after the environment is set

    print(f"{APP_NAME}\n  UI:      {url}\n  Storage: {settings.storage_dir}", flush=True)

    if not args.no_browser:
        def opener() -> None:
            if wait_until_ready(port):
                open_ui(url)
            else:
                print("Server did not become ready in time; open the URL manually.", file=sys.stderr)

        threading.Thread(target=opener, daemon=True).start()

    import uvicorn

    from app.main import app

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        pass
    print(f"{APP_NAME} stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
