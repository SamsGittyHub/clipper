#!/usr/bin/env bash
#
# Build CLIPZ-x86_64.AppImage.
#
# The AppDir holds a relocatable CPython (python-build-standalone) with every
# Python dependency installed into it, plus the app itself. FFmpeg is NOT
# bundled: the launcher checks for it and tells the user to install it.
#
#   ./packaging/build-appimage.sh            # build into dist/
#   CACHE_DIR=/somewhere ./packaging/...     # reuse downloaded tools
#
set -euo pipefail

# Isolate the build from the builder's own Python. Without this, packages in
# ~/.local/lib/pythonX.Y/site-packages are seen as already satisfied and get
# skipped, producing a bundle that is missing dependencies and fails on every
# other machine.
export PYTHONNOUSERSITE=1
export PIP_USER=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
CACHE_DIR="${CACHE_DIR:-$BUILD_DIR/cache}"
APPDIR="$BUILD_DIR/CLIPZ.AppDir"

PYTHON_RELEASE="${PYTHON_RELEASE:-20260901}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.14}"
PYTHON_TARBALL="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${PYTHON_TARBALL//+/%2B}"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
# Ubuntu 24.04 and other current distros ship libfuse3 but not libfuse2, which
# the classic AppImage runtime needs. This runtime works with fuse3.
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

fetch() {  # fetch <url> <destination>
    local url="$1" dest="$2"
    [ -s "$dest" ] && { echo "    cached: $(basename "$dest")"; return; }
    echo "    downloading $(basename "$dest")"
    curl -fsSL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

say "Preparing build directories"
mkdir -p "$CACHE_DIR" "$DIST_DIR"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib/clipz" "$APPDIR/usr/bin"

say "Fetching build tools"
fetch "$PYTHON_URL" "$CACHE_DIR/python.tar.gz"
fetch "$APPIMAGETOOL_URL" "$CACHE_DIR/appimagetool"
fetch "$RUNTIME_URL" "$CACHE_DIR/runtime-x86_64"
chmod +x "$CACHE_DIR/appimagetool"

say "Unpacking Python $PYTHON_VERSION"
tar -xzf "$CACHE_DIR/python.tar.gz" -C "$APPDIR/usr"
mv "$APPDIR/usr/python" "$APPDIR/usr/python-runtime"
PY="$APPDIR/usr/python-runtime/bin/python3"
"$PY" --version

say "Installing Python dependencies"
"$PY" -m pip install --no-cache-dir --upgrade pip >/dev/null
# Test-only dependencies are not shipped.
grep -v -E '^\s*(pytest)' "$REPO_ROOT/requirements.txt" > "$BUILD_DIR/requirements-runtime.txt"
"$PY" -m pip install --no-cache-dir --no-warn-script-location -r "$BUILD_DIR/requirements-runtime.txt"

say "Copying the application"
cp -r "$REPO_ROOT/backend" "$APPDIR/usr/lib/clipz/backend"
cp -r "$REPO_ROOT/frontend" "$APPDIR/usr/lib/clipz/frontend"
cp "$REPO_ROOT/packaging/launch.py" "$APPDIR/usr/lib/clipz/launch.py"
cp "$REPO_ROOT/README.md" "$APPDIR/usr/lib/clipz/README.md"
find "$APPDIR/usr/lib/clipz" -name '__pycache__' -type d -prune -exec rm -rf {} +

say "Trimming the bundle"
# Test suites and headers inside site-packages add tens of MB and are not used
# at runtime; the pip cache and static libs likewise.
SITE="$APPDIR/usr/python-runtime/lib/python3.12/site-packages"
find "$SITE" -type d -name 'tests' -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$APPDIR/usr/python-runtime" -name '*.a' -delete 2>/dev/null || true
rm -rf "$APPDIR/usr/python-runtime/lib/python3.12/test" \
       "$APPDIR/usr/python-runtime/lib/python3.12/idlelib" \
       "$APPDIR/usr/python-runtime/lib/python3.12/tkinter" \
       "$APPDIR/usr/python-runtime/share" 2>/dev/null || true

say "Verifying the bundle imports"
# Catches a bundle that is missing transitive dependencies before it ships.
PYTHONPATH="$APPDIR/usr/lib/clipz/backend" CLIPZ_PACKAGED=1 \
    STORAGE_DIR="$BUILD_DIR/verify-storage" "$PY" - <<'VERIFY'
import importlib.util
import sys

required = [
    "fastapi", "uvicorn", "sqlalchemy", "pydantic", "pydantic_settings",
    "typing_extensions", "anyio", "h11", "yt_dlp", "openai", "httpx",
    "faster_whisper", "ctranslate2", "numpy", "av", "onnxruntime",
    "tokenizers", "huggingface_hub", "requests", "tqdm", "cv2",
]
missing = [name for name in required if not importlib.util.find_spec(name)]
if missing:
    sys.exit("Bundle is missing dependencies: " + ", ".join(missing))

from app.main import app  # the app itself must import cleanly

routes = {getattr(route, "path", "") for route in app.routes}
for path in (
    "/api/health",
    "/api/jobs",
    "/api/jobs/{job_id}/cancel",
    "/api/jobs/{job_id}/retry",
    "/api/settings/ai",
    "/api/auth",
    "/",
):
    if path not in routes:
        sys.exit(f"Bundled app is missing the {path} route")

# A packaged build that cannot migrate an existing database would break every
# upgrade in place, so prove the helper ships and runs.
from app.db import add_missing_columns  # noqa: F401

from app.services.face_tracking import face_tracking_available

if not face_tracking_available():
    sys.exit("Smart crop model did not ship correctly inside the bundle")

print(f"    ok: {len(required)} dependencies, {len(routes)} routes, smart crop ready")
VERIFY
rm -rf "$BUILD_DIR/verify-storage"

say "Writing AppDir metadata"
cp "$REPO_ROOT/packaging/clipz.desktop" "$APPDIR/clipz.desktop"
cp "$REPO_ROOT/packaging/clipz.png" "$APPDIR/clipz.png"
mkdir -p "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp "$REPO_ROOT/packaging/clipz.desktop" "$APPDIR/usr/share/applications/"
cp "$REPO_ROOT/packaging/clipz.png" "$APPDIR/usr/share/icons/hicolor/512x512/apps/"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
# AppImage entry point. $APPDIR is set by the runtime when mounted, but the
# AppDir can also be run directly, so fall back to this script's location.
HERE="$(dirname "$(readlink -f "$0")")"
export CLIPZ_PACKAGED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
exec "$HERE/usr/python-runtime/bin/python3" "$HERE/usr/lib/clipz/launch.py" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

say "Building the AppImage"
OUTPUT="$DIST_DIR/CLIPZ-x86_64.AppImage"
rm -f "$OUTPUT"
# appimagetool itself is an AppImage; --appimage-extract-and-run avoids needing
# libfuse2 to run it on hosts that only have fuse3.
ARCH=x86_64 "$CACHE_DIR/appimagetool" --appimage-extract-and-run \
    --runtime-file "$CACHE_DIR/runtime-x86_64" \
    "$APPDIR" "$OUTPUT" 2>&1 | grep -vE '^\s*$' | tail -5
chmod +x "$OUTPUT"

say "Done"
printf '  %s  (%s)\n\n' "$OUTPUT" "$(du -h "$OUTPUT" | cut -f1)"
