"""Byte-range file responses.

Starlette's FileResponse always sends the whole file, which means the review
player cannot seek. Clips are small but seeking is half of reviewing one, so
range requests are handled here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1024 * 256


def range_response(
    path: Path,
    *,
    range_header: str | None,
    media_type: str = "video/mp4",
    filename: str | None = None,
    download: bool = False,
) -> StreamingResponse | FileResponse:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    size = path.stat().st_size
    disposition = "attachment" if download else "inline"
    headers = {
        "accept-ranges": "bytes",
        "content-disposition": f'{disposition}; filename="{filename or path.name}"',
    }

    match = RANGE_RE.fullmatch((range_header or "").strip()) if range_header else None
    if not match:
        return FileResponse(path, media_type=media_type, headers=headers)

    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    elif raw_end:  # suffix range: the last N bytes
        start = max(0, size - int(raw_end))
        end = size - 1
    else:
        raise HTTPException(status_code=416, detail="Invalid range")

    if start >= size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"content-range": f"bytes */{size}"},
        )
    end = min(end, size - 1)

    def stream() -> Iterator[bytes]:
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers.update(
        {
            "content-range": f"bytes {start}-{end}/{size}",
            "content-length": str(end - start + 1),
        }
    )
    return StreamingResponse(stream(), status_code=206, media_type=media_type, headers=headers)
