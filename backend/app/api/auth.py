"""Optional API token (PRD 12: the app is local, but need not stay that way).

Unset by default, which is right for one user on `127.0.0.1`. Setting
`API_TOKEN` closes the API to anything that cannot present it -- worth doing
before binding to `0.0.0.0`, because the settings panel stores a model API
key and `/api/quit` stops the server.

Three ways to present it, so scripts and the browser can both work:
`Authorization: Bearer <token>`, `X-API-Key: <token>`, or the cookie this
module sets once the user has entered it in the UI.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])

COOKIE_NAME = "clipz_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30

# Paths that must stay reachable without a token: the TikTok OAuth redirect
# (TikTok cannot send our header, and it carries its own one-time state), and
# the endpoints used to find out about and satisfy the requirement itself.
EXEMPT_PATHS = frozenset(
    {
        "/api/auth",
        "/api/auth/",
        "/api/accounts/tiktok/callback",
    }
)


def required() -> bool:
    return bool(settings.api_token)


def _matches(candidate: str | None) -> bool:
    if not candidate or not settings.api_token:
        return False
    return hmac.compare_digest(candidate.strip(), settings.api_token)


def presented_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:]
    return request.headers.get("x-api-key") or request.cookies.get(COOKIE_NAME)


def authorised(request: Request) -> bool:
    if not required():
        return True
    if request.url.path in EXEMPT_PATHS:
        return True
    return _matches(presented_token(request))


class TokenIn(BaseModel):
    token: str = Field(..., max_length=500)


@router.get("")
def auth_state(request: Request) -> dict:
    """Whether a token is needed, and whether this caller already has one."""
    return {
        "required": required(),
        "authenticated": not required() or _matches(presented_token(request)),
    }


@router.post("")
def sign_in(payload: TokenIn, response: Response) -> dict:
    """Exchange the token for a cookie, so the UI asks only once."""
    if not required():
        return {"required": False, "authenticated": True}
    if not _matches(payload.token):
        raise HTTPException(status_code=401, detail="That token is not correct")
    response.set_cookie(
        COOKIE_NAME,
        payload.token.strip(),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return {"required": True, "authenticated": True}


@router.delete("")
def sign_out(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME)
    return {"required": required(), "authenticated": not required()}
