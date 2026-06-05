"""
Storage-service HTTP router.

Endpoints:
  POST   /storage/upload/{key}     — upload bytes at key
  GET    /storage/download/{key}   — download bytes at key
  DELETE /storage/{key}            — delete object at key
  GET    /storage/exists/{key}     — check if key exists
  GET    /storage/backends         — list available backends and active status
"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from raglab_common.exceptions import NotImplementedFeatureError, StorageError
from raglab_common.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/storage", tags=["storage"])


class UploadRequest(BaseModel):
    data_b64: str  # base64-encoded bytes
    content_type: str = "application/octet-stream"


class UploadResponse(BaseModel):
    key: str
    uri: str
    size: int
    backend: str


class ExistsResponse(BaseModel):
    key: str
    exists: bool
    backend: str


class BackendInfo(BaseModel):
    backend: str
    active: bool
    available_in: str | None = None


def _get_backend(request: Request) -> Any:
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(status_code=503, detail="Storage backend not initialised.")
    return backend


@router.post("/upload/{key:path}", response_model=UploadResponse)
async def upload(key: str, body: UploadRequest, request: Request) -> UploadResponse:
    """Upload bytes (base64-encoded) at the given key."""
    backend = _get_backend(request)
    try:
        data = base64.b64decode(body.data_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64 data: {exc}")

    try:
        uri = backend.upload(key, data)
        return UploadResponse(
            key=key, uri=uri, size=len(data), backend=backend.backend_type
        )
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/download/{key:path}")
async def download(key: str, request: Request) -> Response:
    """Download raw bytes at the given key. Returns binary response."""
    backend = _get_backend(request)
    try:
        data = backend.download(key)
        return Response(content=data, media_type="application/octet-stream")
    except StorageError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/{key:path}", status_code=204)
async def delete(key: str, request: Request) -> None:
    """Delete object at key. Idempotent — 204 even if key didn't exist."""
    backend = _get_backend(request)
    try:
        backend.delete(key)
    except StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/exists/{key:path}", response_model=ExistsResponse)
async def exists(key: str, request: Request) -> ExistsResponse:
    """Check whether an object exists at key."""
    backend = _get_backend(request)
    try:
        found = backend.exists(key)
        return ExistsResponse(key=key, exists=found, backend=backend.backend_type)
    except StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/backends", response_model=list[BackendInfo])
async def list_backends(request: Request) -> list[BackendInfo]:
    """List all registered storage backends with active/stub status."""
    from storage.factory import StorageFactory
    return [BackendInfo(**b) for b in StorageFactory.available()]
