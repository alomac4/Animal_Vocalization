"""Authenticated raw-video upload API. Bind to localhost behind private VPN HTTPS."""
from __future__ import annotations

import asyncio
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.responses import FileResponse, Response

from .jobs import CapacityError, JobManager
from .settings import Settings


class LeasedVideoResponse(FileResponse):
    """Release a job file lease even if the client disconnects mid-download."""
    def __init__(self, *args, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release()


def create_app(settings: Settings | None = None, renderer=None) -> FastAPI:
    config = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = JobManager(config, renderer)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.manager.close)

    app = FastAPI(title="BARK private video inference", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    async def authenticate(request: Request) -> None:
        provided = request.headers.get("authorization", "")
        expected = f"Bearer {config.token}"
        if not secrets.compare_digest(provided.encode(), expected.encode()):
            raise HTTPException(401, "Incorrect API token", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        return {"status": "ok", "model": config.model_label, "device": config.device,
                "imgsz": config.imgsz,
                "pipeline": "dog_boxes_then_breed_crops",
                "breed_enabled": config.breed_weights is not None,
                "breed_conf": config.breed_conf,
                "breed_weights_available": bool(config.breed_weights and config.breed_weights.is_file()),
                "max_upload_bytes": config.max_upload_bytes,
                "max_duration_seconds": config.max_duration_seconds,
                "retention_seconds": config.retention_seconds,
                "weights_available": config.weights.is_file()}

    @app.post("/jobs", status_code=202, dependencies=[Depends(authenticate)])
    async def upload(request: Request):
        if request.headers.get("content-type", "").split(";")[0] not in (
                "application/octet-stream", "video/mp4", "video/quicktime", "video/webm"):
            raise HTTPException(415, "Send the video as a raw binary request body")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared_size = int(length)
            except ValueError:
                raise HTTPException(400, "Invalid Content-Length")
            if declared_size < 0 or declared_size > config.max_upload_bytes:
                raise HTTPException(413, "Video exceeds the 250 MiB upload limit")
        manager = request.app.state.manager
        try:
            job = manager.reserve()
        except CapacityError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "15"})
        total = 0
        try:
            # File names come from the server, never from an untrusted upload.
            async with asyncio.timeout(600):
                with (job.directory / "input.video").open("wb") as file:
                    async for chunk in request.stream():
                        if job.cancel.is_set():
                            raise HTTPException(408, "Upload expired")
                        total += len(chunk)
                        if total > config.max_upload_bytes:
                            raise HTTPException(413, "Video exceeds the upload limit")
                        await asyncio.to_thread(file.write, chunk)
            if not total:
                raise HTTPException(400, "Empty video")
            manager.submit(job)
            return job.payload()
        except BaseException as exc:
            with manager.lock:
                job.status = "failed"
                job.cancel.set()
                manager.remove_if_safe(job)
            if isinstance(exc, TimeoutError):
                raise HTTPException(408, "Upload timed out. Try a shorter video or a faster connection.") from exc
            raise

    def get_job(manager, job_id):
        job = manager.jobs.get(job_id)
        if not job or job.cancel.is_set() or (job.expires_at and job.expires_at <= time.time()):
            raise HTTPException(404, "Video expired or was removed. Analyze the video again.")
        return job

    @app.get("/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def status(request: Request, job_id: str):
        manager = request.app.state.manager
        with manager.lock:
            return get_job(manager, job_id).payload()

    @app.get("/jobs/{job_id}/video", dependencies=[Depends(authenticate)])
    def video(request: Request, job_id: str):
        manager = request.app.state.manager
        with manager.lock:
            job = get_job(manager, job_id)
            if job.status != "ready":
                raise HTTPException(409, "Video is not ready")
            job.readers += 1

        def release():
            with manager.lock:
                job.readers -= 1
                manager.remove_if_safe(job)

        return LeasedVideoResponse(job.directory / "result.mp4", media_type="video/mp4",
                            filename="bark_detected.mp4", content_disposition_type="inline",
                            release=release)

    @app.delete("/jobs/{job_id}", dependencies=[Depends(authenticate)], status_code=204)
    def delete(request: Request, job_id: str):
        # Idempotent deletion supports retries after interrupted connections.
        request.app.state.manager.delete(job_id)
        return Response(status_code=204)

    return app