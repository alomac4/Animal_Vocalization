"""Bounded single-worker jobs and temporary-file lifecycle. No database."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .settings import Settings

LOG = logging.getLogger(__name__)


class CapacityError(Exception):
    pass


@dataclass
class Job:
    id: str
    directory: Path
    status: str = "uploading"
    progress: float | None = None
    error: str | None = None
    detection_count: int = 0
    breed_crop_count: int = 0
    breed_labeled_count: int = 0
    frames: int = 0
    expires_at: float | None = None
    created_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    readers: int = 0
    running: bool = False

    def payload(self) -> dict:
        return {"id": self.id, "status": self.status, "progress": self.progress,
                "error": self.error, "detection_count": self.detection_count,
                "breed_crop_count": self.breed_crop_count, "breed_labeled_count": self.breed_labeled_count,
                "frames": self.frames, "expires_at": self.expires_at,
                "video_path": f"/jobs/{self.id}/video" if self.status == "ready" else None}


class JobManager:
    def __init__(self, settings: Settings, renderer: Callable | None = None):
        self.settings = settings
        self.renderer = renderer or self.render
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bark-inference")
        settings.runtime.mkdir(parents=True, exist_ok=True)
        # Only the dedicated job root is cleared; no checkpoints or user files.
        for child in settings.runtime.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
        self.reaper = threading.Thread(target=self.cleanup_loop, daemon=True)
        self.reaper.start()

    def reserve(self) -> Job:
        with self.lock:
            self.cleanup()
            if len(self.jobs) >= self.settings.max_jobs:
                raise CapacityError("The PC is busy. Finish or clear another video first.")
            job_id = uuid.uuid4().hex
            directory = self.settings.runtime / job_id
            directory.mkdir()
            job = Job(job_id, directory)
            self.jobs[job_id] = job
            return job

    def update(self, job: Job, **values) -> None:
        with self.lock:
            for name, value in values.items():
                setattr(job, name, value)

    def submit(self, job: Job) -> None:
        with self.lock:
            if job.cancel.is_set():
                self.remove_if_safe(job)
                raise RuntimeError("Upload cancelled")
            job.status = "queued"
            self.executor.submit(self.run, job)

    def run(self, job: Job) -> None:
        with self.lock:
            if job.cancel.is_set():
                self.remove_if_safe(job)
                return
            job.running = True
        try:
            self.update(job, status="validating")
            self.renderer(job, self.update)
            if not job.cancel.is_set():
                if not (job.directory / "result.mp4").is_file():
                    raise RuntimeError("Renderer produced no video")
                self.update(job, status="ready", progress=1.0,
                            expires_at=time.time() + self.settings.retention_seconds)
        except Exception as exc:
            LOG.warning("Job %s failed: %s", job.id, exc)
            self.update(job, status="failed", error=str(exc)[:240], progress=None,
                        expires_at=time.time() + self.settings.retention_seconds)
        finally:
            with self.lock:
                job.running = False
                # Retain only the final video, never the uploaded clip/intermediates.
                for file in job.directory.glob("*"):
                    if file.name != "result.mp4" or job.status != "ready":
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                        else:
                            file.unlink(missing_ok=True)
                self.remove_if_safe(job)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            job.cancel.set()
            # The upload request owns input files until its streaming body ends.
            if job.status == "uploading":
                return True
            self.remove_if_safe(job)
            return True

    def remove_if_safe(self, job: Job) -> None:
        if job.cancel.is_set() and not job.running and not job.readers:
            shutil.rmtree(job.directory, ignore_errors=True)
            self.jobs.pop(job.id, None)

    def cleanup(self) -> None:
        with self.lock:
            for job in list(self.jobs.values()):
                if (job.expires_at and job.expires_at <= time.time()) or time.time() - job.created_at > 3600:
                    job.cancel.set()
                if job.status != "uploading":
                    self.remove_if_safe(job)

    def cleanup_loop(self) -> None:
        while not self.stop.wait(15):
            self.cleanup()

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            for job in self.jobs.values():
                job.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for job in list(self.jobs.values()):
            shutil.rmtree(job.directory, ignore_errors=True)
        self.jobs.clear()

    def render(self, job: Job, update: Callable) -> None:
        import imageio_ffmpeg

        if not self.settings.breed_weights or not self.settings.breed_weights.is_file():
            raise RuntimeError("Breed checkpoint missing. Run backend/configure.py before processing.")
        command = [str(self.settings.inference_python), str(Path(__file__).with_name("worker.py")),
                   "--input", str(job.directory / "input.video"), "--output", str(job.directory / "result.mp4"),
                   "--weights", str(self.settings.weights), "--ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(),
                   "--max-duration", str(self.settings.max_duration_seconds),
                   "--device", self.settings.device, "--imgsz", str(self.settings.imgsz),
                   "--breed-weights", str(self.settings.breed_weights), "--breed-conf", str(self.settings.breed_conf)]
        import os
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   PYTHONIOENCODING="utf-8", YOLO_CONFIG_DIR=str(job.directory / "yolo-settings"))
        events: queue.Queue = queue.Queue()
        with (job.directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log,
                                       text=True, encoding="utf-8", env=env)
            reader = threading.Thread(target=lambda: self.read_events(process, events), daemon=True)
            reader.start()
            deadline = time.monotonic() + self.settings.process_timeout_seconds
            failure = None
            try:
                while True:
                    if job.cancel.is_set() or time.monotonic() > deadline:
                        self.kill_process_tree(process)
                        raise RuntimeError("Processing cancelled or timed out. Try a shorter clip.")
                    try:
                        event = events.get(timeout=0.2)
                        if event is None:
                            break
                        if "error" in event:
                            failure = event["error"]
                        else:
                            update(job, **{key: event[key] for key in
                                           ("status", "progress", "frames", "detection_count",
                                            "breed_crop_count", "breed_labeled_count") if key in event})
                    except queue.Empty:
                        if process.poll() is not None and not reader.is_alive():
                            break
                process.wait(timeout=10)
                if process.returncode:
                    raise RuntimeError(failure or "Video processing failed. Try another MP4 or MOV clip.")
            finally:
                if process.poll() is None:
                    self.kill_process_tree(process)
                    process.wait()
                reader.join(timeout=2)
                if process.stdout:
                    process.stdout.close()

    @staticmethod
    def kill_process_tree(process) -> None:
        import os
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        else:
            process.kill()

    @staticmethod
    def read_events(process, events: queue.Queue) -> None:
        try:
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        finally:
            events.put(None)