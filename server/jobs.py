"""Background jobs for slow work (OCR, answer-key generation, grading).

One worker thread: every job shares the single 8 GB GPU, and the VRAM budget was measured with
one set of models loaded, so jobs run one at a time in submission order. Each job reports
progress that the UI polls via GET /api/jobs/{id}.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    kind: str
    status: Status = "queued"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def report(self, progress: float, message: str = "") -> None:
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message


class JobRunner:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradeforge-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                job.progress, job.status = 1.0, "done"
            except Exception as e:  # surfaced to the UI, never swallowed
                job.error = f"{type(e).__name__}: {e}"
                job.message = traceback.format_exc(limit=3)
                job.status = "failed"

        self._executor.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
