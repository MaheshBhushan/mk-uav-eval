"""Bounded asyncio API for isolated numerical evaluations. No network listener."""
import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class EvaluationJob:
    task: str
    model: str
    backend: str = "ort"
    data_root: str = "data"
    version: str = "v2"
    subset: int = 0

    def __post_init__(self):
        if self.task not in ("detection", "segmentation"):
            raise ValueError("task must be detection or segmentation")
        if self.backend not in ("cv2", "ort", "ort-cpu", "ort-openvino", "ort-openvino-gpu"):
            raise ValueError("unknown backend")
        if self.version not in ("v1", "v2"):
            raise ValueError("version must be v1 or v2")
        if type(self.subset) is not int or self.subset < 0:
            raise ValueError("subset must be a nonnegative integer")
        # Bind paths at submission configuration time, before any await.
        object.__setattr__(self, "model", str(Path(self.model).resolve()))
        object.__setattr__(self, "data_root", str(Path(self.data_root).resolve()))


class ServiceBusyError(RuntimeError):
    """The active and waiting job capacity is exhausted."""


class BackendJobError(RuntimeError):
    """An isolated backend failed or crashed."""


async def _finish(task):
    """Finish process cleanup even if another cancellation arrives."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class EvaluationService:
    """Use on one event loop; each job owns a fresh subprocess and model handle.

    Capacity includes max_concurrent active jobs and max_queue waiting jobs.
    Excess submissions fail immediately. Timeout/deadline covers queueing and
    execution; cancellation kills and reaps the child before releasing capacity.
    """

    def __init__(self, max_concurrent: int = 2, max_queue: int = 8):
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a nonnegative integer")
        self._slots = asyncio.Semaphore(max_concurrent)
        self._capacity = max_concurrent + max_queue
        self._jobs = set()
        self._closed = False

    async def evaluate(self, job: EvaluationJob, *, timeout: float | None = None,
                       deadline: float | None = None) -> dict:
        """Return metrics, split identity and timings in milliseconds.

        timeout is a positive number of seconds from submission. deadline is
        an absolute asyncio loop.time() value. Supply at most one of them.
        """
        if self._closed:
            raise RuntimeError("evaluation service is closed")
        if not isinstance(job, EvaluationJob):
            raise TypeError("job must be an EvaluationJob")
        if timeout is not None and (
            deadline is not None or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("supply a positive finite timeout or a deadline")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        loop = asyncio.get_running_loop()
        submitted = loop.time()
        if timeout is not None:
            deadline = submitted + timeout
        if deadline is not None and deadline <= submitted:
            raise TimeoutError("evaluation deadline has expired")
        if len(self._jobs) >= self._capacity:
            raise ServiceBusyError("evaluation capacity exhausted")
        task = asyncio.create_task(self._execute(job, submitted))
        self._jobs.add(task)
        try:
            async with asyncio.timeout_at(deadline):
                return await task
        finally:
            self._jobs.discard(task)

    async def _execute(self, job, submitted):
        async with self._slots:
            loop = asyncio.get_running_loop()
            started = loop.time()
            with TemporaryDirectory(prefix="mkuav-job-") as directory:
                root = Path(directory)
                request, result = root / "request.json", root / "result.json"
                request.write_text(json.dumps(asdict(job)))
                with (root / "worker.log").open("w+b") as log:
                    launch = asyncio.create_task(asyncio.create_subprocess_exec(
                        sys.executable, "-m", "mkuav.eval_worker", str(request), str(result),
                        stdout=log, stderr=log,
                    ))
                    try:
                        proc = await asyncio.shield(launch)
                        code = await proc.wait()
                        if code != 0:
                            log.seek(0, 2)
                            log.seek(max(0, log.tell() - 4096))
                            detail = log.read().decode(errors="replace").strip()
                            raise BackendJobError(f"backend exited {code}: {detail}")
                        try:
                            payload = json.loads(result.read_text())
                        except (OSError, ValueError) as exc:
                            raise BackendJobError("backend returned no valid result") from exc
                        payload["queue_ms"] = (started - submitted) * 1000
                        payload["execution_ms"] = (loop.time() - started) * 1000
                        return payload
                    finally:
                        async def stop():
                            child = await launch
                            if child.returncode is None:
                                try:
                                    child.kill()
                                except ProcessLookupError:
                                    pass
                            await child.wait()

                        await _finish(asyncio.create_task(stop()))

    async def aclose(self):
        """Reject new work and cancel/reap all active and queued jobs."""
        self._closed = True
        jobs = list(self._jobs)
        for task in jobs:
            if not task.cancelling():
                task.cancel()
        await _finish(asyncio.ensure_future(asyncio.gather(*jobs, return_exceptions=True)))

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("evaluation service is closed")
        return self

    async def __aexit__(self, *_):
        await self.aclose()
