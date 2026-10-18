"""Process lifecycle tests use a controlled child instead of a vision backend."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from mkuav import service
from mkuav.service import (
    BackendJobError,
    EvaluationJob,
    EvaluationService,
    ServiceBusyError,
)

CHILD = """
import json, os, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[1]).read_text())
config = json.loads(Path(job['model']).read_text())
start = time.perf_counter()
time.sleep(config.get('delay', 0))
if config.get('crash'):
    os._exit(7)
if config.get('fail'):
    raise RuntimeError('controlled backend failure')
Path(sys.argv[2]).write_text(json.dumps({
    'pid': os.getpid(), 'inference': {'total_ms': (time.perf_counter()-start)*1000},
}))
"""


def run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, 8)
    return asyncio.run(bounded())


@pytest.fixture
def children(monkeypatch):
    real_spawn = asyncio.create_subprocess_exec
    processes, requests, concurrency = [], [], []
    launched = asyncio.Event()

    async def spawn(*args, **kwargs):
        child = await real_spawn(sys.executable, "-c", CHILD, *args[-2:], **kwargs)
        concurrency.append(1 + sum(p.returncode is None for p in processes))
        processes.append(child)
        requests.append(args[-2])
        launched.set()
        return child

    monkeypatch.setattr(service.asyncio, "create_subprocess_exec", spawn)
    return processes, requests, concurrency, launched


def job(tmp_path, **config):
    model = tmp_path / "control.json"
    model.write_text(json.dumps(config))
    return EvaluationJob("detection", str(model))


def test_bounded_load_isolation_and_queue_time(tmp_path, children):
    async def scenario():
        request = job(tmp_path, delay=0.08)
        async with EvaluationService(max_concurrent=2, max_queue=4) as api:
            results = await asyncio.gather(*(api.evaluate(request) for _ in range(6)))
        processes, requests, concurrency, _ = children
        assert max(concurrency) == 2
        assert len({r["pid"] for r in results}) == 6
        assert len(set(requests)) == 6
        assert all(p.returncode == 0 for p in processes)
        assert results[-1]["queue_ms"] > results[0]["queue_ms"] + 80
        assert results[-1]["queue_ms"] > results[-1]["inference"]["total_ms"]
        assert all(r["execution_ms"] >= r["inference"]["total_ms"] for r in results)
        assert all(not Path(p).exists() for p in requests)
    run(scenario())


def test_queue_overflow_and_queued_cancellation(tmp_path, children):
    async def scenario():
        request = job(tmp_path, delay=0.15)
        async with EvaluationService(1, 1) as api:
            first = asyncio.create_task(api.evaluate(request))
            await children[3].wait()
            queued = asyncio.create_task(api.evaluate(request))
            await asyncio.sleep(0)
            with pytest.raises(ServiceBusyError):
                await api.evaluate(request)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            replacement = asyncio.create_task(api.evaluate(request))
            await asyncio.gather(first, replacement)
        assert len(children[0]) == 2  # cancelled queued work never spawned
        assert max(children[2]) == 1
    run(scenario())


@pytest.mark.parametrize("queued", [False, True])
def test_deadline_covers_running_and_waiting_jobs(tmp_path, children, queued):
    async def scenario():
        request = job(tmp_path, delay=0.15)
        async with EvaluationService(1, 1) as api:
            first = None
            if queued:
                first = asyncio.create_task(api.evaluate(request))
                await children[3].wait()
            deadline = asyncio.get_running_loop().time() + 0.03
            with pytest.raises(TimeoutError):
                await api.evaluate(request, deadline=deadline)
            if first:
                await first
            else:
                assert children[0][0].returncode is not None
            # The capacity can be reused after a deadline.
            await api.evaluate(request, timeout=1)
        assert len(children[0]) == 2
        assert max(children[2]) == 1
    run(scenario())


def test_running_cancellation_reaps_before_reuse(tmp_path, children):
    async def scenario():
        request = job(tmp_path, delay=1)
        async with EvaluationService(1, 0) as api:
            pending = asyncio.create_task(api.evaluate(request))
            await children[3].wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert children[0][0].returncode is not None
            await api.evaluate(job(tmp_path), timeout=1)
        assert max(children[2]) == 1
    run(scenario())


def test_cancel_during_process_launch(tmp_path, children, monkeypatch):
    async def scenario():
        spawn = service.asyncio.create_subprocess_exec
        release = asyncio.Event()

        async def delayed_spawn(*args, **kwargs):
            proc = await spawn(*args, **kwargs)
            await release.wait()
            return proc

        monkeypatch.setattr(service.asyncio, "create_subprocess_exec", delayed_spawn)
        async with EvaluationService(1, 0) as api:
            pending = asyncio.create_task(api.evaluate(job(tmp_path, delay=1)))
            await children[3].wait()
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            # A repeated cancellation must not abandon the launching child.
            pending.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert children[0][0].returncode is not None
    run(scenario())


@pytest.mark.parametrize("failure", ["fail", "crash"])
def test_worker_failure_does_not_poison_capacity(tmp_path, children, failure):
    async def scenario():
        async with EvaluationService(1, 0) as api:
            with pytest.raises(BackendJobError, match="backend exited"):
                await api.evaluate(job(tmp_path, **{failure: True}))
            assert children[0][0].returncode != 0
            await api.evaluate(job(tmp_path))
        assert children[0][-1].returncode == 0
    run(scenario())


def test_close_cancels_active_and_queued_and_rejects_new_jobs(tmp_path, children):
    async def scenario():
        request = job(tmp_path, delay=1)
        api = EvaluationService(1, 2)
        calls = [asyncio.create_task(api.evaluate(request)) for _ in range(3)]
        await children[3].wait()
        await api.aclose()
        results = await asyncio.gather(*calls, return_exceptions=True)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)
        assert len(children[0]) == 1
        assert children[0][0].returncode is not None
        with pytest.raises(RuntimeError, match="closed"):
            await api.evaluate(request)
        await api.aclose()
    run(scenario())


def test_validation_and_expired_deadline_never_spawn(tmp_path, children):
    with pytest.raises(ValueError):
        EvaluationService(0)
    with pytest.raises(ValueError):
        EvaluationService(1, -1)
    for options in ({"task": "training"}, {"backend": "fake"}, {"subset": -1},
                    {"subset": 1.5}, {"version": "v3"}):
        kwargs = {"task": "detection", "model": "unused", **options}
        with pytest.raises(ValueError):
            EvaluationJob(**kwargs)

    async def scenario():
        async with EvaluationService() as api:
            request = job(tmp_path)
            for kwargs in ({"timeout": 0}, {"timeout": float("nan")},
                           {"deadline": float("inf")}, {"timeout": 1, "deadline": 2}):
                with pytest.raises(ValueError):
                    await api.evaluate(request, **kwargs)
            with pytest.raises(TimeoutError):
                await api.evaluate(request, deadline=asyncio.get_running_loop().time() - 1)
        assert children[0] == []
    run(scenario())
