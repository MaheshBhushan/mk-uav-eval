"""Latency benchmark for detect.run across backends."""
import json
import statistics
import subprocess
import sys
import time

import cv2

from mkuav import detect

NUM_THREADS = 4


def _percentile(values, pct):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return values[idx]


def _time_backend(model_path, image_bgr, backend, warmup, iters):
    cv2.setNumThreads(NUM_THREADS)
    sess_options = None
    if backend != "cv2":
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = NUM_THREADS

    if backend == "cv2":
        handle = detect.load(model_path, "cv2")
    else:
        handle = detect.load(model_path, backend, sess_options=sess_options)

    providers_first = None
    if backend != "cv2":
        _, session = handle
        provs = session.get_providers()
        providers_first = provs[0] if provs else None

    for _ in range(warmup):
        detect.run(handle, image_bgr)

    durations_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        detect.run(handle, image_bgr)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "p50_ms": _percentile(durations_ms, 50),
        "p95_ms": _percentile(durations_ms, 95),
        "mean_ms": statistics.fmean(durations_ms),
        "iters": iters,
        "threads": NUM_THREADS,
    }
    if providers_first is not None:
        result["providers"] = provs
    return result


def bench(model_path, image_bgr, backends, warmup=20, iters=100) -> dict:
    """Measure per-backend latency of detect.run. Returns a dict keyed by backend name."""
    results = {}
    for backend in backends:
        try:
            results[backend] = _time_backend(model_path, image_bgr, backend, warmup, iters)
        except Exception as exc:  # noqa: BLE001 - report any init/runtime failure per backend
            results[backend] = {"unavailable": str(exc)}
    return results


def _bench_gpu_in_subprocess(model_path, image_path, warmup, iters):
    """Run the ort-openvino-gpu probe in a subprocess: OpenVINO GPU init can
    segfault the process outright on unsupported hardware, which a Python
    try/except cannot catch."""
    script = (
        "import cv2, json\n"
        "from mkuav import bench\n"
        f"image = cv2.imread({image_path!r})\n"
        f"r = bench.bench({model_path!r}, image, ['ort-openvino-gpu'], warmup={warmup}, iters={iters})\n"
        "print(json.dumps(r['ort-openvino-gpu']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        reason = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
        return {"unavailable": f"subprocess crashed: {reason}"}
    try:
        return json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return {"unavailable": f"could not parse subprocess output: {exc}"}


def main(args) -> int:
    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read image {args.image}")
        return 2

    backends = ["cv2", "ort-cpu", "ort-openvino"]
    results = bench(args.model, image, backends, warmup=args.warmup, iters=args.iters)

    gpu_result = _bench_gpu_in_subprocess(args.model, args.image, args.warmup, args.iters)
    if "unavailable" not in gpu_result:
        results["ort-openvino-gpu"] = gpu_result

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{'backend':<20}{'p50_ms':>10}{'p95_ms':>10}{'mean_ms':>10}{'iters':>8}")
    for name, r in results.items():
        if "unavailable" in r:
            print(f"{name:<20}unavailable: {r['unavailable']}")
        else:
            print(f"{name:<20}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['mean_ms']:>10.2f}{r['iters']:>8}")
    return 0
