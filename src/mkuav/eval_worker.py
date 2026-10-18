"""Private subprocess entry point for EvaluationService."""
import json
import math
import statistics
import sys
from pathlib import Path

from mkuav.service import EvaluationJob


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(job: EvaluationJob) -> dict:
    import cv2

    from mkuav import metrics_det, metrics_seg
    from mkuav.bench import NUM_THREADS, _percentile

    cv2.setNumThreads(NUM_THREADS)
    # Keep native CPU parallelism bounded within every isolated job, too.
    options = None
    if job.backend != "cv2":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = NUM_THREADS
        options.inter_op_num_threads = 1

    root = Path(job.data_root)
    timings = []
    if job.task == "detection":
        dataset = "visdrone_val" if job.version == "v1" else "visdrone_val_clean"
        annotations = root / dataset / "annotations"
        if not annotations.is_dir():
            raise FileNotFoundError(annotations)
        ignored_json = None if job.version == "v1" else root / dataset / "ignored.json"
        gt, ignored = metrics_det.load_visdrone_gt(annotations, ignored_json)
        stems = sorted(gt)
        if job.subset:
            stems = stems[:job.subset]
        if not stems:
            raise ValueError("detection evaluation split is empty")
        preds = metrics_det.predict_dataset(
            job.model, job.backend, root / "visdrone_val" / "images", stems,
            iou=metrics_det.NMS_IOU, timings_ms=timings, strict=True, sess_options=options,
        )
        metrics = metrics_det.evaluate(
            metrics_det.filter_ignored(preds, ignored), {s: gt[s] for s in stems},
        )
        split = f"visdrone-val-{job.version}"
    else:
        mask_dir = root / "icg" / "masks"
        stems = metrics_seg.holdout_stems(str(mask_dir))
        if job.subset:
            stems = stems[:job.subset]
        if not stems:
            raise ValueError("segmentation holdout is empty")
        handle = metrics_seg.load(job.model, job.backend, sess_options=options)
        result = metrics_seg.evaluate_dataset(
            handle, root / "icg" / "images", mask_dir, stems,
            timings_ms=timings, strict=True,
        )
        names = metrics_seg.load_class_names(root / "icg" / "class_dict_seg.csv")
        metrics = {
            "per_class_iou": {name: float(iou) for name, iou in zip(names, result["per_class_iou"])},
            "miou": result["miou"], "pixel_acc": result["pixel_acc"],
            "num_images": result["num_images"], "num_classes": result["num_classes"],
        }
        split = "icg-holdout"
    return _json_safe({
        "task": job.task, "backend": job.backend, "model": job.model,
        "dataset": {"root": job.data_root, "split": split, "stems": stems},
        "metrics": metrics,
        "inference": {
            "samples_ms": timings, "count": len(timings), "threads": NUM_THREADS,
            "p50_ms": _percentile(timings, 50), "p95_ms": _percentile(timings, 95),
            "mean_ms": statistics.fmean(timings), "total_ms": sum(timings),
        },
    })


if __name__ == "__main__":
    job = EvaluationJob(**json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(evaluate(job), allow_nan=False))
