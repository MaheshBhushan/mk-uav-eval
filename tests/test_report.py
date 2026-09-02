import json

from mkuav import report

REQUIRED_KEYS = {"run", "qa", "detection", "latency", "robustness", "segmentation"}


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _make_stub_dir(tmp_path):
    qa_stub = {
        "visdrone": {"counts": {}, "files_affected": {}, "class_hist": {}, "num_files": 1},
        "icg": {"counts": {}, "files_affected": {}, "class_hist": {}, "num_files": 1},
    }
    per_class = {
        "person": {"ap50": 0.5, "ap": 0.3, "n_gt": 1, "precision_op": 1.0, "recall_op": 1.0},
    }
    det_stub = {
        "classes": ["person"],
        "per_class": per_class,
        "map50": 0.5,
        "map50_95": 0.3,
        "overall": {"precision_op": 1.0, "recall_op": 1.0, "n_gt": 1, "n_pred_op": 1, "tp_op": 1},
        "num_images": 1,
        "backend": "ort",
    }
    det_nan_stub = json.loads(json.dumps(det_stub))
    bench_stub = {"cv2": {"p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.5, "threads": 4, "iters": 20}}
    robust_stub = {
        "baseline": {"map50": 0.5},
        "params": {"blur": [3, 7, 13]},
        "results": {"blur": {"0": {"map50": 0.5}, "1": {"map50": 0.4}}},
    }

    _write(tmp_path / "qa_v1.json", qa_stub)
    _write(tmp_path / "qa_v2.json", qa_stub)
    _write(tmp_path / "eval_det_v1.json", det_stub)
    _write(tmp_path / "eval_det_v2.json", det_stub)
    _write(tmp_path / "eval_det_v2_noignore.json", det_nan_stub)
    _write(tmp_path / "bench.json", bench_stub)
    _write(tmp_path / "robust.json", robust_stub)
    return qa_stub, det_stub, bench_stub, robust_stub


def test_render_from_dir(tmp_path):
    _make_stub_dir(tmp_path)
    stages, num_images = report.load_stages(tmp_path)
    run_info = report.build_run_info("ort", num_images)

    md = report.render(stages, run_info)
    metrics = report.build_metrics(stages, run_info)

    for header in (
        "## Run info", "## Annotation QA", "## Detection", "## Latency",
        "## Robustness", "## Segmentation", "## Limitations",
    ):
        assert header in md

    (tmp_path / "report.md").write_text(md)
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))

    parsed = json.loads((tmp_path / "metrics.json").read_text())
    assert set(parsed) == REQUIRED_KEYS
    assert parsed["segmentation"] is None


def test_nan_becomes_null():
    data = {"a": float("nan"), "b": [1.0, float("nan")], "c": {"d": float("nan")}}
    cleaned = report._nan_to_none(data)
    dumped = json.dumps(cleaned)
    reloaded = json.loads(dumped)
    assert reloaded["a"] is None
    assert reloaded["b"][1] is None
    assert reloaded["c"]["d"] is None


def test_segmentation_section_handles_null_iou(tmp_path):
    from mkuav import report
    seg = {"per_class_iou": {"tree": 0.5, "conflicting": None}, "miou": 0.5, "pixel_acc": 0.9, "num_images": 2}
    md = report._md_segmentation_section(seg)
    assert "| conflicting | nan |" in md
    assert "mIoU: 0.5000" in md
