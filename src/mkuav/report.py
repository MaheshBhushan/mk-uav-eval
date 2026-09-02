"""Report generator: runs (or loads) all evaluation stages into report.md + metrics.json."""
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2

DATASET_TAGS = {
    "v1": "data/visdrone_val (raw annotations)",
    "v2": "data/visdrone_val_clean (cleaned annotations)",
}

LIMITATIONS = [
    "COCO weights are not fine-tuned on VisDrone; class mapping is a bridge, not a match.",
    "VisDrone tricycle and awning-tricycle classes are excluded (no COCO equivalent).",
    "Rotation ground-truth boxes are the enclosing axis-aligned box of the rotated "
    "corners, so rotation mAP is a lower bound on true rotational robustness.",
    "Ignored-region filtering is a centre-point approximation, not exact polygon containment.",
]


def _nan_to_none(obj):
    """Recursively replace float NaN with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cpu_model():
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _versions(backend):
    onnxruntime_version = "n/a"
    try:
        import onnxruntime as ort

        onnxruntime_version = ort.__version__
    except Exception:  # noqa: BLE001
        pass
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "onnxruntime": onnxruntime_version,
        "backend": backend,
    }


def build_run_info(backend, num_images):
    v = _versions(backend)
    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "python": v["python"],
        "opencv": v["opencv"],
        "onnxruntime": v["onnxruntime"],
        "backend": backend,
        "num_images": num_images,
        "git_sha": _git_sha(),
        "dataset_v1": DATASET_TAGS["v1"],
        "dataset_v2": DATASET_TAGS["v2"],
    }


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_nan_to_none(data), f, indent=2, sort_keys=True)


def run_stages(subset, backend, model, skip_seg, json_dir=None):
    """Run every stage in-process and return the stage dicts. Optionally write each
    stage's JSON to `json_dir` so a report can be re-rendered without recomputation."""
    from mkuav import metrics_det, perturb, qa

    class _NS:
        pass

    # --- QA ---
    qa_v1_args = _NS()
    qa_v1_args.dataset, qa_v1_args.version, qa_v1_args.json_path, qa_v1_args.clean = "all", "v1", None, False
    qa_v2_args = _NS()
    qa_v2_args.dataset, qa_v2_args.version, qa_v2_args.json_path, qa_v2_args.clean = "all", "v2", None, False

    visdrone_v1 = qa.run_visdrone("data/visdrone_val/annotations", "data/visdrone_val/images")
    icg = qa.run_icg("data/icg/images", "data/icg/masks", "data/icg/class_dict_seg.csv")
    qa_v1 = {"visdrone": visdrone_v1, "icg": icg}

    visdrone_v2 = qa.run_visdrone("data/visdrone_val_clean/annotations", "data/visdrone_val/images")
    qa_v2 = {"visdrone": visdrone_v2, "icg": icg}

    # --- Detection ---
    gt_v1, ignored_v1 = metrics_det.load_visdrone_gt("data/visdrone_val/annotations", None)
    gt_v2, ignored_v2 = metrics_det.load_visdrone_gt(
        "data/visdrone_val_clean/annotations", "data/visdrone_val_clean/ignored.json"
    )
    stems = sorted(gt_v2)
    if subset:
        stems = stems[:subset]
    gt_v1 = {s: gt_v1.get(s, []) for s in stems}
    gt_v2 = {s: gt_v2.get(s, []) for s in stems}
    ignored_v1 = {s: v for s, v in ignored_v1.items() if s in set(stems)}
    ignored_v2 = {s: v for s, v in ignored_v2.items() if s in set(stems)}

    img_dir = "data/visdrone_val/images"
    cache_path = None
    if json_dir:
        cache_path = str(Path(json_dir) / "pred_cache.json")

    preds = metrics_det.predict_dataset(
        model, backend, img_dir, stems, conf=0.001, iou=metrics_det.NMS_IOU, cache_path=cache_path,
    )

    preds_v1 = metrics_det.filter_ignored(preds, ignored_v1)
    result_v1 = metrics_det.evaluate(preds_v1, gt_v1)
    result_v1.update({"version": "v1", "backend": backend, "model": model, "conf": 0.001, "ignore_regions": True})

    preds_v2 = metrics_det.filter_ignored(preds, ignored_v2)
    result_v2 = metrics_det.evaluate(preds_v2, gt_v2)
    result_v2.update({"version": "v2", "backend": backend, "model": model, "conf": 0.001, "ignore_regions": True})

    result_v2_noignore = metrics_det.evaluate(preds, gt_v2)
    result_v2_noignore.update(
        {"version": "v2", "backend": backend, "model": model, "conf": 0.001, "ignore_regions": False}
    )

    # --- Bench ---
    from mkuav import bench as bench_mod

    bench_img = cv2.imread("/var/tmp/bus.jpg")
    if bench_img is None:
        bench_img = cv2.imread(str(Path(img_dir) / f"{stems[0]}.jpg")) if stems else None
    iters = 20 if subset and subset <= 50 else 50
    bench_backends = ["cv2", "ort-cpu", "ort-openvino"]
    bench_result = bench_mod.bench(model, bench_img, bench_backends, warmup=5, iters=iters)
    gpu_result = bench_mod._bench_gpu_in_subprocess(model, "/var/tmp/bus.jpg", 5, iters)
    if "unavailable" not in gpu_result:
        bench_result["ort-openvino-gpu"] = gpu_result

    # --- Robustness ---
    robust_result = perturb.run_robustness(
        model, backend, stems, gt_v2, ignored_v2, img_dir, list(perturb.PERTURBATIONS), conf=0.001,
    )
    baseline = robust_result[list(perturb.PERTURBATIONS)[0]]["0"] if robust_result else None
    params = {
        "blur": [perturb.BLUR_KSIZE[i] for i in (1, 2, 3)],
        "brightness_down": [perturb.BRIGHTNESS_DOWN_BETA[i] for i in (1, 2, 3)],
        "brightness_up": [perturb.BRIGHTNESS_UP_BETA[i] for i in (1, 2, 3)],
        "jpeg": [perturb.JPEG_QUALITY[i] for i in (1, 2, 3)],
        "rotate": [perturb.ROTATE_DEGREES[i] for i in (1, 2, 3)],
    }
    robust = {
        "num_images": len(stems),
        "backend": backend,
        "baseline": baseline,
        "results": robust_result,
        "params": params,
    }

    # --- Segmentation (optional) ---
    seg = None
    if not skip_seg and Path("models/seg_unet_r18.onnx").exists():
        try:
            from mkuav import metrics_seg

            seg = metrics_seg.run_for_report(
                "models/seg_unet_r18.onnx", "data/icg/images", "data/icg/masks",
                "data/icg/class_dict_seg.csv", stems, backend,
            )
        except ImportError:
            seg = None

    stages = {
        "qa_v1": qa_v1,
        "qa_v2": qa_v2,
        "eval_det_v1": result_v1,
        "eval_det_v2": result_v2,
        "eval_det_v2_noignore": result_v2_noignore,
        "bench": bench_result,
        "robust": robust,
        "eval_seg": seg,
    }

    if json_dir:
        for name, data in stages.items():
            if name == "eval_seg" and data is None:
                continue
            _dump_json(Path(json_dir) / f"{name}.json", data)

    return stages, stems


def load_stages(from_dir):
    from_dir = Path(from_dir)

    def _maybe(name):
        p = from_dir / f"{name}.json"
        return _load_json(p) if p.exists() else None

    stages = {
        "qa_v1": _load_json(from_dir / "qa_v1.json"),
        "qa_v2": _load_json(from_dir / "qa_v2.json"),
        "eval_det_v1": _load_json(from_dir / "eval_det_v1.json"),
        "eval_det_v2": _load_json(from_dir / "eval_det_v2.json"),
        "eval_det_v2_noignore": _load_json(from_dir / "eval_det_v2_noignore.json"),
        "bench": _load_json(from_dir / "bench.json"),
        "robust": _load_json(from_dir / "robust.json"),
        "eval_seg": _maybe("eval_seg"),
    }
    num_images = stages["eval_det_v2"].get("num_images", 0)
    return stages, num_images


def _md_qa_section(qa_v1, qa_v2):
    lines = ["## Annotation QA", ""]
    lines.append("### VisDrone")
    lines.append("| rule | v1 count | v1 files | v2 count | v2 files |")
    lines.append("|---|---|---|---|---|")
    v1c = qa_v1["visdrone"]["counts"]
    v1f = qa_v1["visdrone"]["files_affected"]
    v2c = qa_v2["visdrone"]["counts"]
    v2f = qa_v2["visdrone"]["files_affected"]
    for rule in sorted(set(v1c) | set(v2c)):
        lines.append(
            f"| {rule} | {v1c.get(rule, 0)} | {len(v1f.get(rule, []))} "
            f"| {v2c.get(rule, 0)} | {len(v2f.get(rule, []))} |"
        )
    lines.append("")
    lines.append("### ICG")
    lines.append("| rule | count | files |")
    lines.append("|---|---|---|")
    icg = qa_v1["icg"]
    for rule in sorted(icg["counts"]):
        lines.append(f"| {rule} | {icg['counts'][rule]} | {len(icg['files_affected'].get(rule, []))} |")
    lines.append("")
    lines.append("### Class histogram (top-8, ICG pixels)")
    lines.append("| class id | pixel count |")
    lines.append("|---|---|")
    top8 = sorted(icg["class_hist"].items(), key=lambda kv: -kv[1])[:8]
    for cls_id, cnt in top8:
        lines.append(f"| {cls_id} | {cnt} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(v, spec="{:.4f}"):
    return spec.format(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else "nan"


def _md_detection_section(v1, v2, v2ni):
    lines = ["## Detection", ""]
    lines.append("### Per-class (v2)")
    lines.append("| class | n_gt | AP@0.5 | AP@0.5:0.95 | P@op | R@op |")
    lines.append("|---|---|---|---|---|---|")
    for name in v2["classes"]:
        c = v2["per_class"][name]
        lines.append(
            f"| {name} | {c['n_gt']} | {_fmt(c['ap50'])} | {_fmt(c['ap'])} "
            f"| {_fmt(c['precision_op'], '{:.3f}')} | {_fmt(c['recall_op'], '{:.3f}')} |"
        )
    lines.append("")
    lines.append("### Summary")
    lines.append("| version | mAP@0.5 | mAP@0.5:0.95 | P | R |")
    lines.append("|---|---|---|---|---|")
    for tag, r in (("v1", v1), ("v2", v2), ("v2-no-ignore", v2ni)):
        o = r["overall"]
        lines.append(
            f"| {tag} | {r['map50']:.4f} | {r['map50_95']:.4f} "
            f"| {o['precision_op']:.3f} | {o['recall_op']:.3f} |"
        )
    lines.append("")
    if abs(v1["map50"] - v2["map50"]) < 1e-9 and abs(v1["map50_95"] - v2["map50_95"]) < 1e-9:
        lines.append(
            "v1 and v2 detection metrics are identical: the only cleaning applied here "
            "is dropping category 0/11 rows that the loader already ignores."
        )
        lines.append("")
    return "\n".join(lines)


def _md_latency_section(bench_result):
    lines = ["## Latency", ""]
    lines.append("| backend | p50_ms | p95_ms | mean_ms | threads |")
    lines.append("|---|---|---|---|---|")
    for name, r in bench_result.items():
        if "unavailable" in r:
            continue
        lines.append(f"| {name} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['mean_ms']:.2f} | {r['threads']} |")
    lines.append("")
    if "ort-openvino-gpu" not in bench_result:
        lines.append("ort-openvino-gpu is unavailable on this machine.")
        lines.append("")
    return "\n".join(lines)


def _md_robustness_section(robust):
    lines = ["## Robustness", ""]
    baseline = robust["baseline"]
    baseline_map50 = baseline["map50"] if baseline else float("nan")
    lines.append("| perturbation | severity | param | mAP@0.5 | drop vs baseline (%) |")
    lines.append("|---|---|---|---|---|")
    for name, sevs in robust["results"].items():
        params = robust["params"].get(name, [])
        for sev_str in sorted(sevs, key=int):
            sev = int(sev_str)
            r = sevs[sev_str]
            param = params[sev - 1] if sev > 0 and sev - 1 < len(params) else "-"
            map50 = r["map50"]
            drop = (
                (baseline_map50 - map50) / baseline_map50 * 100.0
                if baseline_map50 else float("nan")
            )
            lines.append(f"| {name} | {sev} | {param} | {map50:.4f} | {drop:.1f} |")
    lines.append("")
    return "\n".join(lines)


def _md_segmentation_section(seg):
    lines = ["## Segmentation", ""]
    if seg is None:
        lines.append("not available in this run: seg_unet_r18.onnx model or metrics_seg module missing.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| class | IoU |")
    lines.append("|---|---|")
    for name, iou in seg.get("per_class_iou", {}).items():
        lines.append(f"| {name} | {iou:.4f} |")
    lines.append("")
    lines.append(f"mIoU: {seg.get('miou', float('nan')):.4f}  pixel acc: {seg.get('pixel_acc', float('nan')):.4f}")
    lines.append("")
    return "\n".join(lines)


def _md_limitations_section():
    lines = ["## Limitations", ""]
    for item in LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _md_run_info_section(run_info):
    lines = ["## Run info", ""]
    lines.append(f"| date_utc | {run_info['date_utc']} |")
    lines.append("|---|---|")
    lines.append(f"| hostname | {run_info['hostname']} |")
    lines.append(f"| cpu_model | {run_info['cpu_model']} |")
    lines.append(f"| python | {run_info['python']} |")
    lines.append(f"| opencv | {run_info['opencv']} |")
    lines.append(f"| onnxruntime | {run_info['onnxruntime']} |")
    lines.append(f"| backend | {run_info['backend']} |")
    lines.append(f"| num_images | {run_info['num_images']} |")
    lines.append(f"| git_sha | {run_info['git_sha']} |")
    lines.append(f"| dataset_v1 | {run_info['dataset_v1']} |")
    lines.append(f"| dataset_v2 | {run_info['dataset_v2']} |")
    lines.append("")
    return "\n".join(lines)


def render(stages, run_info):
    parts = [
        "# mk-uav-eval report",
        "",
        _md_run_info_section(run_info),
        _md_qa_section(stages["qa_v1"], stages["qa_v2"]),
        _md_detection_section(stages["eval_det_v1"], stages["eval_det_v2"], stages["eval_det_v2_noignore"]),
        _md_latency_section(stages["bench"]),
        _md_robustness_section(stages["robust"]),
        _md_segmentation_section(stages["eval_seg"]),
        _md_limitations_section(),
    ]
    return "\n".join(parts)


def build_metrics(stages, run_info):
    return _nan_to_none({
        "run": run_info,
        "qa": {"v1": stages["qa_v1"], "v2": stages["qa_v2"]},
        "detection": {
            "v1": stages["eval_det_v1"],
            "v2": stages["eval_det_v2"],
            "v2_noignore": stages["eval_det_v2_noignore"],
        },
        "latency": stages["bench"],
        "robustness": stages["robust"],
        "segmentation": stages["eval_seg"],
    })


def main(args) -> int:
    if args.from_dir:
        stages, num_images = load_stages(args.from_dir)
        run_info = build_run_info(stages["eval_det_v2"].get("backend", args.backend), num_images)
    else:
        stages, stems = run_stages(args.subset, args.backend, args.model, args.skip_seg, args.json_dir)
        run_info = build_run_info(args.backend, len(stems))

    md = render(stages, run_info)
    metrics = build_metrics(stages, run_info)

    with open(args.out, "w") as f:
        f.write(md)
    with open(args.json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")
    print(f"wrote {args.json}")
    return 0
