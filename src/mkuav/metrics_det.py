"""Detection metrics (precision/recall/AP) for VisDrone evaluated with a COCO-trained model.

COCO YOLOv8 predicts 80 classes; VisDrone annotates 10. CLASS_MAP below is the only
place that bridge lives:

    COCO 0  person     -> VisDrone {1 pedestrian, 2 people}  merged as "person"
    COCO 1  bicycle    -> VisDrone {3 bicycle}               as "bicycle"
    COCO 2  car        -> VisDrone {4 car, 5 van}    merged  as "car"
    COCO 3  motorcycle -> VisDrone {10 motor}                as "motor"
    COCO 5  bus        -> VisDrone {9 bus}                   as "bus"
    COCO 7  truck      -> VisDrone {6 truck}                 as "truck"

VisDrone categories 7 (tricycle) and 8 (awning-tricycle) have no COCO equivalent and
are dropped from the ground truth entirely; predictions of any COCO class not listed
above are discarded before matching. Evaluation therefore runs over the 6 merged
classes in CLASSES.

numpy only: pycocotools/torchmetrics are used in the tests, never here.
"""
import json
from pathlib import Path

import numpy as np

CLASSES = ("person", "bicycle", "car", "motor", "bus", "truck")

# COCO class id -> merged class name.
COCO_TO_CLASS = {0: "person", 1: "bicycle", 2: "car", 3: "motor", 5: "bus", 7: "truck"}
# VisDrone category id -> merged class name (7 and 8 intentionally absent).
VISDRONE_TO_CLASS = {
    1: "person",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "car",
    6: "truck",
    9: "bus",
    10: "motor",
}
CLASS_MAP = {"coco": COCO_TO_CLASS, "visdrone": VISDRONE_TO_CLASS}

DEFAULT_IOU_THRS = np.arange(0.5, 0.96, 0.05)
MAX_DETS = 100
NMS_IOU = 0.6


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes. Returns shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match(preds, gts, iou_thr):
    """Greedy one-to-one matching of `preds` to `gts` for a single image and class.

    Predictions are visited in descending score; each takes the highest-IoU ground
    truth still free. Returns (tp_flags, scores, n_gt).
    """
    scores = np.array([p["score"] for p in preds], dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    tp = np.zeros(len(preds), dtype=bool)
    n_gt = len(gts)
    if len(preds) == 0:
        return tp, scores, n_gt
    ious = iou_matrix([preds[i]["bbox"] for i in order], [g["bbox"] for g in gts])
    taken = np.zeros(n_gt, dtype=bool)
    for i in range(len(order)):
        best_j, best_iou = -1, iou_thr - 1e-12
        for j in range(n_gt):
            if taken[j] or ious[i, j] <= best_iou:
                continue
            best_j, best_iou = j, ious[i, j]
        if best_j >= 0:
            taken[best_j] = True
            tp[i] = True
    return tp, scores[order], n_gt


def average_precision(tp, scores, n_gt) -> float:
    """COCO-style 101-point interpolated AP. `tp`/`scores` must be score-sorted."""
    if n_gt == 0:
        return float("nan")
    tp = np.asarray(tp, dtype=np.float64)
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    rc = tp_cum / n_gt
    pr = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    # precision envelope: running max from the right
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    rec_thrs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rc, rec_thrs, side="left")
    q = np.zeros(101, dtype=np.float64)
    valid = idx < len(pr)
    q[valid] = pr[idx[valid]]
    return float(q.mean())


def _by_class(items, key):
    out = {name: [] for name in CLASSES}
    for it in items:
        name = key(it)
        if name is not None:
            out[name].append(it)
    return out


def _pred_class(pred):
    return COCO_TO_CLASS.get(int(pred["class_id"]))


def _topk(preds, k=MAX_DETS):
    if len(preds) <= k:
        return list(preds)
    order = np.argsort([-p["score"] for p in preds], kind="mergesort")[:k]
    return [preds[i] for i in order]


def evaluate(pred_by_image, gt_by_image, iou_thrs=DEFAULT_IOU_THRS, op_conf=0.25, max_dets=MAX_DETS):
    """Full COCO-style evaluation over the 6 merged classes.

    Returns per-class AP@0.5 / AP@0.5:0.95, the two mAPs, and precision/recall at
    `op_conf` per class and overall.
    """
    iou_thrs = np.asarray(iou_thrs, dtype=np.float64)
    stems = sorted(set(gt_by_image) | set(pred_by_image))

    # per class: list of (tp_flags per iou_thr, scores) plus gt counts
    acc = {name: {"tp": [[] for _ in iou_thrs], "scores": [], "n_gt": 0} for name in CLASSES}
    for stem in stems:
        preds = _topk([p for p in pred_by_image.get(stem, []) if _pred_class(p) is not None], max_dets)
        p_cls = _by_class(preds, _pred_class)
        g_cls = _by_class(gt_by_image.get(stem, []), lambda g: g["class_name"])
        for name in CLASSES:
            p, g = p_cls[name], g_cls[name]
            acc[name]["n_gt"] += len(g)
            if not p:
                continue
            sorted_scores = None
            for t, thr in enumerate(iou_thrs):
                tp, sc, _ = match(p, g, float(thr))
                acc[name]["tp"][t].append(tp)
                sorted_scores = sc
            acc[name]["scores"].append(sorted_scores)

    per_class = {}
    for name in CLASSES:
        a = acc[name]
        n_gt = a["n_gt"]
        if a["scores"]:
            scores = np.concatenate(a["scores"])
            order = np.argsort(-scores, kind="mergesort")
            scores = scores[order]
            tps = [np.concatenate(a["tp"][t])[order] for t in range(len(iou_thrs))]
        else:
            scores = np.zeros(0)
            tps = [np.zeros(0, dtype=bool) for _ in iou_thrs]
        aps = [average_precision(tp, scores, n_gt) for tp in tps]
        keep = scores >= op_conf
        tp_op = int(tps[0][keep].sum()) if len(scores) else 0
        n_pred_op = int(keep.sum())
        per_class[name] = {
            "ap50": aps[0],
            "ap": float(np.mean(aps)) if n_gt else float("nan"),
            "n_gt": n_gt,
            "n_pred_op": n_pred_op,
            "tp_op": tp_op,
            "precision_op": tp_op / n_pred_op if n_pred_op else 0.0,
            "recall_op": tp_op / n_gt if n_gt else 0.0,
        }

    present = [n for n in CLASSES if per_class[n]["n_gt"] > 0]
    map50 = float(np.mean([per_class[n]["ap50"] for n in present])) if present else 0.0
    map5095 = float(np.mean([per_class[n]["ap"] for n in present])) if present else 0.0
    tot_tp = sum(per_class[n]["tp_op"] for n in CLASSES)
    tot_pred = sum(per_class[n]["n_pred_op"] for n in CLASSES)
    tot_gt = sum(per_class[n]["n_gt"] for n in CLASSES)
    return {
        "classes": list(CLASSES),
        "per_class": per_class,
        "map50": map50,
        "map50_95": map5095,
        "iou_thrs": [float(t) for t in iou_thrs],
        "op_conf": float(op_conf),
        "max_dets": int(max_dets),
        "num_images": len(stems),
        "overall": {
            "precision_op": tot_tp / tot_pred if tot_pred else 0.0,
            "recall_op": tot_tp / tot_gt if tot_gt else 0.0,
            "n_gt": tot_gt,
            "n_pred_op": tot_pred,
            "tp_op": tot_tp,
        },
    }


def load_visdrone_gt(ann_dir, ignored_json=None):
    """Parse VisDrone annotation .txt files into mapped ground truth.

    Works on both v1 (raw, contains category 0 ignored-region rows and category 11
    "others" rows) and v2 (cleaned): any row whose category has no COCO counterpart
    is dropped. For v1, category-0 rows are collected as ignored regions when no
    ignored.json is supplied. Returns (gt_by_image, ignored_by_image).
    """
    ann_dir = Path(ann_dir)
    gt_by_image = {}
    ignored_by_image = {}
    for ann_path in sorted(ann_dir.glob("*.txt")):
        stem = ann_path.stem
        rows = []
        ignored = []
        for raw in ann_path.read_text().splitlines():
            fields = raw.strip().split(",")
            if len(fields) < 8:
                continue
            try:
                left, top, w, h, score, category = (int(f) for f in fields[:6])
            except ValueError:
                continue
            if category == 0:
                ignored.append([left, top, w, h])
                continue
            name = VISDRONE_TO_CLASS.get(category)
            if name is None or w <= 0 or h <= 0:
                continue
            rows.append(
                {
                    "bbox": [float(left), float(top), float(left + w), float(top + h)],
                    "class_name": name,
                    "category": category,
                }
            )
        gt_by_image[stem] = rows
        if ignored:
            ignored_by_image[stem] = ignored
    if ignored_json is not None:
        with open(ignored_json) as f:
            ignored_by_image = json.load(f)
    return gt_by_image, ignored_by_image


def filter_ignored(pred_by_image, ignored_by_image):
    """Drop predictions whose box centre falls inside any ignored region."""
    out = {}
    for stem, preds in pred_by_image.items():
        regions = ignored_by_image.get(stem)
        if not regions:
            out[stem] = list(preds)
            continue
        kept = []
        for p in preds:
            x1, y1, x2, y2 = p["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(rx <= cx <= rx + rw and ry <= cy <= ry + rh for rx, ry, rw, rh in regions):
                continue
            kept.append(p)
        out[stem] = kept
    return out


def predict_dataset(model_path, backend, img_dir, stems, conf=0.001, iou=0.6, cache_path=None):
    """Run detection over `stems`, optionally caching the raw predictions to JSON."""
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if set(stems) <= set(cached):
            return {s: cached[s] for s in stems}

    import cv2

    from mkuav import detect

    handle = detect.load(model_path, backend)
    img_dir = Path(img_dir)
    out = {}
    for stem in stems:
        img = cv2.imread(str(img_dir / f"{stem}.jpg"))
        if img is None:
            out[stem] = []
            continue
        out[stem] = detect.run(handle, img, conf=conf, iou=iou)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)
    return out


def main(args) -> int:
    version = args.version
    if version == "v1":
        ann_dir = "data/visdrone_val/annotations"
        ignored_json = None
    else:
        ann_dir = "data/visdrone_val_clean/annotations"
        ignored_json = "data/visdrone_val_clean/ignored.json"
    img_dir = "data/visdrone_val/images"

    gt_by_image, ignored_by_image = load_visdrone_gt(ann_dir, ignored_json)
    stems = sorted(gt_by_image)
    if args.subset:
        stems = stems[: args.subset]
        gt_by_image = {s: gt_by_image[s] for s in stems}

    preds = predict_dataset(
        args.model, args.backend, img_dir, stems, conf=args.conf, iou=NMS_IOU,
        cache_path=args.pred_cache,
    )
    if not args.no_ignore:
        preds = filter_ignored(preds, ignored_by_image)

    result = evaluate(preds, gt_by_image, op_conf=args.op_conf)
    result["version"] = version
    result["backend"] = args.backend
    result["model"] = args.model
    result["conf"] = args.conf
    result["ignore_regions"] = not args.no_ignore

    print(f"{'class':<10}{'n_gt':>8}{'AP@0.5':>10}{'AP@0.5:0.95':>14}{'P@op':>8}{'R@op':>8}")
    for name in CLASSES:
        c = result["per_class"][name]
        print(
            f"{name:<10}{c['n_gt']:>8}{c['ap50']:>10.4f}{c['ap']:>14.4f}"
            f"{c['precision_op']:>8.3f}{c['recall_op']:>8.3f}"
        )
    o = result["overall"]
    print(f"{'overall':<10}{o['n_gt']:>8}{'':>10}{'':>14}{o['precision_op']:>8.3f}{o['recall_op']:>8.3f}")
    print(f"\nimages={result['num_images']} op_conf={args.op_conf} ignore_regions={not args.no_ignore}")
    print(f"mAP@0.5={result['map50']:.4f}  mAP@0.5:0.95={result['map50_95']:.4f}")

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {args.json_path}")
    return 0
