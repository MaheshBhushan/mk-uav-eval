"""Segmentation metrics (per-class IoU, mIoU, pixel accuracy) via confusion matrix.

numpy only: torchmetrics is used to cross-check in the tests, never here.
"""
import csv
import glob
from pathlib import Path

import cv2
import numpy as np

from mkuav.segment import NUM_CLASSES, load, run

IGNORE = 255


def confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES, ignore: int = IGNORE) -> np.ndarray:
    """Accumulate a (num_classes, num_classes) confusion matrix: rows=gt, cols=pred."""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    valid = (pred != ignore) & (gt < num_classes) & (gt >= 0)
    pred, gt = pred[valid], gt[valid]
    idx = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray):
    """Return (per_class_iou, miou, pixel_acc) from an accumulated confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    gt_totals = cm.sum(axis=1).astype(np.float64)
    pred_totals = cm.sum(axis=0).astype(np.float64)
    union = gt_totals + pred_totals - tp

    present = (gt_totals > 0) | (pred_totals > 0)
    per_class_iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_iou[present] = tp[present] / union[present]

    miou = float(np.nanmean(per_class_iou)) if present.any() else float("nan")
    total = cm.sum()
    pixel_acc = float(tp.sum() / total) if total > 0 else float("nan")
    return per_class_iou, miou, pixel_acc


def load_class_names(csv_path: str) -> list[str]:
    names = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
    return names


def evaluate_dataset(handle, img_dir: str, mask_dir: str, stems: list[str]) -> dict:
    """Run segmentation over `stems` and accumulate one confusion matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for stem in stems:
        img_path = str(Path(img_dir) / f"{stem}.jpg")
        mask_path = str(Path(mask_dir) / f"{stem}.png")
        image = cv2.imread(img_path)
        gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or gt is None:
            continue
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        pred = run(handle, image)
        cm += confusion(pred, gt)

    per_class_iou, miou, pixel_acc = iou_from_confusion(cm)
    return {
        "confusion": cm,
        "per_class_iou": per_class_iou,
        "miou": miou,
        "pixel_acc": pixel_acc,
        "num_images": len(stems),
        "num_classes": NUM_CLASSES,
    }


def run_for_report(model_path: str, img_dir: str, mask_dir: str, csv_path: str, stems: list[str], backend: str) -> dict:
    """Evaluate and shape the result the way report.py expects."""
    handle = load(model_path, backend)
    result = evaluate_dataset(handle, img_dir, mask_dir, stems)
    class_names = load_class_names(csv_path)
    per_class_iou_named = {
        name: (None if np.isnan(iou) else float(iou))
        for name, iou in zip(class_names, result["per_class_iou"])
    }
    return {
        "per_class_iou": per_class_iou_named,
        "miou": result["miou"],
        "pixel_acc": result["pixel_acc"],
        "num_images": result["num_images"],
        "num_classes": result["num_classes"],
    }


def main(args) -> int:
    img_dir = "data/icg/images"
    mask_dir = "data/icg/masks"
    csv_path = "data/icg/class_dict_seg.csv"

    stems = sorted(Path(p).stem for p in glob.glob(f"{mask_dir}/*.png"))
    if args.subset:
        stems = stems[: args.subset]

    class_names = load_class_names(csv_path)
    handle = load(args.model, args.backend)
    result = evaluate_dataset(handle, img_dir, mask_dir, stems)

    print(f"{'class':<14}{'IoU':>8}")
    for name, iou in zip(class_names, result["per_class_iou"]):
        iou_str = "n/a" if np.isnan(iou) else f"{iou:.4f}"
        print(f"{name:<14}{iou_str:>8}")
    print(f"\nmIoU={result['miou']:.4f} pixel_acc={result['pixel_acc']:.4f} "
          f"num_images={result['num_images']}")

    if args.json_path:
        import json

        out = {
            "per_class_iou": {
                name: (None if np.isnan(iou) else float(iou))
                for name, iou in zip(class_names, result["per_class_iou"])
            },
            "miou": result["miou"],
            "pixel_acc": result["pixel_acc"],
            "num_images": result["num_images"],
            "num_classes": result["num_classes"],
            "model": args.model,
            "backend": args.backend,
        }
        Path(args.json_path).write_text(json.dumps(out, indent=2))
    return 0
