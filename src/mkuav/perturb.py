"""Robustness perturbation suite: re-evaluate detection mAP under image perturbations.

Each perturbation is a pure function `(img_bgr, severity: int) -> img_bgr` with
severity 0 == identity (returns the image unchanged). Severities 1/2/3 are defined
per-perturbation below.

Perturbations:
    blur:            cv2.GaussianBlur, kernel size 3 / 7 / 13.
    brightness_down:  cv2.convertScaleAbs(img, alpha=1, beta=b), beta -30 / -60 / -90.
    brightness_up:    cv2.convertScaleAbs(img, alpha=1, beta=b), beta +30 / +60 / +90.
    jpeg:            re-encode/decode at JPEG quality 50 / 20 / 8.
    rotate:          warpAffine about image centre by 5 / 15 / 30 degrees, constant
                     border (114), canvas size kept fixed. GT/ignored-region boxes are
                     rotated with the same affine matrix via `transform_boxes`, taking
                     the enclosing axis-aligned box of the four rotated corners and
                     clipping to the image. This enclosing box is loose (it grows with
                     angle), so rotation mAP should be read as a lower bound on true
                     rotational robustness, not an exact measurement.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from mkuav import detect
from mkuav.metrics_det import evaluate, filter_ignored, load_visdrone_gt

BLUR_KSIZE = {1: 3, 2: 7, 3: 13}
BRIGHTNESS_DOWN_BETA = {1: -30, 2: -60, 3: -90}
BRIGHTNESS_UP_BETA = {1: 30, 2: 60, 3: 90}
JPEG_QUALITY = {1: 50, 2: 20, 3: 8}
ROTATE_DEGREES = {1: 5, 2: 15, 3: 30}


def blur(img_bgr, severity: int):
    if severity == 0:
        return img_bgr
    k = BLUR_KSIZE[severity]
    return cv2.GaussianBlur(img_bgr, (k, k), 0)


def brightness_down(img_bgr, severity: int):
    if severity == 0:
        return img_bgr
    return cv2.convertScaleAbs(img_bgr, alpha=1, beta=BRIGHTNESS_DOWN_BETA[severity])


def brightness_up(img_bgr, severity: int):
    if severity == 0:
        return img_bgr
    return cv2.convertScaleAbs(img_bgr, alpha=1, beta=BRIGHTNESS_UP_BETA[severity])


def jpeg(img_bgr, severity: int):
    if severity == 0:
        return img_bgr
    q = JPEG_QUALITY[severity]
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img_bgr
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rotation_matrix(img_bgr, degrees: float):
    """Affine matrix rotating `img_bgr` about its centre by `degrees` (canvas size kept)."""
    h, w = img_bgr.shape[:2]
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)


def rotate(img_bgr, severity: int):
    if severity == 0:
        return img_bgr
    degrees = ROTATE_DEGREES[severity]
    M = rotation_matrix(img_bgr, degrees)
    h, w = img_bgr.shape[:2]
    return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


PERTURBATIONS = {
    "blur": blur,
    "brightness_down": brightness_down,
    "brightness_up": brightness_up,
    "jpeg": jpeg,
    "rotate": rotate,
}


def transform_boxes(boxes_xyxy, M, w, h):
    """Apply affine matrix `M` to xyxy boxes, returning the enclosing axis-aligned
    box of each box's four rotated corners, clipped to (w, h).

    `boxes_xyxy` is an iterable of [x1, y1, x2, y2]. Returns a list of [x1, y1, x2, y2].
    """
    M = np.asarray(M, dtype=np.float64)
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        )
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated = corners_h @ M.T
        nx1, ny1 = rotated[:, 0].min(), rotated[:, 1].min()
        nx2, ny2 = rotated[:, 0].max(), rotated[:, 1].max()
        nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
        nx2, ny2 = min(float(w), nx2), min(float(h), ny2)
        out.append([nx1, ny1, nx2, ny2])
    return out


def run_robustness(
    model_path, backend, stems, gt_by_image, ignored_by_image, img_dir,
    perturbations, severities=(0, 1, 2, 3), conf=0.001,
):
    """Evaluate mAP for each (perturbation, severity) combination.

    Images are perturbed on the fly (never written to disk). For every perturbation
    other than "rotate", ground truth and ignored regions are unchanged (only the
    image content is perturbed, not geometry). For "rotate", GT and ignored-region
    boxes are transformed with the same affine matrix used on the image, via
    `transform_boxes`; the resulting box is a loose enclosing box (see module
    docstring), so rotation mAP is a lower bound.

    Returns {name: {str(severity): evaluate(...) dict}}.
    """
    from mkuav.metrics_det import NMS_IOU

    img_dir = Path(img_dir)
    handle = detect.load(model_path, backend)

    # Pre-load raw images once.
    raw_images = {}
    for stem in stems:
        img = cv2.imread(str(img_dir / f"{stem}.jpg"))
        raw_images[stem] = img

    results = {}
    for name in perturbations:
        fn = PERTURBATIONS[name]
        results[name] = {}
        for sev in severities:
            pred_by_image = {}
            gt_this = {}
            ignored_this = {}
            for stem in stems:
                img = raw_images[stem]
                if img is None:
                    pred_by_image[stem] = []
                    gt_this[stem] = gt_by_image.get(stem, [])
                    continue
                pimg = fn(img, sev)
                pred_by_image[stem] = detect.run(handle, pimg, conf=conf, iou=NMS_IOU)

                if name == "rotate" and sev != 0:
                    h, w = img.shape[:2]
                    M = rotation_matrix(img, ROTATE_DEGREES[sev])
                    gts = gt_by_image.get(stem, [])
                    boxes = [g["bbox"] for g in gts]
                    rboxes = transform_boxes(boxes, M, w, h)
                    gt_this[stem] = [
                        {**g, "bbox": rb} for g, rb in zip(gts, rboxes)
                    ]
                    regions = ignored_by_image.get(stem)
                    if regions:
                        region_boxes = [[rx, ry, rx + rw, ry + rh] for rx, ry, rw, rh in regions]
                        rregions = transform_boxes(region_boxes, M, w, h)
                        ignored_this[stem] = [
                            [rx1, ry1, rx2 - rx1, ry2 - ry1] for rx1, ry1, rx2, ry2 in rregions
                        ]
                else:
                    gt_this[stem] = gt_by_image.get(stem, [])
                    if stem in ignored_by_image:
                        ignored_this[stem] = ignored_by_image[stem]

            filtered_preds = filter_ignored(pred_by_image, ignored_this)
            results[name][str(sev)] = evaluate(filtered_preds, gt_this)
    return results


def main(args) -> int:
    ann_dir = "data/visdrone_val_clean/annotations"
    ignored_json = "data/visdrone_val_clean/ignored.json"
    img_dir = "data/visdrone_val/images"

    gt_by_image, ignored_by_image = load_visdrone_gt(ann_dir, ignored_json)
    stems = sorted(gt_by_image)
    if args.subset:
        stems = stems[: args.subset]
        gt_by_image = {s: gt_by_image[s] for s in stems}

    names = [n.strip() for n in args.perturbations.split(",")] if args.perturbations else list(PERTURBATIONS)
    for n in names:
        if n not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation {n!r}, expected one of {list(PERTURBATIONS)}")

    results = run_robustness(
        args.model, args.backend, stems, gt_by_image, ignored_by_image, img_dir,
        names, conf=args.conf,
    )

    baseline = results[names[0]]["0"] if names else None

    print(f"{'perturbation':<18}{'sev':>4}{'mAP@0.5':>10}{'mAP@0.5:0.95':>14}{'P@0.25':>8}{'R@0.25':>8}")
    print(f"{'baseline':<18}{0:>4}{baseline['map50']:>10.4f}{baseline['map50_95']:>14.4f}"
          f"{baseline['overall']['precision_op']:>8.3f}{baseline['overall']['recall_op']:>8.3f}")
    for name in names:
        for sev in (0, 1, 2, 3):
            r = results[name][str(sev)]
            print(
                f"{name:<18}{sev:>4}{r['map50']:>10.4f}{r['map50_95']:>14.4f}"
                f"{r['overall']['precision_op']:>8.3f}{r['overall']['recall_op']:>8.3f}"
            )

    params = {
        "blur": [BLUR_KSIZE[1], BLUR_KSIZE[2], BLUR_KSIZE[3]],
        "brightness_down": [BRIGHTNESS_DOWN_BETA[1], BRIGHTNESS_DOWN_BETA[2], BRIGHTNESS_DOWN_BETA[3]],
        "brightness_up": [BRIGHTNESS_UP_BETA[1], BRIGHTNESS_UP_BETA[2], BRIGHTNESS_UP_BETA[3]],
        "jpeg": [JPEG_QUALITY[1], JPEG_QUALITY[2], JPEG_QUALITY[3]],
        "rotate": [ROTATE_DEGREES[1], ROTATE_DEGREES[2], ROTATE_DEGREES[3]],
    }

    out = {
        "num_images": len(stems),
        "backend": args.backend,
        "baseline": baseline,
        "results": results,
        "params": {name: params[name] for name in names},
    }

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"wrote {args.json_path}")
    return 0
