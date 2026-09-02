"""Annotation QA for VisDrone-DET and ICG segmentation datasets."""
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

VISDRONE_IGNORED_CATEGORY = 0
VISDRONE_OTHERS_CATEGORY = 11
VISDRONE_VALID_CATEGORIES = set(range(1, 11))
TINY_BOX_AREA = 16
ICG_VALID_CLASS_COUNT = 24


def _image_size(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _parse_row(raw):
    fields = raw.strip().split(",")
    if len(fields) != 8:
        return None
    try:
        return [int(f) for f in fields]
    except ValueError:
        return None


def run_visdrone(ann_dir, img_dir):
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)
    counts = Counter()
    files_affected = {}
    class_hist = Counter()
    truncation_hist = Counter()
    occlusion_hist = Counter()

    def flag(rule, stem):
        counts[rule] += 1
        files_affected.setdefault(rule, set()).add(stem)

    ann_files = sorted(ann_dir.glob("*.txt"))
    for ann_path in ann_files:
        stem = ann_path.stem
        img_path = img_dir / f"{stem}.jpg"
        size = _image_size(img_path) if img_path.exists() else None
        lines = ann_path.read_text().splitlines()
        lines = [l for l in lines if l.strip()]
        if not lines:
            flag("empty_file", stem)
        seen_boxes = set()
        for raw in lines:
            row = _parse_row(raw)
            if row is None:
                flag("malformed_row", stem)
                continue
            left, top, w, h, score, category, truncation, occlusion = row
            if w <= 0 or h <= 0:
                flag("zero_area", stem)
            if size is not None:
                img_w, img_h = size
                if left < 0 or top < 0 or left + w > img_w or top + h > img_h:
                    flag("out_of_bounds", stem)
            if category == VISDRONE_IGNORED_CATEGORY:
                flag("ignored_region", stem)
            if category == VISDRONE_OTHERS_CATEGORY:
                flag("others_class", stem)
            if score == 0 and category in VISDRONE_VALID_CATEGORIES:
                flag("score_zero", stem)
            box_key = (left, top, w, h, category)
            if box_key in seen_boxes:
                flag("duplicate_box", stem)
            else:
                seen_boxes.add(box_key)
            if w > 0 and h > 0 and w * h < TINY_BOX_AREA:
                flag("tiny_box", stem)
            class_hist[category] += 1
            truncation_hist[truncation] += 1
            occlusion_hist[occlusion] += 1

    return {
        "counts": dict(counts),
        "files_affected": {k: sorted(v) for k, v in files_affected.items()},
        "class_hist": dict(class_hist),
        "truncation_hist": dict(truncation_hist),
        "occlusion_hist": dict(occlusion_hist),
        "num_files": len(ann_files),
    }


def run_icg(img_dir, mask_dir, csv_path):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    counts = Counter()
    files_affected = {}
    class_hist = Counter()

    def flag(rule, stem):
        counts[rule] += 1
        files_affected.setdefault(rule, set()).add(stem)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]
    num_classes = len(rows)

    img_stems = {p.stem for p in img_dir.glob("*.jpg")}
    mask_stems = {p.stem for p in mask_dir.glob("*.png")}

    for stem in sorted(img_stems - mask_stems):
        flag("mask_missing", stem)
    for stem in sorted(mask_stems - img_stems):
        flag("image_missing", stem)

    classes_seen = set()
    for stem in sorted(img_stems & mask_stems):
        img_path = img_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue
        if img.shape[:2] != mask.shape[:2]:
            flag("dim_mismatch", stem)
            continue
        uniq, cnts = np.unique(mask, return_counts=True)
        for val, cnt in zip(uniq, cnts):
            val = int(val)
            cnt = int(cnt)
            class_hist[val] += cnt
            if val >= num_classes:
                flag("invalid_class_id", stem)
            else:
                classes_seen.add(val)

    for cls_id in range(num_classes):
        if cls_id not in classes_seen:
            flag("unused_class", f"class_{cls_id}")

    return {
        "counts": dict(counts),
        "files_affected": {k: sorted(v) for k, v in files_affected.items()},
        "class_hist": dict(class_hist),
        "num_classes": num_classes,
    }


def clean_visdrone(ann_dir, img_dir, out_dir):
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_ann_dir = out_dir / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    ignored = {}

    for ann_path in sorted(ann_dir.glob("*.txt")):
        stem = ann_path.stem
        img_path = img_dir / f"{stem}.jpg"
        size = _image_size(img_path) if img_path.exists() else None
        lines = [l for l in ann_path.read_text().splitlines() if l.strip()]

        clean_rows = []
        seen_boxes = set()
        ignored_boxes = []

        for raw in lines:
            row = _parse_row(raw)
            if row is None:
                stats["dropped_malformed"] += 1
                continue
            left, top, w, h, score, category, truncation, occlusion = row

            if category == VISDRONE_IGNORED_CATEGORY:
                ignored_boxes.append([left, top, w, h])
                continue

            if category not in VISDRONE_VALID_CATEGORIES:
                stats["dropped_category"] += 1
                continue
            if score != 1:
                stats["dropped_score"] += 1
                continue
            if w <= 0 or h <= 0:
                stats["dropped_zero_area"] += 1
                continue

            if size is not None:
                img_w, img_h = size
                clipped_left = max(0, left)
                clipped_top = max(0, top)
                clipped_right = min(img_w, left + w)
                clipped_bottom = min(img_h, top + h)
                left, top = clipped_left, clipped_top
                w, h = clipped_right - clipped_left, clipped_bottom - clipped_top
                if w <= 0 or h <= 0:
                    stats["dropped_zero_area"] += 1
                    continue

            box_key = (left, top, w, h, category)
            if box_key in seen_boxes:
                stats["dropped_duplicate"] += 1
                continue
            seen_boxes.add(box_key)

            clean_rows.append([left, top, w, h, score, category, truncation, occlusion])
            stats["kept"] += 1

        with open(out_ann_dir / ann_path.name, "w") as f:
            for row in clean_rows:
                f.write(",".join(str(v) for v in row) + "\n")

        if ignored_boxes:
            ignored[stem] = ignored_boxes

    with open(out_dir / "ignored.json", "w") as f:
        json.dump(ignored, f, indent=2, sort_keys=True)

    images_link = out_dir / "images"
    symlink_ok = False
    if not images_link.exists():
        try:
            images_link.symlink_to(Path("..") / "visdrone_val" / "images")
            symlink_ok = True
        except OSError:
            symlink_ok = False
    else:
        symlink_ok = images_link.is_symlink()

    return {"stats": dict(stats), "images_symlink": symlink_ok}


def _print_table(title, counts, num_files):
    print(f"\n{title}")
    print(f"{'rule':<20}{'count':>10}{'files_affected':>16}")
    for rule in sorted(counts.get("counts", {})):
        count = counts["counts"][rule]
        affected = len(counts.get("files_affected", {}).get(rule, []))
        print(f"{rule:<20}{count:>10}{affected:>16}")
    print(f"(files scanned: {num_files})")


def main(args) -> int:
    version = args.version
    visdrone_dir = "data/visdrone_val" if version == "v1" else "data/visdrone_val_clean"
    report = {}

    if args.clean:
        result = clean_visdrone(
            "data/visdrone_val/annotations",
            "data/visdrone_val/images",
            "data/visdrone_val_clean",
        )
        print("clean_visdrone:", json.dumps(result, indent=2))
        return 0

    if args.dataset in ("visdrone", "all"):
        ann_dir = f"{visdrone_dir}/annotations"
        img_dir = "data/visdrone_val/images"
        visdrone_result = run_visdrone(ann_dir, img_dir)
        _print_table(f"VisDrone ({version})", visdrone_result, visdrone_result["num_files"])
        report["visdrone"] = visdrone_result

    if args.dataset in ("icg", "all"):
        icg_result = run_icg("data/icg/images", "data/icg/masks", "data/icg/class_dict_seg.csv")
        _print_table("ICG", icg_result, len(icg_result.get("class_hist", {})))
        top5 = sorted(icg_result["class_hist"].items(), key=lambda kv: -kv[1])[:5]
        print("top-5 class pixel histogram:", top5)
        report["icg"] = icg_result

    json_path = args.json_path or f"qa_report_{version}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nwrote {json_path}")
    return 0
