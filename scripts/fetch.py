"""Fetch VisDrone2019-DET val split and Semantic Drone Dataset from Kaggle.

Downloads two Kaggle datasets, extracts only the slices this project needs,
and lays them out under data/. Idempotent: skips a download/extract step if
its target directory is already populated.

Usage: uv run python scripts/fetch.py
"""
import csv
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VISDRONE_DATASET = "banuprasadb/visdrone-dataset"
ICG_DATASET = "bulentsiyah/semantic-drone-dataset"

VISDRONE_VAL_DIR = DATA / "visdrone_val"
ICG_DIR = DATA / "icg"

ICG_LONG_EDGE = 1500


def _populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _download_and_unzip(dataset: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        check=True,
    )


def fetch_visdrone_val() -> None:
    images_dir = VISDRONE_VAL_DIR / "images"
    ann_dir = VISDRONE_VAL_DIR / "annotations"
    if _populated(images_dir) and _populated(ann_dir):
        print(f"visdrone_val already populated, skipping ({len(list(images_dir.iterdir()))} images)")
        return

    raw = ROOT / "_raw" / "visdrone"
    if not _populated(raw):
        _download_and_unzip(VISDRONE_DATASET, raw)

    # The Kaggle mirror ships train/val/test-dev/test-challenge splits under
    # VisDrone_Dataset/VisDrone2019-DET-<split>/{images,labels}. We only want val.
    val_root = next(raw.rglob("VisDrone2019-DET-val"))
    val_images = val_root / "images"
    val_labels = val_root / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for f in val_images.iterdir():
        shutil.copy2(f, images_dir / f.name)
    for f in val_labels.iterdir():
        shutil.copy2(f, ann_dir / f.name)

    # Free disk: drop the other splits and the raw download.
    shutil.rmtree(raw, ignore_errors=True)
    print(f"visdrone_val: {len(list(images_dir.iterdir()))} images, {len(list(ann_dir.iterdir()))} annotations")


def fetch_icg() -> None:
    images_dir = ICG_DIR / "images"
    masks_dir = ICG_DIR / "masks"
    if _populated(images_dir) and _populated(masks_dir):
        print(f"icg already populated, skipping ({len(list(images_dir.iterdir()))} images)")
        return

    raw = ROOT / "_raw" / "icg"
    if not _populated(raw):
        _download_and_unzip(ICG_DATASET, raw)

    src_images = next(raw.rglob("original_images"))
    src_masks = next(raw.rglob("label_images_semantic"))
    src_class_dict = next(raw.rglob("class_dict_seg.csv"))

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(src_images.glob("*.jpg")):
        mask_path = src_masks / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"warning: no mask for {img_path.name}, skipping", file=sys.stderr)
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        h, w = img.shape[:2]
        scale = ICG_LONG_EDGE / max(h, w)
        new_size = (round(w * scale), round(h * scale))

        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(images_dir / img_path.name), img_resized)
        cv2.imwrite(str(masks_dir / mask_path.name), mask_resized)

    shutil.copy2(src_class_dict, ICG_DIR / "class_dict_seg.csv")
    shutil.rmtree(raw, ignore_errors=True)
    print(f"icg: {len(list(images_dir.iterdir()))} images, {len(list(masks_dir.iterdir()))} masks")


def main() -> int:
    fetch_visdrone_val()
    fetch_icg()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
