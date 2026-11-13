"""Kaggle GPU script kernel: fine-tune COCO YOLOv8n on VisDrone2019-DET train, export ONNX.

Pushed as kodurimahesh/mk-uav-det-finetune with `kaggle kernels push -p notebooks/det_finetune`.
Input: the same Kaggle mirror scripts/fetch.py uses (hassanmojab/visdrone-det, original
8-column annotations). Only the train split is used. The VisDrone val split that the harness
scores is never read here; Ultralytics' in-training validation runs on a held-out slice of
train sequences instead, and the exported model is last.pt, not a checkpoint picked on val.

Outputs in /kaggle/working:
  yolov8n_visdrone.onnx   opset 12, static 1x3x640x640 -> 1x14x8400 (4 box + 10 VisDrone classes)
  last.pt                 final PyTorch checkpoint
  results.csv, args.yaml  Ultralytics per-epoch log and effective training arguments
  data.yaml               dataset definition used for training
  det_train_log.json      GPU name, split sizes, epochs, wall time
"""
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ULTRALYTICS = "ultralytics==8.4.210"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS, "onnx", "onnxslim"], check=True)

import torch  # noqa: E402
from ultralytics import YOLO, __version__ as ul_version  # noqa: E402

SEED = 0
EPOCHS = 50
IMGSZ = 640
BATCH = 32
HOLDOUT_EVERY = 20  # every 20th train sequence becomes the in-training val split

# Ultralytics VisDrone.yaml order: VisDrone category id - 1. Categories 0 (ignored region)
# and 11 (others) are not trained on, matching the harness, which never scores them.
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
         "awning-tricycle", "bus", "motor"]

WORK = Path("/kaggle/working")
DATA = Path("/tmp/visdrone_yolo")
RUNS = Path("/tmp/runs")


def find_train_root() -> Path:
    for p in sorted(Path("/kaggle/input").rglob("VisDrone2019-DET-train")):
        if (p / "annotations").is_dir() and (p / "images").is_dir():
            return p
    raise SystemExit("VisDrone2019-DET-train with images/ and annotations/ not found under /kaggle/input")


def convert(train_root: Path) -> dict:
    """Write YOLO labels for every train image and split by sequence id."""
    import cv2

    stems = sorted(p.stem for p in (train_root / "images").glob("*.jpg"))
    sequences = sorted({s.split("_")[0] for s in stems})
    held = set(sequences[::HOLDOUT_EVERY])
    counts = {"train": 0, "val": 0, "boxes": 0, "skipped_rows": 0}
    for split in ("train", "val"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        split = "val" if stem.split("_")[0] in held else "train"
        img_path = train_root / "images" / f"{stem}.jpg"
        h, w = cv2.imread(str(img_path)).shape[:2]
        lines = []
        for raw in (train_root / "annotations" / f"{stem}.txt").read_text().splitlines():
            f = raw.strip().split(",")
            if len(f) < 6:
                continue
            left, top, bw, bh, score, cat = (int(x) for x in f[:6])
            if score == 0 or cat in (0, 11) or bw <= 0 or bh <= 0:
                counts["skipped_rows"] += 1
                continue
            cx, cy = (left + bw / 2) / w, (top + bh / 2) / h
            lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (DATA / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        os.symlink(img_path, DATA / "images" / split / f"{stem}.jpg")
        counts[split] += 1
        counts["boxes"] += len(lines)
    counts["held_out_sequences"] = sorted(held)
    return counts


def main() -> None:
    random.seed(SEED)
    assert torch.cuda.is_available(), "kernel must run with a GPU accelerator"
    gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"GPUs: {gpu}  torch {torch.__version__}  ultralytics {ul_version}", flush=True)

    train_root = find_train_root()
    print(f"train root: {train_root}", flush=True)
    t0 = time.time()
    counts = convert(train_root)
    print(f"converted: {counts['train']} train / {counts['val']} held-out images, "
          f"{counts['boxes']} boxes in {time.time() - t0:.0f}s", flush=True)

    data_yaml = DATA / "data.yaml"
    data_yaml.write_text(
        f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )

    model = YOLO("yolov8n.pt")  # COCO weights, the same starting point as the harness baseline
    t1 = time.time()
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=4,
        cache="ram", seed=SEED, deterministic=False, project=str(RUNS), name="visdrone",
        exist_ok=True, plots=False, verbose=True,
    )
    train_s = time.time() - t1
    run_dir = RUNS / "visdrone"

    last = YOLO(str(run_dir / "weights" / "last.pt"))
    onnx_path = Path(last.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False))

    shutil.copy2(onnx_path, WORK / "yolov8n_visdrone.onnx")
    shutil.copy2(run_dir / "weights" / "last.pt", WORK / "last.pt")
    for name in ("results.csv", "args.yaml"):
        shutil.copy2(run_dir / name, WORK / name)
    shutil.copy2(data_yaml, WORK / "data.yaml")

    log = {
        "gpus": gpu,
        "torch": torch.__version__,
        "ultralytics": ul_version,
        "python": platform.python_version(),
        "start_weights": "yolov8n.pt (COCO)",
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "exported_checkpoint": "last.pt",
        "onnx_opset": 12,
        "train_images": counts["train"],
        "holdout_images": counts["val"],
        "holdout_sequences": counts["held_out_sequences"],
        "train_boxes_incl_holdout": counts["boxes"],
        "train_wall_s": round(train_s, 1),
        "class_names": NAMES,
    }
    (WORK / "det_train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2), flush=True)


if __name__ == "__main__":
    main()
