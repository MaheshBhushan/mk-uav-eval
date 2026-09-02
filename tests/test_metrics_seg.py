"""Segmentation metric tests, including a cross-check against torchmetrics."""
from pathlib import Path

import numpy as np
import pytest

from mkuav import metrics_seg as ms

RANDOM_MODEL = "/var/tmp/seg_random.onnx"
ICG_IMAGE = "data/icg/images/000.jpg"


def test_synthetic_confusion_and_iou():
    """Two 8x8 masks with known overlap; per-class IoU and mIoU by hand.

    gt is all class 0 in the left half (4x8=32 px) and class 1 in the right half
    (32 px). pred matches gt exactly except the top row of the right half (4 px)
    is predicted as class 0 instead of class 1.
    class 0: TP=32+4=36, gt=32, pred=36 -> union=32+36-36=32 -> IoU=36/32? no:
    Let's just compute directly instead of asserting a pre-derived formula.
    """
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[:, 4:] = 1
    pred = gt.copy()
    pred[0, 4:] = 0  # 4 pixels of class 1 mispredicted as class 0

    cm = ms.confusion(pred, gt, num_classes=2, ignore=255)
    # rows=gt, cols=pred
    assert cm[0, 0] == 32  # all gt-class-0 pixels correctly predicted 0
    assert cm[1, 0] == 4  # gt-class-1 pixels wrongly predicted 0
    assert cm[1, 1] == 28  # gt-class-1 pixels correctly predicted 1

    per_class_iou, miou, pixel_acc = ms.iou_from_confusion(cm)
    # class 0: tp=32, gt_total=32, pred_total=36, union=32+36-32=36 -> iou=32/36
    assert per_class_iou[0] == pytest.approx(32 / 36)
    # class 1: tp=28, gt_total=32, pred_total=28, union=32+28-28=32 -> iou=28/32
    assert per_class_iou[1] == pytest.approx(28 / 32)
    assert miou == pytest.approx((32 / 36 + 28 / 32) / 2)
    assert pixel_acc == pytest.approx(60 / 64)


def test_iou_from_confusion_absent_class_is_nan():
    """A class absent from both gt and pred contributes nan, excluded from mIoU."""
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 1] = 10
    # class 2 never appears in gt or pred.
    per_class_iou, miou, _ = ms.iou_from_confusion(cm)
    assert np.isnan(per_class_iou[2])
    assert miou == pytest.approx(1.0)


def test_cross_check_against_torchmetrics():
    """Our mIoU/pixel-acc must match torchmetrics on random data with ignore pixels.

    Convention notes (matched deliberately):
    - torchmetrics' `ignore_index` masks pixels by the *target* (gt) value, so the
      ignore pixels here are injected into `gt`, and `pred` is left with valid
      class ids everywhere at those positions (our `confusion` already treats
      pred==255 as invalid too, but that path isn't exercised by this test).
    - torchmetrics `average="macro"` for JaccardIndex nanmeans only over classes
      that are present in gt or pred (absent classes don't count against the
      average) -- our `iou_from_confusion` uses the same "present" mask.
    """
    torch = pytest.importorskip("torch")
    torchmetrics = pytest.importorskip("torchmetrics")

    rng = np.random.default_rng(0)
    num_classes = 24
    gts, preds = [], []
    for _ in range(5):
        gt = rng.integers(0, num_classes, size=(16, 16)).astype(np.int64)
        pred = gt.copy()
        # ~70% agreement: flip ~30% of pixels to a random other class.
        flip = rng.random(gt.shape) < 0.30
        pred[flip] = rng.integers(0, num_classes, size=flip.sum())
        # sprinkle some ignore pixels into gt (torchmetrics ignore_index targets gt).
        ignore_mask = rng.random(gt.shape) < 0.05
        gt[ignore_mask] = 255
        gts.append(gt)
        preds.append(pred)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(gts, preds):
        cm += ms.confusion(pred, gt, num_classes=num_classes, ignore=255)
    _, our_miou, our_acc = ms.iou_from_confusion(cm)

    gt_t = torch.from_numpy(np.stack(gts))
    pred_t = torch.from_numpy(np.stack(preds))
    jaccard = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=255, average="macro",
    )
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=255, average="micro",
    )
    tm_miou = jaccard(pred_t, gt_t).item()
    tm_acc = accuracy(pred_t, gt_t).item()

    print(f"ours miou={our_miou:.6f} acc={our_acc:.6f} | torchmetrics miou={tm_miou:.6f} acc={tm_acc:.6f}")
    assert abs(our_miou - tm_miou) < 1e-5
    assert abs(our_acc - tm_acc) < 1e-5


def test_backend_parity_cv2_vs_ort():
    """cv2 and ort argmax maps must agree on >99% of pixels for a real image."""
    if not Path(RANDOM_MODEL).exists():
        pytest.skip(f"{RANDOM_MODEL} not present; run /var/tmp/make_seg_random.py first")
    if not Path(ICG_IMAGE).exists():
        pytest.skip(f"{ICG_IMAGE} not present")

    import cv2

    from mkuav import segment as seg

    image = cv2.imread(ICG_IMAGE)
    assert image is not None

    handle_cv2 = seg.load(RANDOM_MODEL, "cv2")
    handle_ort = seg.load(RANDOM_MODEL, "ort")

    mask_cv2 = seg.run(handle_cv2, image)
    mask_ort = seg.run(handle_ort, image)

    agree = float((mask_cv2 == mask_ort).mean())
    print(f"cv2 vs ort pixel agreement: {agree:.4f}")
    assert agree > 0.99
