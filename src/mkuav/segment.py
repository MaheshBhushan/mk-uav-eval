"""Segmentation inference over a plain-conv UNet ONNX model, via cv2.dnn or ort.

Preprocessing mirrors the val path of notebooks/seg_finetune.py exactly: resize to
768x512, centre-crop 512x512 (x0=(768-512)//2, y0=0), BGR->RGB, /255, ImageNet
mean/std normalisation, NCHW float32.

`run` maps the 512x512 argmax prediction back to the original image resolution by
placing the crop into a 768x512 canvas (filling the rest with 255, an "unpredicted"/
ignore sentinel value that is not a valid class index — the model only ever saw
pixels inside the crop) and resizing that canvas to the original H x W with
INTER_NEAREST (no interpolation across class boundaries).
"""
import cv2
import numpy as np

from mkuav.detect import load  # noqa: F401  (re-exported: model-agnostic)

RESIZE_W, RESIZE_H = 768, 512
CROP = 512
NUM_CLASSES = 24
IGNORE = 255

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray):
    """Resize to 768x512, centre-crop 512x512, normalise. Returns (blob, meta)."""
    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    x0 = (RESIZE_W - CROP) // 2
    y0 = (RESIZE_H - CROP) // 2
    crop = resized[y0:y0 + CROP, x0:x0 + CROP]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis].astype(np.float32)

    meta = {"orig_h": h, "orig_w": w, "x0": x0, "y0": y0}
    return blob, meta


def _forward(handle, blob):
    backend, model = handle
    if backend == "cv2":
        model.setInput(blob)
        return model.forward()
    return model.run(None, {model.get_inputs()[0].name: blob})[0]


def run(handle, image_bgr: np.ndarray) -> np.ndarray:
    """Run segmentation, returning a uint8 HxW class map at the original resolution.

    Pixels outside the 512x512 crop (once placed back into the 768x512 canvas) are
    set to IGNORE (255): the model never saw them, so there is no prediction to give.
    """
    blob, meta = preprocess(image_bgr)
    logits = np.asarray(_forward(handle, blob))  # (1, 24, 512, 512)
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)  # (512, 512)

    canvas = np.full((RESIZE_H, RESIZE_W), IGNORE, dtype=np.uint8)
    x0, y0 = meta["x0"], meta["y0"]
    canvas[y0:y0 + CROP, x0:x0 + CROP] = pred

    out = cv2.resize(
        canvas, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST,
    )
    return out


def main(args) -> int:
    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read image {args.image}")
        return 2
    handle = load(args.model, args.backend)
    mask = run(handle, image)
    cv2.imwrite(args.out, mask)
    print(f"wrote {args.out} ({mask.shape[1]}x{mask.shape[0]})")
    return 0
