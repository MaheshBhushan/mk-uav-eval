"""Detection inference over a YOLOv8 ONNX model via two interchangeable backends."""
import cv2
import numpy as np

BACKENDS = ("cv2", "ort", "ort-cpu", "ort-openvino", "ort-openvino-gpu")


def preprocess(image_bgr, size=640):
    """Letterbox to a square `size` blob. Returns (blob, scale, pad)."""
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad = ((size - new_w) / 2, (size - new_h) / 2)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, swapRB=True)
    return blob, scale, (left, top)


def load(model_path: str, backend: str, sess_options=None):
    """Return an opaque handle for `backend` (one of BACKENDS)."""
    if backend == "cv2":
        net = cv2.dnn.readNetFromONNX(model_path)
        return ("cv2", net)
    if backend in ("ort", "ort-cpu", "ort-openvino", "ort-openvino-gpu"):
        import onnxruntime as ort

        if backend in ("ort", "ort-cpu"):
            providers = ["CPUExecutionProvider"]
        elif backend == "ort-openvino":
            providers = [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"]
        else:
            providers = [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"]

        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        wanted = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if session.get_providers()[0] != wanted:
            # ORT silently drops providers that are not compiled in and falls back to CPU;
            # a benchmark row for a backend that did not actually run would be a lie.
            raise RuntimeError(f"{wanted} not available in this onnxruntime build "
                               f"(got {session.get_providers()})")
        return ("ort", session)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")


def _forward(handle, blob):
    backend, model = handle
    if backend == "cv2":
        model.setInput(blob)
        return model.forward()
    return model.run(None, {model.get_inputs()[0].name: blob})[0]


def run(handle, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    """Run detection, returning boxes in original-image pixel coords."""
    blob, scale, pad = preprocess(image_bgr)
    output = np.asarray(_forward(handle, blob))
    # [1, 84, 8400] -> [8400, 84]: 4 box coords (cx, cy, w, h) + 80 class scores.
    preds = output[0].T
    scores = preds[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    keep = confidences >= conf
    if not keep.any():
        return []
    preds, class_ids, confidences = preds[keep], class_ids[keep], confidences[keep]

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    boxes = np.stack([x1, y1, bw / scale, bh / scale], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.astype(np.float32).tolist(), conf, iou)
    if len(indices) == 0:
        return []

    h, w = image_bgr.shape[:2]
    results = []
    for i in np.asarray(indices).flatten():
        bx, by, bw_i, bh_i = boxes[i]
        results.append(
            {
                "bbox": [
                    float(max(0.0, min(bx, w))),
                    float(max(0.0, min(by, h))),
                    float(max(0.0, min(bx + bw_i, w))),
                    float(max(0.0, min(by + bh_i, h))),
                ],
                "score": float(confidences[i]),
                "class_id": int(class_ids[i]),
            }
        )
    return results


def main(args) -> int:
    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read image {args.image}")
        return 2
    handle = load(args.model, args.backend)
    for det in run(handle, image, conf=args.conf, iou=args.iou):
        x1, y1, x2, y2 = det["bbox"]
        print(f"{det['class_id']} {det['score']:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
    return 0
