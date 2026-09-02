"""Detection metric tests, including a cross-check against pycocotools."""
import numpy as np
import pytest

from mkuav import metrics_det as md

IMG_DIR = "data/visdrone_val/images"
ANN_DIR = "data/visdrone_val_clean/annotations"
IGNORED_JSON = "data/visdrone_val_clean/ignored.json"
MODEL = "models/yolov8n.onnx"


def _box(x, y, w=10, h=10):
    return [float(x), float(y), float(x + w), float(y + h)]


def test_iou_matrix_basic():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    got = md.iou_matrix(a, b)[0]
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(50 / 150)
    assert got[2] == pytest.approx(0.0)


def test_synthetic_ap50():
    """3 images, one class. 4 GT, 4 preds of which 3 are true positives.

    Score order: 0.9 TP, 0.8 FP, 0.7 TP, 0.6 TP -> precision 1, .5, .667, .75
    recall .25, .25, .5, .75. Envelope from the right: 1, .75, .75, .75.
    26 recall thresholds (0.00-0.25) take precision 1, the next 50 (0.26-0.75)
    take 0.75, the rest 0 -> AP = (26 + 50 * 0.75) / 101.
    """
    gt = {
        "a": [{"bbox": _box(0, 0), "class_name": "car"}, {"bbox": _box(100, 100), "class_name": "car"}],
        "b": [{"bbox": _box(0, 0), "class_name": "car"}],
        "c": [{"bbox": _box(50, 50), "class_name": "car"}],
    }
    preds = {
        "a": [{"bbox": _box(0, 0), "score": 0.9, "class_id": 2},
              {"bbox": _box(300, 300), "score": 0.8, "class_id": 2}],
        "b": [{"bbox": _box(1, 1), "score": 0.7, "class_id": 2}],
        "c": [{"bbox": _box(50, 51), "score": 0.6, "class_id": 2}],
    }
    res = md.evaluate(preds, gt, iou_thrs=[0.5], op_conf=0.5)
    expected = (26 + 50 * 0.75) / 101
    assert res["per_class"]["car"]["ap50"] == pytest.approx(expected, abs=1e-9)
    assert res["map50"] == pytest.approx(expected, abs=1e-9)
    assert res["per_class"]["car"]["n_gt"] == 4
    assert res["overall"]["precision_op"] == pytest.approx(0.75)
    assert res["overall"]["recall_op"] == pytest.approx(0.75)


def test_average_precision_edges():
    assert np.isnan(md.average_precision(np.zeros(0), np.zeros(0), 0))
    assert md.average_precision(np.zeros(0), np.zeros(0), 5) == 0.0
    # single perfect detection covering the only GT -> AP = fraction of rec thrs <= 1
    assert md.average_precision(np.ones(1), np.ones(1), 1) == pytest.approx(1.0)


def test_load_visdrone_gt_v1_drops_extra_categories():
    gt, ignored = md.load_visdrone_gt("data/visdrone_val/annotations", None)
    cats = {g["category"] for rows in gt.values() for g in rows}
    assert cats <= set(md.VISDRONE_TO_CLASS)
    assert 0 not in cats and 11 not in cats
    assert ignored  # v1 supplies its own category-0 regions


def test_filter_ignored_drops_by_centre():
    preds = {"a": [{"bbox": [0, 0, 10, 10], "score": 1.0, "class_id": 2},
                   {"bbox": [100, 100, 110, 110], "score": 1.0, "class_id": 2}]}
    out = md.filter_ignored(preds, {"a": [[0, 0, 20, 20]]})
    assert len(out["a"]) == 1
    assert out["a"][0]["bbox"][0] == 100


@pytest.mark.slow
def test_cross_check_pycocotools(tmp_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_by_image, ignored_by_image = md.load_visdrone_gt(ANN_DIR, IGNORED_JSON)
    stems = sorted(gt_by_image)[:50]
    gt_by_image = {s: gt_by_image[s] for s in stems}
    preds = md.predict_dataset(
        MODEL, "ort", IMG_DIR, stems, conf=0.001, iou=md.NMS_IOU,
        cache_path=str(tmp_path / "preds.json"),
    )
    # same filtered predictions feed both sides
    preds = md.filter_ignored(preds, ignored_by_image)
    preds = {s: md._topk([p for p in preds[s] if md._pred_class(p) is not None]) for s in stems}

    iou_thrs = md.DEFAULT_IOU_THRS
    ours = md.evaluate(preds, gt_by_image, iou_thrs=iou_thrs)

    cat_ids = {name: i + 1 for i, name in enumerate(md.CLASSES)}
    images, annotations, results = [], [], []
    ann_id = 1
    for img_id, stem in enumerate(stems, start=1):
        images.append({"id": img_id, "file_name": f"{stem}.jpg", "width": 10000, "height": 10000})
        for g in gt_by_image[stem]:
            x1, y1, x2, y2 = g["bbox"]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_ids[g["class_name"]],
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
            ann_id += 1
        for p in preds[stem]:
            x1, y1, x2, y2 = p["bbox"]
            results.append({
                "image_id": img_id, "category_id": cat_ids[md._pred_class(p)],
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": p["score"],
            })

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {}, "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = np.asarray(iou_thrs, dtype=np.float64)
    ev.params.areaRng = [[0.0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [md.MAX_DETS]
    ev.evaluate()
    ev.accumulate()

    prec = ev.eval["precision"]  # [T, R, K, A, M]
    valid = prec > -1
    coco_map = float(prec[valid].mean())
    p50 = prec[0]
    coco_map50 = float(p50[p50 > -1].mean())

    print(f"\nours  mAP@0.5={ours['map50']:.6f}  mAP@0.5:0.95={ours['map50_95']:.6f}")
    print(f"coco  mAP@0.5={coco_map50:.6f}  mAP@0.5:0.95={coco_map:.6f}")
    print(f"delta mAP@0.5={abs(ours['map50'] - coco_map50):.2e}  "
          f"mAP@0.5:0.95={abs(ours['map50_95'] - coco_map):.2e}")
    assert abs(ours["map50"] - coco_map50) < 1e-3
    assert abs(ours["map50_95"] - coco_map) < 1e-3
