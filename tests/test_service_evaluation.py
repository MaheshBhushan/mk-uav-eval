"""Real cv2/ORT evaluations with tiny synthetic ONNX models; no downloads."""
import asyncio
import base64
import json
import zlib

import cv2
import numpy as np
import pytest

from mkuav.service import BackendJobError, EvaluationJob, EvaluationService

# ONNX opset 13, IR 8: GlobalAveragePool -> zero-weight 1x1 Conv -> Reshape
# (detection) or Identity (segmentation). Detection emits a full-image car box;
# segmentation emits class zero. These fixtures exercise both real backends.
MODELS = {
    "detection": "eJzj4LC6xcOlxMWamZuYnirEVpCfn5OaoiTonpOflJjjWJZaBBQOAApyWXJBJbnYylMz0zNKuFiSMhOLhViS8/PKlFicgaSWKBdPdmpRXmpOfHFGYkGqA6MD4wJGdi45LrAiLlawqBBbfmlJQWmJEntQKkSAMyW1JDW5JDM/T6uVgyOEg5mDkYNRgNEJapHXB3aGUTAKRsGIAVp3mThCgCUAuIjxusDEwLDAGYIVXCAYAtLS0uwH0p1DFWipcjALsDtBCmQvCUaocAiUhvGjFGEVgwSXGAejkBAXEwcjEDNzMXM0sIKJJBkuaHkOlBUAquGBqgkB0U4sXAwCvACfUUXm",
    "segmentation": "eJzj4LD6xMKlxMWamZuYnirEVpCfn5OaoiTonpOflJjjWJZaBBQOAApyWXJBJbnYylMz0zNKuFiSMhOLhViS8/PKlFicgaSWKBdPdmpRXmpOfHFGYkGqA6MD4wJGdi4JLrAiIbb80pKC0hIlDs+U1LySzJJKIZ7i1PRcIDuxJDM/T2srE4cEBzMHIwejAKMT1BavBUwMowAv0MrhkAAGGDg6vBJobVuUIiyxSHCJcTAKCXExcTACMTMXM0cDC5hIUuCCRrWQGJcIUI0AVI0ElAY6lotBgBcA2Jc8hA==",
}


@pytest.fixture
def dataset(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name in ("visdrone_val/annotations", "visdrone_val/images",
                 "visdrone_val_clean/annotations", "icg/images", "icg/masks"):
        (tmp_path / name).mkdir(parents=True)
    for stem in ("a", "b"):
        (tmp_path / "visdrone_val/annotations" / f"{stem}.txt").write_text("0,0,8,8,1,4,0,0\n")
        cv2.imwrite(str(tmp_path / "visdrone_val/images" / f"{stem}.jpg"), image)
    (tmp_path / "visdrone_val_clean/annotations/b.txt").write_text("0,0,8,8,1,4,0,0\n")
    (tmp_path / "visdrone_val_clean/ignored.json").write_text("{}")
    for i in range(45):
        stem = f"{i:03}"
        cv2.imwrite(str(tmp_path / "icg/images" / f"{stem}.jpg"), image)
        cv2.imwrite(str(tmp_path / "icg/masks" / f"{stem}.png"), image[:, :, 0])
    (tmp_path / "icg/class_dict_seg.csv").write_text("name\n" + "\n".join(f"class{i}" for i in range(24)))
    for task, data in MODELS.items():
        (tmp_path / f"{task}.onnx").write_bytes(zlib.decompress(base64.b64decode(data)))
    return tmp_path


@pytest.mark.parametrize("backend", ["cv2", "ort"])
def test_real_backend_metrics_splits_and_timings(dataset, backend):
    if backend == "ort":
        pytest.importorskip("onnxruntime")

    async def scenario():
        async with EvaluationService(2, 2) as api:
            jobs = [
                EvaluationJob("detection", str(dataset / "detection.onnx"), backend,
                              str(dataset), version=version, subset=1)
                for version in ("v1", "v2")
            ]
            jobs.append(EvaluationJob("segmentation", str(dataset / "segmentation.onnx"),
                                      backend, str(dataset), subset=2))
            raw, clean, seg = await asyncio.gather(*(api.evaluate(j, timeout=5) for j in jobs))
        assert raw["dataset"]["stems"] == ["a"]
        assert clean["dataset"]["stems"] == ["b"]
        assert raw["dataset"]["split"] == "visdrone-val-v1"
        assert clean["dataset"]["split"] == "visdrone-val-v2"
        assert raw["metrics"]["per_class"]["car"]["ap50"] == pytest.approx(1)
        assert clean["metrics"]["per_class"]["car"]["ap50"] == pytest.approx(1)
        assert seg["dataset"]["stems"] == ["005", "006"]
        assert seg["dataset"]["split"] == "icg-holdout"
        assert seg["metrics"]["miou"] == pytest.approx(1)
        assert seg["metrics"]["num_images"] == 2
        assert seg["metrics"]["per_class_iou"]["class1"] is None
        for result in (raw, clean, seg):
            latency = result["inference"]
            assert latency["count"] == len(result["dataset"]["stems"])
            assert latency["total_ms"] == pytest.approx(sum(latency["samples_ms"]))
            assert 0 < latency["p50_ms"] <= latency["p95_ms"]
            assert result["execution_ms"] >= latency["total_ms"]
            assert result["queue_ms"] >= 0
            json.dumps(result, allow_nan=False)
    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_ignored_regions_and_missing_image_failures(dataset):
    async def scenario():
        request = EvaluationJob("detection", str(dataset / "detection.onnx"), "cv2", str(dataset))
        async with EvaluationService(1, 0) as api:
            (dataset / "visdrone_val_clean/ignored.json").write_text('{"b": [[0,0,8,8]]}')
            result = await api.evaluate(request)
            assert result["metrics"]["overall"]["n_pred_op"] == 0
            # Corrupt the fixture without deleting any data.
            (dataset / "visdrone_val/images/b.jpg").write_bytes(b"invalid image")
            with pytest.raises(BackendJobError, match="cannot read image"):
                await api.evaluate(request)
            with pytest.raises(BackendJobError):
                await api.evaluate(EvaluationJob("detection", "missing.onnx", "cv2", str(dataset)))
            (dataset / "icg/masks/005.png").write_bytes(b"invalid mask")
            with pytest.raises(BackendJobError, match="cannot read image/mask pair"):
                await api.evaluate(EvaluationJob(
                    "segmentation", str(dataset / "segmentation.onnx"), "cv2", str(dataset),
                ))
    asyncio.run(asyncio.wait_for(scenario(), 10))
