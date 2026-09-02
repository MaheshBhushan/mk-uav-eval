# mk-uav-eval

[![ci](https://github.com/MaheshBhushan/mk-uav-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/MaheshBhushan/mk-uav-eval/actions/workflows/ci.yml)

An OpenCV evaluation harness for versioned UAV datasets. It validates annotations,
runs ONNX models through OpenCV DNN and onnxruntime, scores them against
pycocotools-verified metrics, measures latency and robustness, and publishes a
Markdown + JSON report from CI on every push.

Every number below was produced by `mkuav report` on the current `main`.

## What it does

| Stage | Command | Output |
|---|---|---|
| Annotation QA | `mkuav qa --version v1\|v2` | rule counts per dataset, `qa_report_*.json` |
| Dataset versioning | DVC, git tags `data-v1` (raw) and `data-v2` (QA-cleaned) | `data/*.dvc` |
| Detection | `mkuav detect`, `mkuav eval-det` | boxes; per-class AP, mAP@0.5, mAP@0.5:0.95, P/R |
| Latency | `mkuav bench` | p50 / p95 per backend |
| Robustness | `mkuav robust` | mAP vs. blur, brightness, JPEG, rotation severity |
| Report | `mkuav report` | `report.md`, `metrics.json`, per-stage JSON |

Backends: `cv2` (OpenCV DNN), `ort-cpu` (onnxruntime), `ort-openvino`
(onnxruntime OpenVINO execution provider). All three share one preprocessing
and one `cv2.dnn.NMSBoxes` post-processing path, so the forward pass is the only
variable. If onnxruntime silently drops a requested provider, the loader raises
rather than reporting a CPU number under the wrong name.

## Data

| Dataset | Split used | Task | Source |
|---|---|---|---|
| VisDrone2019-DET | val, 548 images | detection | Kaggle mirror `hassanmojab/visdrone-det` (original 8-column annotations) |
| ICG Semantic Drone | all 400 images, resized to 1500 px | segmentation | Kaggle mirror `bulentsiyah/semantic-drone-dataset` |

`scripts/fetch.py` pulls both with the Kaggle CLI and lays them out under `data/`.
Full data is DVC-tracked against a local remote. A 50-image VisDrone and
20-image ICG subset is committed in `data/ci_subset/` so CI needs no remote.

Two dataset versions are tagged. `data-v1` is the raw download. `data-v2` drops
ignored-region and "others" rows, clips boxes to the image and de-duplicates.
Ignored regions are kept in `ignored.json` and used at evaluation time to
suppress predictions whose centre falls inside them.

## Results

### Annotation QA, VisDrone val (v1)

| Rule | Boxes | Files |
|---|---|---|
| ignored_region | 1378 | 340 |
| others_class | 32 | 29 |
| tiny_box (< 16 px², informational) | 103 | 45 |

Zero-area, out-of-bounds, duplicate, malformed and score-zero rows: none in this
split. ICG masks: 0 invalid class ids, 0 dimension mismatches, 23 of 24 classes present.
Because the only cleaning that mattered was dropping category 0 and 11 rows the
loader already ignores, v1 and v2 score identically. The report says so rather
than implying the cleaning moved the number.

### Detection, YOLOv8n COCO weights on VisDrone val v2, 548 images

COCO classes are mapped onto six merged VisDrone classes (person, bicycle, car,
motor, bus, truck). Tricycle classes have no COCO equivalent and are excluded.

| Class | GT boxes | AP@0.5 | AP@0.5:0.95 | P@0.25 | R@0.25 |
|---|---|---|---|---|---|
| car | 16039 | 0.396 | 0.231 | 0.801 | 0.299 |
| person | 13969 | 0.148 | 0.057 | 0.769 | 0.076 |
| bus | 251 | 0.106 | 0.080 | 0.225 | 0.179 |
| truck | 750 | 0.037 | 0.026 | 0.247 | 0.049 |
| motor | 4886 | 0.021 | 0.008 | 0.695 | 0.008 |
| bicycle | 1287 | 0.010 | 0.009 | 0.286 | 0.002 |
| **all** | 37182 | **0.120** | **0.069** | 0.769 | 0.161 |

Low absolute scores are expected: the weights were never trained on drone
imagery, and VisDrone objects are tiny. The harness is the deliverable. Our AP
implementation matches pycocotools bit-for-bit on the CI subset
(`tests/test_metrics_det.py`, delta 2.8e-17).

### Latency, single image 640 px, 4 threads, p50 ms

| Backend | NANI, i5-1135G7 | GitHub runner, EPYC 9V74 |
|---|---|---|
| cv2 (OpenCV DNN) | 93 | 117 |
| ort-cpu | 86 | 218 |
| ort-openvino (CPU device) | 49 | n/a, plain onnxruntime build |

OpenVINO GPU on the Iris Xe segfaults inside the provider; the benchmark probes
it in a subprocess and records it as unavailable.

### Robustness, mAP@0.5 on a 100-image subset, baseline 0.180

| Perturbation | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| Gaussian blur k=3 / 7 / 13 | +1 % | −10 % | −24 % |
| brightness −30 / −60 / −90 | −1 % | −16 % | −32 % |
| brightness +30 / +60 / +90 | −1 % | −5 % | −25 % |
| JPEG quality 50 / 20 / 8 | −6 % | −10 % | −43 % |
| rotation 5° / 15° / 30° | +1 % | −35 % | −58 % |

Severity 0 of every row reproduces the unperturbed mAP exactly. Rotated ground
truth uses the enclosing axis-aligned box, so rotation numbers are a lower bound.

### Segmentation

Pending. The UNet fine-tune runs as a Kaggle script kernel
(`notebooks/seg_finetune.py`); the account currently has no GPU quota, so the
report marks the section "not available in this run" and `metrics.json`
carries `"segmentation": null`. Mask QA already runs.

## Run it

```bash
uv sync --extra openvino --extra dev     # or --extra cpu on machines without OpenVINO
uv run python scripts/fetch.py           # needs KAGGLE_API_TOKEN
uv run dvc pull                          # or: cp -r data/ci_subset/. data/
uv run pytest -q
uv run mkuav report --subset 50 --backend ort --out report.md --json metrics.json
```

`onnxruntime` and `onnxruntime-openvino` overwrite each other's files, so they
are mutually exclusive extras. CI uses `cpu`.

## CI

`.github/workflows/ci.yml` installs with uv, copies the committed subset into
`data/`, runs the tests, runs the report and uploads `report.md`, `metrics.json`
and per-stage JSON as artifacts. Wall time is about 2.5 minutes.

## Limitations

- COCO weights, not fine-tuned on VisDrone. Absolute mAP is low by design.
- Tricycle and awning-tricycle GT are excluded, person/people and car/van are merged.
- Ignored-region filtering is a centre-point approximation of the VisDrone protocol.
- Rotation robustness uses enclosing boxes for rotated GT.
- Robustness numbers are on 100 images, all other detection numbers on 548.
