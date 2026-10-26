<h1 align="center">mk-uav-eval</h1>
<p align="center">An OpenCV evaluation harness for versioned UAV datasets: annotation QA, OpenCV DNN and onnxruntime inference, metrics cross-checked against pycocotools and torchmetrics, latency, robustness, and a report published by CI on every push.</p>

<p align="center">
  <a href="https://github.com/MaheshBhushan/mk-uav-eval/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/MaheshBhushan/mk-uav-eval/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/github/license/MaheshBhushan/mk-uav-eval"></a>
  <img alt="last commit" src="https://img.shields.io/github/last-commit/MaheshBhushan/mk-uav-eval">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
</p>
<p align="center">
  <a href="https://github.com/MaheshBhushan/mk-uav-eval/actions/workflows/ci.yml">Latest report (CI artifact)</a> ·
  <a href="https://github.com/MaheshBhushan/mk-uav-eval/releases/tag/models-v1">Models</a> ·
  <a href="https://www.kaggle.com/code/kodurimahesh/mk-uav-seg-finetune">Training kernel</a> ·
  <a href="#results">Results</a> ·
  <a href="#quickstart">Quickstart</a>
</p>

![YOLOv8n detections on a VisDrone frame through OpenCV DNN, and UNet segmentation on a held-out ICG image](assets/hero.jpg)

## Overview

Evaluating a vision model on drone imagery has three failure modes that a training
script never shows you: the annotations are wrong, the metric implementation is
wrong, or the number you quote was measured on a different backend than the one
you deploy. This repo attacks all three. It validates VisDrone boxes and ICG
semantic masks with named QA rules and stores the raw and cleaned datasets as two
DVC versions. It runs the same ONNX file through OpenCV DNN, onnxruntime and the
onnxruntime OpenVINO provider with one shared pre- and post-processing path, so
the forward pass is the only variable. Its own mAP and mIoU implementations are
asserted equal to pycocotools and torchmetrics in the test suite.

The harness is the deliverable. The models are pretrained COCO YOLOv8n and a
small UNet trained from scratch, and their absolute scores are low on purpose.

## Quickstart

```bash
git clone https://github.com/MaheshBhushan/mk-uav-eval.git && cd mk-uav-eval
uv sync --extra cpu --extra dev                              # or --extra openvino instead of cpu
gh release download models-v1 -p "*.onnx" -D models/         # 12 MB YOLOv8n + 31 MB UNet
cp -r data/ci_subset/. data/                                 # 50 VisDrone + 20 ICG images, committed
uv run pytest -q                                             # 15 tests, ~20 s
uv run mkuav report --subset 50 --backend ort --out report.md --json metrics.json
```

That is exactly what CI runs, in about two minutes. For the full datasets:

```bash
export KAGGLE_API_TOKEN=...                                  # kaggle.com/settings
uv run python scripts/fetch.py                               # VisDrone val (548) + ICG (400), ~6 GB download
uv run mkuav report --backend ort
```

> [!NOTE]
> `onnxruntime` and `onnxruntime-openvino` overwrite each other's files, so they are mutually exclusive uv extras. Pick `cpu` or `openvino`, never both. The default DVC remote is MinIO (`s3://mk-uav-eval/dvc`); point it at your server with `uv run dvc remote modify --local minio endpointurl <url>` and set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`. Without MinIO access, fetch from Kaggle or use the committed subset.

## Asyncio evaluation service

Python callers can submit detection and segmentation evaluations concurrently:

```python
import asyncio
from mkuav.service import EvaluationJob, EvaluationService

async def main():
    async with EvaluationService(max_concurrent=2, max_queue=8) as service:
        jobs = [
            EvaluationJob("detection", "models/yolov8n.onnx", backend="cv2",
                          data_root="data/ci_subset", version="v2", subset=10),
            EvaluationJob("segmentation", "models/seg_unet_r18.onnx", backend="ort",
                          data_root="data/ci_subset", subset=10),
        ]
        results = await asyncio.gather(
            *(service.evaluate(job, timeout=60) for job in jobs)
        )
        for result in results:
            print(result["dataset"], result["metrics"])
            print(result["queue_ms"], result["inference"]["p95_ms"])

asyncio.run(main())
```

The API uses the existing numerical evaluators. Detection selects sorted
annotation stems from the requested VisDrone v1/v2 version and applies its ignored
regions. Segmentation selects the last 40 sorted ICG mask stems, matching the
training holdout. Both tasks apply `subset` after split selection. Results include
the selected stems, split name, resolved data/model paths, backend and metrics.
Unreadable selected images or masks fail the job. The service never cleans datasets
or shares prediction caches. Inputs must remain unchanged while jobs run.

Each active job owns a fresh subprocess and model handle. Native CPU execution uses
four threads per process. `max_concurrent` bounds active workers; `max_queue` bounds
additional waiting jobs. Excess submissions raise `ServiceBusyError` immediately.
Backend errors and native crashes raise `BackendJobError` without stopping other
jobs. Missing models or backend providers fail explicitly.

`timeout` covers queueing and execution in seconds. Alternatively, pass
`deadline=asyncio.get_running_loop().time() + seconds`. Expiry raises `TimeoutError`.
Cancelling the awaiting task kills and reaps its worker before releasing capacity.
Context exit or `await service.aclose()` cancels all remaining jobs and rejects
new submissions. Use one service per event loop.

`queue_ms` measures submission to worker dispatch. `execution_ms` includes process
startup, imports, model loading, input reads, evaluation and serialization.
`inference` contains per-image `samples_ms`, count, total, mean, p50 and p95 for
the shared inference call, including preprocessing and postprocessing. These
samples exclude queueing, model loading, image reads and metric computation.
They include the first call and reflect CPU contention under concurrent load.
Undefined metric values serialize as JSON `null`.

## Dataset storage on MinIO

DVC stores dataset versions in a MinIO bucket. CI reads the 50-image subset through presigned URLs, so the workflow never holds MinIO credentials.

```bash
mc mirror data/ci_subset minio/mk-uav-eval/subsets/data-v2      # upload the subset once
MINIO_ENDPOINT=https://minio.example MINIO_ACCESS_KEY=... MINIO_SECRET_KEY=... \
  uv run python scripts/presign_subset.py --version data-v2 --expires 3600 --out subset-manifest.json
uv run python scripts/fetch_presigned.py subset-manifest.json --dest data/
```

`src/mkuav/storage.py` signs SigV4 GET URLs with the standard library and is tested against the AWS reference vector. Publish the manifest somewhere CI can read and set the repository variable `SUBSET_MANIFEST_URL`. The fetcher rejects expired manifests and paths outside the destination.

## What it does

| Stage | Command | Output |
|---|---|---|
| Annotation QA | `mkuav qa --version v1\|v2` | rule counts per dataset, `qa_report_*.json` |
| Dataset versioning | DVC, git tags `data-v1` (raw) and `data-v2` (cleaned) | `data/*.dvc` |
| Detection | `mkuav detect`, `mkuav eval-det` | boxes; per-class AP, mAP@0.5, mAP@0.5:0.95, P/R |
| Segmentation | `mkuav segment`, `mkuav eval-seg` | class map PNG; per-class IoU, mIoU, pixel accuracy |
| Latency | `mkuav bench` | p50 / p95 per backend |
| Robustness | `mkuav robust` | mAP under blur, brightness, JPEG, rotation |
| Report | `mkuav report` | `report.md`, `metrics.json`, per-stage JSON |

```mermaid
flowchart LR
  K[Kaggle mirrors] -->|scripts/fetch.py| V1[(data-v1 raw, DVC)]
  V1 -->|mkuav qa| V2[(data-v2 cleaned, DVC)]
  M[ONNX models, release asset] --> D[detect.py / segment.py]
  D --> B1[cv2.dnn] & B2[onnxruntime] & B3[ORT + OpenVINO]
  B1 & B2 & B3 --> P[shared NMS / argmax]
  V2 --> E[metrics_det.py / metrics_seg.py]
  P --> E
  E --> R[report.py]
  P --> L[bench.py] --> R
  P --> Q[perturb.py] --> R
  R --> A[report.md + metrics.json, CI artifact]
```

## Results

Every number here comes from `mkuav report` on `main`, on a laptop i5-1135G7 unless stated.

### Annotation QA, VisDrone val, raw (v1)

| Rule | Boxes | Files |
|---|---|---|
| ignored_region | 1378 | 340 |
| others_class | 32 | 29 |
| tiny_box (< 16 px², informational) | 103 | 45 |

Zero-area, out-of-bounds, duplicate, malformed and score-zero rows: none in this
split. ICG masks: 0 invalid class ids, 0 dimension mismatches, 23 of 24 classes
present. Because the only cleaning that mattered was dropping category 0 and 11
rows the loader already ignores, v1 and v2 score identically, and the report
says so instead of implying the cleaning moved the number. Ignored regions are
kept and used at evaluation time to suppress predictions inside them.

### Detection, YOLOv8n COCO weights, VisDrone val v2, 548 images

COCO classes are mapped onto six merged VisDrone classes. Tricycle classes have
no COCO equivalent and are excluded.

| Class | GT boxes | AP@0.5 | AP@0.5:0.95 | P@0.25 | R@0.25 |
|---|---|---|---|---|---|
| car | 16039 | 0.396 | 0.231 | 0.801 | 0.299 |
| person | 13969 | 0.148 | 0.057 | 0.769 | 0.076 |
| bus | 251 | 0.106 | 0.080 | 0.225 | 0.179 |
| truck | 750 | 0.037 | 0.026 | 0.247 | 0.049 |
| motor | 4886 | 0.021 | 0.008 | 0.695 | 0.008 |
| bicycle | 1287 | 0.010 | 0.009 | 0.286 | 0.002 |
| **all** | 37182 | **0.120** | **0.069** | 0.769 | 0.161 |

The weights never saw drone imagery and VisDrone objects are tiny, so low recall
is expected. What the harness proves is that the AP implementation matches
pycocotools to 2.8e-17 on the CI subset (`tests/test_metrics_det.py`).

![detections](assets/detection.jpg)

### Segmentation, from-scratch UNet, ICG Semantic Drone, 40 held-out images

| Metric | Value |
|---|---|
| mIoU, 24 classes | 0.167 |
| pixel accuracy | 0.721 |

Plain-convolution UNet, 360 training images, 25 epochs, ten minutes on a Kaggle
T4 (`notebooks/seg_finetune.py`). Large classes such as paved area and grass
score well, rare ones such as AR markers and bicycles score zero. OpenCV DNN and
onnxruntime produce identical argmax maps, and mIoU matches torchmetrics to 1e-5
(`tests/test_metrics_seg.py`). Only the 40 stems the kernel never trained on are
scored; `mkuav eval-seg --all` gives the optimistic number.

![ground truth versus prediction](assets/segmentation.jpg)

### Latency, one 640 px image, 4 threads, p50 in ms

| Backend | Laptop i5-1135G7 | GitHub runner, EPYC 9V74 |
|---|---|---|
| cv2 (OpenCV DNN) | 93 | 117 |
| ort-cpu | 86 | 218 |
| ort-openvino, CPU device | 49 | n/a, plain onnxruntime build |

OpenVINO halves the onnxruntime time on the same cores because its CPU plugin
fuses and re-tiles the convolutions for the specific ISA at load time, where the
default onnxruntime CPU provider runs generic kernels. OpenVINO's GPU device on
the Iris Xe segfaults inside the provider; the benchmark probes it in a
subprocess and records it as unavailable. If onnxruntime silently drops a
requested provider, the loader raises rather than reporting a CPU number under
the wrong name.

### Robustness, mAP@0.5 relative to clean, 100 images

![robustness](assets/robustness.png)

Severity 0 of every perturbation reproduces the unperturbed mAP exactly. JPEG
quality 8 and 30° rotation hurt most. Rotated ground truth uses the enclosing
axis-aligned box, so rotation is a lower bound.

## Repository structure

```
.github/workflows/ci.yml   tests + report on the committed subset, artifacts uploaded
assets/                    images in this README, generated by the harness
data/ci_subset/            50 VisDrone + 20 ICG images committed for CI
data/*.dvc                 DVC pointers for the full datasets (MinIO remote)
models/                    ONNX files land here (release asset), class list, training log
notebooks/seg_finetune.py  Kaggle script kernel that trains and exports the UNet
scripts/fetch.py           Kaggle download + layout for both datasets
src/mkuav/
  cli.py                   argparse entry point, one subcommand per stage
  qa.py                    annotation rules and the v2 cleaning step
  detect.py  segment.py    backend-agnostic loaders, shared pre/post-processing
  metrics_det.py           COCO-style AP, class mapping, ignored-region filter
  metrics_seg.py           confusion-matrix IoU, holdout selection
  service.py eval_worker.py bounded asyncio API and isolated evaluation worker
  bench.py  perturb.py     latency and robustness
  report.py                Markdown + JSON rendering
tests/                     15 tests incl. pycocotools and torchmetrics cross-checks
```

## Limitations

- COCO weights, not fine-tuned on VisDrone. Absolute mAP is low by design.
- Tricycle and awning-tricycle GT are excluded, person/people and car/van are merged.
- Ignored-region filtering is a centre-point approximation of the VisDrone protocol.
- The segmentation model has no pretrained encoder and 360 training images; it exists to exercise the harness.
- Robustness is measured on 100 images, all other detection numbers on 548.

## Acknowledgements

Data comes from [VisDrone2019-DET](http://aiskyeye.com/) (Zhu et al., *Detection and Tracking Meet Drones Challenge*, TPAMI 2021) via the Kaggle mirror `hassanmojab/visdrone-det`, and the [ICG Semantic Drone Dataset](http://dronedataset.icg.tugraz.at/) (TU Graz) via `bulentsiyah/semantic-drone-dataset`. Both are for non-commercial research use under their own terms. Detection weights are Ultralytics YOLOv8n, exported once to ONNX.

## License

MIT. See [LICENSE](LICENSE).
