"""Fine-tune a small plain-conv UNet on the ICG Semantic Drone Dataset and export ONNX.

Kaggle script kernel. Flat, one-off — no config files, no argparse.

No pip installs and no pretrained-weight downloads: Kaggle GPU kernels in this
account run without real internet access even with enable_internet=true, so
this script trains a from-scratch plain-conv UNet (torch + cv2 + onnxruntime,
all preinstalled in the Kaggle python image) instead of depending on
segmentation-models-pytorch or torchvision ImageNet weights.
"""
import csv
import glob
import json
import os
import random
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Attached datasets mount somewhere under /kaggle/input; the exact slug has
# varied between runs, so search from the top and log what is there.
DATA_ROOT = "/kaggle/input"
for _root, _dirs, _files in os.walk(DATA_ROOT):
    if _root.count(os.sep) - DATA_ROOT.count(os.sep) < 3:
        print(f"{_root}: dirs={_dirs[:8]} files={_files[:4]}")

_csv_matches = glob.glob(f"{DATA_ROOT}/**/class_dict_seg.csv", recursive=True)
assert _csv_matches, f"class_dict_seg.csv not found under {DATA_ROOT}"
CSV_PATH = _csv_matches[0]

_img_matches = glob.glob(f"{DATA_ROOT}/**/original_images/*.jpg", recursive=True)
assert _img_matches, f"original_images/*.jpg not found under {DATA_ROOT}"
IMG_DIR = os.path.dirname(_img_matches[0])

_mask_matches = glob.glob(f"{DATA_ROOT}/**/label_images_semantic/*.png", recursive=True)
assert _mask_matches, f"label_images_semantic/*.png not found under {DATA_ROOT}"
MASK_DIR = os.path.dirname(_mask_matches[0])

print(f"CSV_PATH={CSV_PATH}")
print(f"IMG_DIR={IMG_DIR}")
print(f"MASK_DIR={MASK_DIR}")

RESIZE_W, RESIZE_H = 768, 512
CROP = 512
NUM_CLASSES = 24
BATCH_SIZE = 8
MAX_EPOCHS = 140
MAX_SECONDS = 55 * 60  # T4 does ~25 s/epoch with the in-RAM cache
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} cuda_available={torch.cuda.is_available()} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
assert device.type == "cuda", "No GPU allocated to this kernel; check GPU quota / phone verification"

# class names, in class-index order (0..23)
class_names = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row["name"])
assert len(class_names) == NUM_CLASSES, f"expected {NUM_CLASSES} classes, got {len(class_names)}"
with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=2)

img_paths = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
mask_paths = sorted(glob.glob(f"{MASK_DIR}/*.png"))
assert len(img_paths) == len(mask_paths) and len(img_paths) > 0, "image/mask count mismatch"

# sanity-check masks are single-channel class-index maps, not RGB
_sample_mask = cv2.imread(mask_paths[0], cv2.IMREAD_UNCHANGED)
assert _sample_mask is not None, f"failed to read mask {mask_paths[0]}"
assert _sample_mask.ndim == 2, f"expected single-channel mask, got shape {_sample_mask.shape} for {mask_paths[0]}"
assert _sample_mask.max() < NUM_CLASSES, (
    f"mask values exceed {NUM_CLASSES - 1} (max={_sample_mask.max()}) for {mask_paths[0]}; "
    "looks like it was read as RGB instead of a class-index map"
)

pairs = list(zip(img_paths, mask_paths))
train_pairs = pairs[:-40]
val_pairs = pairs[-40:]
print(f"train={len(train_pairs)} val={len(val_pairs)}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_resized(pair):
    img_path, mask_path = pair
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # single-channel class index
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_NEAREST)
    return img, mask


class DroneSegDataset(Dataset):
    """Decodes and resizes every 6000x4000 image once, up front; ~0.6 GB for 400 pairs."""

    def __init__(self, pairs, train):
        t0 = time.time()
        self.items = [_load_resized(p) for p in pairs]
        self.train = train
        print(f"cached {len(self.items)} resized pairs in {time.time() - t0:.0f}s")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, mask = self.items[idx]

        if self.train:
            max_x = RESIZE_W - CROP
            max_y = RESIZE_H - CROP
            x0 = random.randint(0, max_x)
            y0 = random.randint(0, max_y)
        else:
            x0 = (RESIZE_W - CROP) // 2
            y0 = (RESIZE_H - CROP) // 2
        img = img[y0:y0 + CROP, x0:x0 + CROP]
        mask = mask[y0:y0 + CROP, x0:x0 + CROP]

        if self.train and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img.copy()), torch.from_numpy(mask.astype(np.int64).copy())


train_ds = DroneSegDataset(train_pairs, train=True)
val_ds = DroneSegDataset(val_pairs, train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Plain-conv UNet: maxpool-downsample encoder, upsample+conv decoder.

    No transformer/attention ops and no ONNX-unfriendly upsampling modes, so
    it loads cleanly in cv2.dnn (see export settings below).
    """

    def __init__(self, in_ch=3, num_classes=24, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


model = UNet(in_ch=3, num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR / 50)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def compute_miou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


train_log = []
start_time = time.time()

for epoch in range(MAX_EPOCHS):
    if time.time() - start_time > MAX_SECONDS:
        print("Wall-time budget exhausted, stopping training.")
        break

    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    model.eval()
    ious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            for p, m in zip(preds, masks):
                ious.append(compute_miou(p, m, NUM_CLASSES))
    val_miou = float(np.mean(ious))
    elapsed = time.time() - start_time
    scheduler.step()
    print(f"epoch={epoch} train_loss={train_loss:.4f} val_mIoU={val_miou:.4f} elapsed={elapsed:.0f}s")
    train_log.append({"epoch": epoch, "train_loss": train_loss, "val_miou": val_miou, "elapsed_s": elapsed})

    if time.time() - start_time > MAX_SECONDS:
        print("Wall-time budget exhausted after epoch, stopping training.")
        break

with open("train_log.json", "w") as f:
    json.dump(train_log, f, indent=2)

# --- Export to ONNX ---
model.eval()
dummy = torch.randn(1, 3, CROP, CROP, device=device)
onnx_path = "seg_unet_r18.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=12,
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=True,
    dynamo=False,  # legacy TorchScript exporter; the dynamo path needs onnxscript, absent on Kaggle
)
print(f"Exported ONNX to {onnx_path}")

# --- Verify ORT matches torch (onnxruntime is preinstalled in the Kaggle python image) ---
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("onnxruntime not installed in this image; ORT parity is checked locally instead")

if ort is not None:
    with torch.no_grad():
        torch_logits = model(dummy)
    torch_argmax = torch.argmax(torch_logits, dim=1).cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_input = dummy.cpu().numpy()
    ort_out = sess.run(None, {"images": ort_input})[0]
    ort_argmax = np.argmax(ort_out, axis=1)

    match_frac = float((torch_argmax == ort_argmax).mean())
    print(f"ORT vs torch argmax agreement: {match_frac:.4f}")
    assert match_frac > 0.99, f"ORT/torch mismatch too high: {match_frac}"

    print("DONE")
