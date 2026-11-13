"""Train a dual-branch forensic neural network to detect AI-cloned Chinese student homework."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "cn_homework_dataset"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cn_homework_forensic_net.pth"


def compute_physics_features(bgr: np.ndarray) -> np.ndarray:
    """Extract 8 page-level physical & micro-structural forensic features."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # 1. Bayer CFA lateral dispersion asymmetry: |corr(hf_R, hf_G) - corr(hf_R, hf_B)|
    k_hp = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    hf_r = cv2.filter2D(rgb[:, :, 0], -1, k_hp).ravel()
    hf_g = cv2.filter2D(rgb[:, :, 1], -1, k_hp).ravel()
    hf_b = cv2.filter2D(rgb[:, :, 2], -1, k_hp).ravel()
    corr_rg = float(np.corrcoef(hf_r, hf_g)[0, 1]) if np.std(hf_r) > 1e-6 and np.std(hf_g) > 1e-6 else 0.0
    corr_rb = float(np.corrcoef(hf_r, hf_b)[0, 1]) if np.std(hf_r) > 1e-6 and np.std(hf_b) > 1e-6 else 0.0
    bayer_asymmetry = abs(corr_rg - corr_rb)

    # 2. NPR 2x2 parity jump deviation
    p00 = gray[0 : h - 1 : 2, 0 : w - 1 : 2]
    p01 = gray[0 : h - 1 : 2, 1:w:2]
    p10 = gray[1:h:2, 0 : w - 1 : 2]
    p11 = gray[1:h:2, 1:w:2]
    mh = min(p00.shape[0], p10.shape[0])
    mw = min(p00.shape[1], p01.shape[1])
    intra_diff = np.mean(np.abs(p00[:mh, :mw] - p01[:mh, :mw])) + 1e-6
    inter_diff = np.mean(np.abs(p00[:mh, :mw] - p10[:mh, :mw]))
    cross_diff = np.mean(np.abs(p00[:mh, :mw] - p11[:mh, :mw]))
    jump_2x2 = (inter_diff + cross_diff) / (2.0 * intra_diff)
    npr_jump_dev = abs(1.0 - float(jump_2x2))

    # 3 & 4. VAE 4x4 and 8x8 harmonic peaks on clipped Laplacian of center 256x256
    cy, cx = h // 2, w // 2
    crop256 = gray[max(0, cy - 128) : cy + 128, max(0, cx - 128) : cx + 128]
    lap = cv2.Laplacian(crop256, cv2.CV_32F, ksize=3)
    lap_clip = np.clip(lap, -0.08, 0.08)
    fft_mag = np.abs(np.fft.fft2(lap_clip))
    med_mag = float(np.median(fft_mag)) + 1e-6
    nh, nw = fft_mag.shape
    vae_4x4 = float(fft_mag[nh // 4, nw // 4] / med_mag)
    vae_8x8 = float(fft_mag[nh // 8, nw // 8] / med_mag)

    # 5. Stroke ink CV (pen pressure variation)
    ink_mask = gray < 0.48
    if np.sum(ink_mask) > 50:
        ink_vals = gray[ink_mask]
        stroke_ink_cv = float(np.std(ink_vals) / (np.mean(ink_vals) + 1e-6))
    else:
        stroke_ink_cv = 0.0

    # 6. Baseline jitter std across horizontal text bands
    row_ink = np.sum(gray < 0.50, axis=1)
    active_rows = np.where(row_ink > w * 0.02)[0]
    if len(active_rows) > 10:
        # Split into 16 vertical slices and track local dark-pixel centroid shifts
        slice_w = max(8, w // 16)
        centroids = []
        y_grid = np.arange(h, dtype=np.float32)[:, None]
        dark = np.clip(0.55 - gray, 0.0, 1.0)
        for sx in range(0, w - slice_w + 1, slice_w):
            band = dark[:, sx : sx + slice_w]
            mass = float(np.sum(band))
            if mass > 1.0:
                cy_local = float(np.sum(band * y_grid) / mass)
                centroids.append(cy_local)
        baseline_jitter_std = float(np.std(np.diff(centroids))) if len(centroids) > 3 else 0.0
    else:
        baseline_jitter_std = 0.0

    # 7. Paper background high-pass noise std
    paper_mask = gray > 0.68
    lap_full = cv2.filter2D(gray, -1, k_hp)
    paper_noise_std = float(np.std(lap_full[paper_mask])) if np.sum(paper_mask) > 200 else float(np.std(lap_full))

    # 8. Edge halo ratio (soft anti-aliasing pixels vs core stroke pixels)
    core_count = float(np.sum(gray < 0.38)) + 1.0
    halo_count = float(np.sum((gray >= 0.38) & (gray < 0.62)))
    edge_halo_ratio = halo_count / core_count

    return np.array(
        [
            bayer_asymmetry,
            npr_jump_dev,
            vae_4x4,
            vae_8x8,
            stroke_ink_cv,
            baseline_jitter_std,
            paper_noise_std,
            edge_halo_ratio,
        ],
        dtype=np.float32,
    )


def build_4ch_tensor(bgr: np.ndarray) -> np.ndarray:
    """Build 4-channel (4, H, W) forensic tensor: [Gray, SRM_Lap, SRM_Diag, NPR_Parity_2x2]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    k_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    k_diag = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]], dtype=np.float32) * 0.5

    srm_lap = np.clip(cv2.filter2D(gray, -1, k_lap) * 6.0, -1.0, 1.0)
    srm_diag = np.clip(cv2.filter2D(gray, -1, k_diag) * 6.0, -1.0, 1.0)

    # Local 2x2 parity residual map upsampled back to (H, W)
    h, w = gray.shape
    even_r = gray[0 : h - 1 : 2, 0 : w - 1 : 2]
    odd_c = gray[0 : h - 1 : 2, 1:w:2]
    odd_r = gray[1:h:2, 0 : w - 1 : 2]
    diag_rc = gray[1:h:2, 1:w:2]
    mh = min(even_r.shape[0], odd_r.shape[0])
    mw = min(even_r.shape[1], odd_c.shape[1])
    parity = (even_r[:mh, :mw] + diag_rc[:mh, :mw]) - (odd_c[:mh, :mw] + odd_r[:mh, :mw])
    parity_full = cv2.resize(np.clip(parity * 8.0, -1.0, 1.0), (w, h), interpolation=cv2.INTER_NEAREST)

    return np.stack([gray * 2.0 - 1.0, srm_lap, srm_diag, parity_full], axis=0).astype(np.float32)


def extract_grid_crops(tensor_4ch: np.ndarray, crop_size: int = 128) -> List[np.ndarray]:
    """Extract 4 deterministic 128x128 crops covering the main handwriting region."""
    _, h, w = tensor_4ch.shape
    coords = [
        (h // 4 - crop_size // 2, w // 4 - crop_size // 2),
        (h // 4 - crop_size // 2, 3 * w // 4 - crop_size // 2),
        (3 * h // 4 - crop_size // 2, w // 4 - crop_size // 2),
        (3 * h // 4 - crop_size // 2, 3 * w // 4 - crop_size // 2),
    ]
    crops = []
    for y, x in coords:
        y0 = max(0, min(h - crop_size, y))
        x0 = max(0, min(w - crop_size, x))
        crops.append(tensor_4ch[:, y0 : y0 + crop_size, x0 : x0 + crop_size])
    return crops


class CNHomeworkDataset(Dataset):
    """In-memory cached dataset for fast GPU training and evaluation."""

    def __init__(
        self,
        records: List[Dict],
        dataset_root: Path,
        feat_mean: np.ndarray | None = None,
        feat_std: np.ndarray | None = None,
        is_train: bool = True,
    ) -> None:
        self.is_train = is_train
        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        raw_feats = []
        tensors = []
        labels = []

        for rec in records:
            img_path = dataset_root / rec["path"]
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            tensors.append(build_4ch_tensor(bgr))
            raw_feats.append(compute_physics_features(bgr))
            labels.append(int(rec["label"]))

        feats_arr = np.stack(raw_feats, axis=0)
        if feat_mean is None or feat_std is None:
            self.feat_mean = np.mean(feats_arr, axis=0)
            self.feat_std = np.std(feats_arr, axis=0) + 1e-6
        else:
            self.feat_mean = feat_mean
            self.feat_std = feat_std

        feats_norm = (feats_arr - self.feat_mean) / self.feat_std

        for t, f, lbl in zip(tensors, feats_norm, labels):
            if is_train:
                for crop in extract_grid_crops(t, crop_size=128):
                    self.samples.append((crop, f.astype(np.float32), lbl))
            else:
                # Keep all 4 crops stacked (4, 4, 128, 128) for page-level ensemble inference
                crops = np.stack(extract_grid_crops(t, crop_size=128), axis=0)
                self.samples.append((crops, f.astype(np.float32), lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_img, x_phys, lbl = self.samples[idx]
        if self.is_train and np.random.rand() < 0.5:
            x_img = x_img[:, :, ::-1].copy()
        return (
            torch.from_numpy(x_img),
            torch.from_numpy(x_phys),
            torch.tensor(lbl, dtype=torch.long),
        )


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class HomeworkForensicNet(nn.Module):
    """Dual-Branch Network: 4-ch SRM/NPR Spatial ResNet + 8-dim Handwriting Physics MLP."""

    def __init__(self, num_phys_features: int = 8) -> None:
        super().__init__()
        # Branch A: Spatial micro-texture + lattice CNN
        self.stem = nn.Sequential(
            nn.Conv2d(4, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.layer1 = ResBlock(24, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 96, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Linear(96, 64)

        # Branch B: Page-level handwriting physics MLP
        self.phys_mlp = nn.Sequential(
            nn.LayerNorm(num_phys_features),
            nn.Linear(num_phys_features, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        # Fusion Classifier
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 2),
        )

    def forward(self, x_img: torch.Tensor, x_phys: torch.Tensor) -> torch.Tensor:
        s = self.stem(x_img)
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.pool(s).flatten(1)
        s_emb = F.gelu(self.spatial_proj(s))
        p_emb = self.phys_mlp(x_phys)
        return self.head(torch.cat([s_emb, p_emb], dim=1))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC without external sklearn dependency."""
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1) / float(pos)
    fps = np.cumsum(y_sorted == 0) / float(neg)
    tps = np.concatenate(([0.0], tps))
    fps = np.concatenate(([0.0], fps))
    return float(np.trapezoid(tps, fps))


@torch.no_grad()
def evaluate_page_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_probs = []
    all_labels = []
    for crops_batch, phys_batch, lbl_batch in loader:
        # crops_batch: (B, 4, 4, 128, 128)
        b, n_crops, c, h, w = crops_batch.shape
        flat_crops = crops_batch.view(b * n_crops, c, h, w).to(device)
        flat_phys = phys_batch.unsqueeze(1).repeat(1, n_crops, 1).view(b * n_crops, -1).to(device)
        logits = model(flat_crops, flat_phys)
        probs = F.softmax(logits, dim=1)[:, 1].view(b, n_crops).mean(dim=1)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(lbl_batch.numpy().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = float(np.mean(y_pred == y_true))
    auc = compute_roc_auc(y_true, y_prob)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {
        "acc": round(acc, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "total": int(len(y_true)),
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.exists():
        from data.generate_cn_homework_dataset import build_dataset

        build_dataset(DATASET_DIR)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    train_ds = CNHomeworkDataset(manifest["train"], DATASET_DIR, is_train=True)
    val_ds = CNHomeworkDataset(
        manifest["val"],
        DATASET_DIR,
        feat_mean=train_ds.feat_mean,
        feat_std=train_ds.feat_std,
        is_train=False,
    )
    test_ds = CNHomeworkDataset(
        manifest["test"],
        DATASET_DIR,
        feat_mean=train_ds.feat_mean,
        feat_std=train_ds.feat_std,
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = HomeworkForensicNet(num_phys_features=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_val_auc = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        for x_img, x_phys, y in train_loader:
            x_img = x_img.to(device)
            x_phys = x_phys.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_img, x_phys)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_items += int(y.size(0))

        scheduler.step()
        train_loss = total_loss / max(1, total_items)
        train_acc = total_correct / max(1, total_items)
        val_metrics = evaluate_page_level(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_crop_acc": round(train_acc, 4),
                "val_acc": val_metrics["acc"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        if val_metrics["acc"] > best_val_acc or (
            math.isclose(val_metrics["acc"], best_val_acc) and val_metrics["roc_auc"] >= best_val_auc
        ):
            best_val_acc = val_metrics["acc"]
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feat_mean": train_ds.feat_mean.tolist(),
                    "feat_std": train_ds.feat_std.tolist(),
                    "best_val_acc": best_val_acc,
                    "best_val_roc_auc": best_val_auc,
                    "best_epoch": best_epoch,
                    "n_params": n_params,
                },
                CHECKPOINT_PATH,
            )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_page_level(model, test_loader, device)

    summary = {
        "device": device_name,
        "torch_version": torch.__version__,
        "n_params": n_params,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_size_kb": round(CHECKPOINT_PATH.stat().st_size / 1024.0, 1),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_roc_auc": best_val_auc,
        "test_metrics": test_metrics,
        "history_tail": history[-5:],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
