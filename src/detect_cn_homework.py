"""Inference and forensic visualization for AI-cloned Chinese student homework detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_cn_homework_detector import (
    CHECKPOINT_PATH,
    DATASET_DIR,
    HomeworkForensicNet,
    build_4ch_tensor,
    compute_physics_features,
    extract_grid_crops,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "eval_cn_homework"

PHYS_NAMES = [
    "bayer_r_g_b_asymmetry",
    "npr_jump_2x2_dev",
    "vae_harmonic_4x4",
    "vae_harmonic_8x8",
    "stroke_ink_cv",
    "baseline_jitter_std",
    "paper_noise_std",
    "edge_halo_ratio",
]


@torch.no_grad()
def predict_image(
    image_path: Path,
    model: HomeworkForensicNet,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    device: torch.device,
) -> Dict:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tensor_4ch = build_4ch_tensor(bgr)
    crops = np.stack(extract_grid_crops(tensor_4ch, crop_size=128), axis=0)
    raw_phys = compute_physics_features(bgr)
    norm_phys = ((raw_phys - feat_mean) / feat_std).astype(np.float32)

    crops_t = torch.from_numpy(crops).to(device)
    phys_t = torch.from_numpy(norm_phys).unsqueeze(0).repeat(crops_t.size(0), 1).to(device)

    logits = model(crops_t, phys_t)
    probs = F.softmax(logits, dim=1)[:, 1]
    ai_prob = float(probs.mean().item())
    verdict = "AI-CLONED HOMEWORK" if ai_prob >= 0.5 else "REAL STUDENT HANDWRITING"

    return {
        "image": str(image_path),
        "verdict": verdict,
        "ai_probability": round(ai_prob, 4),
        "real_probability": round(1.0 - ai_prob, 4),
        "crop_ai_probs": [round(float(p), 4) for p in probs.cpu().tolist()],
        "physics_features": {name: round(float(val), 5) for name, val in zip(PHYS_NAMES, raw_phys)},
        "bgr": bgr,
        "tensor_4ch": tensor_4ch,
    }


def save_forensic_panel(result: Dict, out_path: Path) -> None:
    bgr = result["bgr"]
    t4 = result["tensor_4ch"]
    h, w = bgr.shape[:2]

    # Panel 1: Original image with verdict header
    p1 = bgr.copy()
    is_ai = result["ai_probability"] >= 0.5
    banner_color = (30, 30, 200) if is_ai else (30, 150, 40)
    cv2.rectangle(p1, (0, 0), (w, 42), banner_color, -1)
    cv2.putText(
        p1,
        f"{result['verdict']} (AI={result['ai_probability']:.1%})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Panel 2: SRM Laplacian micro-texture map
    srm_vis = np.clip((np.abs(t4[1]) * 255.0), 0, 255).astype(np.uint8)
    p2 = cv2.applyColorMap(srm_vis, cv2.COLORMAP_INFERNO)
    cv2.rectangle(p2, (0, 0), (w, 42), (35, 35, 35), -1)
    cv2.putText(p2, "SRM Laplacian Residual", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: NPR 2x2 parity residual map + physics summary
    npr_vis = np.clip((np.abs(t4[3]) * 255.0), 0, 255).astype(np.uint8)
    p3 = cv2.applyColorMap(npr_vis, cv2.COLORMAP_VIRIDIS)
    cv2.rectangle(p3, (0, 0), (w, 42), (35, 35, 35), -1)
    cv2.putText(p3, "NPR 2x2 Parity Lattice", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    pf = result["physics_features"]
    lines = [
        f"bayer_asym: {pf['bayer_r_g_b_asymmetry']:.4f}",
        f"npr_dev:    {pf['npr_jump_2x2_dev']:.4f}",
        f"vae_4x4:    {pf['vae_harmonic_4x4']:.2f}",
        f"ink_cv:     {pf['stroke_ink_cv']:.4f}",
        f"halo_ratio: {pf['edge_halo_ratio']:.4f}",
    ]
    cv2.rectangle(p3, (6, h - 118), (245, h - 6), (20, 20, 20), -1)
    for idx, line in enumerate(lines):
        cv2.putText(p3, line, (14, h - 95 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

    panel = np.hstack([p1, p2, p3])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AI-cloned Chinese student homework images.")
    parser.add_argument("images", nargs="*", help="Paths to homework images to analyze.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for visual forensic panels.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = HomeworkForensicNet(num_phys_features=8).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feat_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.array(ckpt["feat_std"], dtype=np.float32)

    if args.images:
        targets = [Path(p) for p in args.images]
    else:
        targets = [
            DATASET_DIR / "test" / "real" / "cn_hw_real_0000.jpg",
            DATASET_DIR / "test" / "ai_cloned" / "cn_hw_ai_0000.jpg",
        ]

    out_dir = Path(args.out_dir)
    results: List[Dict] = []
    for img_path in targets:
        res = predict_image(img_path, model, feat_mean, feat_std, device)
        panel_path = out_dir / f"{img_path.stem}_forensic.jpg"
        save_forensic_panel(res, panel_path)
        res_clean = {k: v for k, v in res.items() if k not in ("bgr", "tensor_4ch")}
        res_clean["forensic_panel"] = str(panel_path)
        results.append(res_clean)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
