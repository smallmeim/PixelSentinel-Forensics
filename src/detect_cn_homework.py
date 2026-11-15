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

from opencut_forensics import analyze_opencut_tampering
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
    "patchcraft_rich_poor_ratio",
    "patchcraft_poor_vae_snr",
    "ffdn_hh_ortho_ratio",
    "ffdn_hh_kurtosis",
]


@torch.no_grad()
def predict_image(
    image_path: Path,
    model: HomeworkForensicNet,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    device: torch.device,
    out_dir: Path,
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
    neural_prob = float(probs.mean().item())

    pf = {name: round(float(val), 5) for name, val in zip(PHYS_NAMES, raw_phys)}

    # 1. Frontier Diffusion Physics Gate (PatchCraft 2024 + FFDN Wavelet HH 2024 + VAE 4x4 Harmonic)
    phys_gate_prob = 0.0
    if pf["vae_harmonic_4x4"] >= 3.2 and (
        pf["ffdn_hh_kurtosis"] >= 24.5
        or pf["patchcraft_poor_vae_snr"] >= 2.2
        or pf["bayer_r_g_b_asymmetry"] <= 0.0005
    ):
        phys_gate_prob = min(
            0.98,
            0.55
            + 0.08 * max(0.0, pf["vae_harmonic_4x4"] - 3.0)
            + 0.02 * max(0.0, pf["ffdn_hh_kurtosis"] - 23.0),
        )

    # 2. AAAI 2025 DAF + OpenCut Intra-Page Peer Token Calibration (catches local 2% edits)
    oc_res = analyze_opencut_tampering(image_path, out_dir)
    is_opencut = bool(oc_res.get("is_opencut_tampered", False))
    opencut_prob = float(oc_res.get("opencut_confidence", 0.0)) if is_opencut else 0.0

    ai_prob = max(neural_prob, phys_gate_prob, opencut_prob)
    if is_opencut and neural_prob < 0.5 and phys_gate_prob < 0.5:
        verdict = "OPENCUT-TAMPERED HOMEWORK"
    elif ai_prob >= 0.5:
        verdict = "AI-CLONED HOMEWORK"
    else:
        verdict = "REAL STUDENT HANDWRITING"

    return {
        "image": str(image_path),
        "verdict": verdict,
        "ai_probability": round(ai_prob, 4),
        "neural_branch_prob": round(neural_prob, 4),
        "physics_gate_prob": round(phys_gate_prob, 4),
        "opencut_local_tampered": is_opencut,
        "opencut_max_peer_zscore": oc_res.get("max_peer_zscore", 0.0),
        "opencut_primary_bbox": (
            oc_res["primary_opencut_token"]["bbox"] if oc_res.get("primary_opencut_token") else None
        ),
        "real_probability": round(1.0 - ai_prob, 4),
        "crop_ai_probs": [round(float(p), 4) for p in probs.cpu().tolist()],
        "physics_features": pf,
        "bgr": bgr,
        "tensor_4ch": tensor_4ch,
    }


def save_forensic_panel(result: Dict, out_path: Path) -> None:
    bgr = result["bgr"]
    t4 = result["tensor_4ch"]
    h, w = bgr.shape[:2]

    # Panel 1: Original image with verdict header + local OpenCut bbox if present
    p1 = bgr.copy()
    is_ai = result["ai_probability"] >= 0.5
    if result.get("opencut_local_tampered") and result.get("opencut_primary_bbox"):
        bx0, by0, bx1, by1 = result["opencut_primary_bbox"]
        cv2.rectangle(p1, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
        cv2.putText(
            p1,
            f"OpenCut Z={result['opencut_max_peer_zscore']:.2f}s",
            (bx0, max(54, by0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    banner_color = (30, 30, 200) if is_ai else (30, 150, 40)
    cv2.rectangle(p1, (0, 0), (w, 42), banner_color, -1)
    cv2.putText(
        p1,
        f"{result['verdict']} (AI={result['ai_probability']:.1%})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Panel 2: SRM Laplacian micro-texture map
    srm_vis = np.clip((np.abs(t4[1]) * 255.0), 0, 255).astype(np.uint8)
    p2 = cv2.applyColorMap(srm_vis, cv2.COLORMAP_INFERNO)
    cv2.rectangle(p2, (0, 0), (w, 42), (35, 35, 35), -1)
    cv2.putText(p2, "SRM Laplacian Residual", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: NPR 2x2 parity residual map + SOTA 12-dim physics summary
    npr_vis = np.clip((np.abs(t4[3]) * 255.0), 0, 255).astype(np.uint8)
    p3 = cv2.applyColorMap(npr_vis, cv2.COLORMAP_VIRIDIS)
    cv2.rectangle(p3, (0, 0), (w, 42), (35, 35, 35), -1)
    cv2.putText(p3, "NPR 2x2 + PatchCraft/FFDN", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    pf = result["physics_features"]
    lines = [
        f"bayer_asym:  {pf['bayer_r_g_b_asymmetry']:.4f}",
        f"vae_4x4:     {pf['vae_harmonic_4x4']:.2f}",
        f"pc_poor_snr: {pf['patchcraft_poor_vae_snr']:.2f}",
        f"ffdn_kurt:   {pf['ffdn_hh_kurtosis']:.2f}",
        f"daf_zscore:  {result.get('opencut_max_peer_zscore', 0.0):.2f}s",
    ]
    cv2.rectangle(p3, (6, h - 118), (255, h - 6), (20, 20, 20), -1)
    for idx, line in enumerate(lines):
        cv2.putText(p3, line, (14, h - 95 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

    panel = np.hstack([p1, p2, p3])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AI-cloned and OpenCut-tampered Chinese student homework.")
    parser.add_argument("images", nargs="*", help="Paths to homework images to analyze.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for visual forensic panels.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    n_phys = int(len(ckpt["feat_mean"]))
    model = HomeworkForensicNet(num_phys_features=n_phys).to(device)
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
        res = predict_image(img_path, model, feat_mean, feat_std, device, out_dir)
        panel_path = out_dir / f"{img_path.stem}_forensic.jpg"
        save_forensic_panel(res, panel_path)
        res_clean = {k: v for k, v in res.items() if k not in ("bgr", "tensor_4ch")}
        res_clean["forensic_panel"] = str(panel_path)
        results.append(res_clean)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
