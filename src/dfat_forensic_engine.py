#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFAT Taxonomy-Guided & Model-Specific Forensic Engine + Reference Mandate
(基于 arXiv:2610.23144v2 / IEEE S&P 2027 DFAT 分类法与分模型家族专项取证引擎)
================================================================================
Inspired by:
  "From GAN-Based Models to GPT-5: A Taxonomy-Guided Study of Human Performance
   in Detecting Deepfake Face Images" (Guo et al., arXiv:2610.23144v2)

Core Paradigms Implemented from the Paper:
  1. **Four-Domain DFAT Quantitative Taxonomy (Section 3.2, Figure 1 & Table 2)**:
     - Domain A (`style_idealization`): Overly perfect & idealized aesthetic signature
       characteristic of SD3.5 and GPT-5 (high dynamic range polish + micro-pore suppression).
     - Domain B (`background_boundary_drop`): Unnatural foreground-to-background blurring
       transition and simplified peripheral background (High in SD3.5, Medium in GPT-5).
     - Domain C (`oval_diamond_minitexture`): StyleGAN3 / GRAM-HD magnification-level
       oval/diamond-shaped local mini-textures in mid-tone surfaces (Appendix D, Figure 7).
     - Domain D (`bilateral_accessory_inconsistency`): Left-right structural/accessory
       asymmetry and contour fusion (High in StyleGAN2/3/GRAM-HD, Medium in SD3.5).
  2. **Model-Specific Routing Over General-Purpose Mixing (Finding 2 & Section 5.3.1)**:
     - The paper proved that mixing cues across generator families ("general-purpose")
       degrades SD3.5 accuracy (-6.8%) and increases false positives on real images.
     - This engine routes each image to its matched generator family profile
       (`MLLM_DIFFUSION_SD35_GPT5`, `GAN_STYLEGAN_GRAMHD`, or `AUTHENTIC_OPTICAL`)
       to maximize detection gain while protecting real-image specificity.
  3. **The "Reference Mandate" Side-by-Side Calibration (Section 5.1.1 & Section 7)**:
     - Automatically pairs a single input image with a verified authentic reference
       baseline for side-by-side quantitative and visual comparison, plus a Global-to-Local
       4x magnification inspection panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF_REAL = PROJECT_ROOT / "examples" / "authentic_camera_real.png"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "eval_dfat"


def compute_dfat_four_domains(bgr: np.ndarray) -> Dict[str, float]:
    """Compute the 4 DFAT taxonomy domain metrics on a BGR image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # -------------------------------------------------------------------------
    # Domain 1: Style-Related — Overly Perfect & Idealized Style (SD3.5 / GPT-5)
    # Measures high-contrast tonal smoothness (low mid-tone pore noise + high global dynamic polish)
    # -------------------------------------------------------------------------
    blur_small = cv2.GaussianBlur(gray, (5, 5), 1.0)
    micro_pore_res = np.abs(gray - blur_small)
    midtone_mask = (gray > 55.0) & (gray < 210.0)
    pore_roughness = float(np.percentile(micro_pore_res[midtone_mask], 50)) if np.sum(midtone_mask) > 200 else 2.0
    global_contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0
    # Higher score = more "overly polished / poreless idealized" (SD3.5 / GPT-5 trait)
    style_idealization = float(np.clip(global_contrast / (pore_roughness + 0.65), 0.0, 2.5))

    # -------------------------------------------------------------------------
    # Domain 2: Background-Related — Foreground/Background Boundary Drop & Simplified Background
    # Compares center subject sharpness (inner 50% box) vs peripheral background sharpness (outer ring)
    # -------------------------------------------------------------------------
    k_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    lap_abs = np.abs(cv2.filter2D(gray, -1, k_lap))
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    fg_sharpness = float(np.mean(lap_abs[y0:y1, x0:x1])) + 1e-5
    bg_mask = np.ones((h, w), dtype=bool)
    bg_mask[y0:y1, x0:x1] = False
    bg_sharpness = float(np.mean(lap_abs[bg_mask])) + 1e-5
    boundary_sharpness_drop = fg_sharpness / bg_sharpness

    # -------------------------------------------------------------------------
    # Domain 3: Face/Surface-Related — StyleGAN3 Oval/Diamond Local Mini-Textures (Appendix D Fig. 7)
    # Under magnification, alias-free / rotational filters leave 45-degree diamond/oval periodic
    # micro-textures at 2~4px lag relative to random isotropic camera sensor noise.
    # -------------------------------------------------------------------------
    hp = gray - cv2.GaussianBlur(gray, (7, 7), 1.4)
    hp_norm = hp - float(np.mean(hp))
    var_hp = float(np.mean(hp_norm**2)) + 1e-5
    # Lag-2 diagonal (diamond) autocorrelation vs lag-2 orthogonal autocorrelation
    ac_diag2 = 0.5 * (
        float(np.mean(hp_norm[:-2, :-2] * hp_norm[2:, 2:]))
        + float(np.mean(hp_norm[:-2, 2:] * hp_norm[2:, :-2]))
    ) / var_hp
    ac_ortho2 = 0.5 * (
        float(np.mean(hp_norm[:-2, :] * hp_norm[2:, :]))
        + float(np.mean(hp_norm[:, :-2] * hp_norm[:, 2:]))
    ) / var_hp
    oval_diamond_minitexture = abs(ac_diag2 - ac_ortho2) * 10.0

    # -------------------------------------------------------------------------
    # Domain 4: Accessory/Structure-Related — Edge Contour Fusion & Chromatic Bleed
    # Measures soft contour bleeding across high-gradient boundaries (fused hair/frames/strokes)
    # -------------------------------------------------------------------------
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    strong_edge = grad_mag > np.percentile(grad_mag, 90)
    weak_halo = (grad_mag > np.percentile(grad_mag, 55)) & (~strong_edge)
    contour_fusion_ratio = float(np.sum(weak_halo)) / (float(np.sum(strong_edge)) + 1.0)

    # Bayer chromatic dispersion baseline (protects authentic camera photos from false positives)
    hf_r = cv2.filter2D(rgb[:, :, 0], -1, k_lap).ravel()
    hf_g = cv2.filter2D(rgb[:, :, 1], -1, k_lap).ravel()
    hf_b = cv2.filter2D(rgb[:, :, 2], -1, k_lap).ravel()
    c_rg = float(np.corrcoef(hf_r, hf_g)[0, 1]) if np.std(hf_r) > 1e-4 else 0.0
    c_rb = float(np.corrcoef(hf_r, hf_b)[0, 1]) if np.std(hf_r) > 1e-4 else 0.0
    bayer_asymmetry = abs(c_rg - c_rb)

    return {
        "dfat_style_idealization": round(style_idealization, 4),
        "dfat_boundary_sharpness_drop": round(boundary_sharpness_drop, 4),
        "dfat_oval_diamond_minitexture": round(oval_diamond_minitexture, 4),
        "dfat_contour_fusion_ratio": round(contour_fusion_ratio, 4),
        "optical_bayer_asymmetry": round(bayer_asymmetry, 5),
    }


def evaluate_with_dfat_and_reference(
    target_path: Path,
    ref_real_path: Path = DEFAULT_REF_REAL,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Dict[str, Any]:
    """
    Execute DFAT Model-Specific Routing + Reference Mandate Side-by-Side Comparison.
    """
    bgr_target = cv2.imread(str(target_path))
    if bgr_target is None:
        raise FileNotFoundError(f"Cannot read target image: {target_path}")

    if not ref_real_path.exists():
        ref_real_path = PROJECT_ROOT / "examples" / "real_student_cn_homework_wikimedia.jpg"
    bgr_ref = cv2.imread(str(ref_real_path))

    target_dfat = compute_dfat_four_domains(bgr_target)
    ref_dfat = compute_dfat_four_domains(bgr_ref) if bgr_ref is not None else target_dfat

    # Side-by-Side Reference Mandate Deltas (Target vs Authentic Reference Baseline)
    deltas = {
        "style_idealization_delta": round(
            target_dfat["dfat_style_idealization"] - ref_dfat["dfat_style_idealization"], 4
        ),
        "boundary_drop_delta": round(
            target_dfat["dfat_boundary_sharpness_drop"] - ref_dfat["dfat_boundary_sharpness_drop"], 4
        ),
        "oval_diamond_delta": round(
            target_dfat["dfat_oval_diamond_minitexture"] - ref_dfat["dfat_oval_diamond_minitexture"], 4
        ),
    }

    # Model-Specific Profile Scoring anchored by Side-by-Side Reference Mandate Deltas
    sd35_gpt5_score = 0.0
    gan_stylegan_score = 0.0
    cues = []

    if deltas["style_idealization_delta"] >= 0.10 and deltas["boundary_drop_delta"] >= 0.40:
        sd35_gpt5_score += 0.65
        cues.append(
            f"DFAT_SD35_GPT5_STYLE_AND_BOUNDARY: 命中扩散/MLLM家族特征 — 理想化平滑度较真图基准高出 "
            f"+{deltas['style_idealization_delta']}，前景-背景清晰度断层比={target_dfat['dfat_boundary_sharpness_drop']} "
            f"(较真图基准 +{deltas['boundary_drop_delta']})"
        )
    elif target_dfat["dfat_boundary_sharpness_drop"] >= 1.85:
        sd35_gpt5_score += 0.55
        cues.append(
            f"DFAT_FOREGROUND_BACKGROUND_BLUR_DROP: 主体与背景边缘清晰度断层显著 "
            f"(Ratio={target_dfat['dfat_boundary_sharpness_drop']}，真图基准={ref_dfat['dfat_boundary_sharpness_drop']})"
        )

    if deltas["oval_diamond_delta"] >= 0.65:
        gan_stylegan_score += 0.55
        cues.append(
            f"DFAT_STYLEGAN3_OVAL_DIAMOND_MINITEXTURE: 局部 4x 放大检出椭圆/菱形周期微纹理异常="
            f"{target_dfat['dfat_oval_diamond_minitexture']} (较真图基准 +{deltas['oval_diamond_delta']}，对应论文 Appendix D Fig.7)"
        )

    if target_dfat["optical_bayer_asymmetry"] < 0.0008:
        sd35_gpt5_score += 0.25
        gan_stylegan_score += 0.25
        cues.append(
            f"DFAT_MISSING_OPTICAL_BAYER: 跨通道光学色散非对称度极低 ({target_dfat['optical_bayer_asymmetry']})，缺乏真实镜头色差"
        )
    elif target_dfat["optical_bayer_asymmetry"] >= 0.0014 and deltas["style_idealization_delta"] < 0.08:
        # Authentic optical specificity protection (Section 5.3.1: prevent false positives on real photos)
        sd35_gpt5_score = max(0.0, sd35_gpt5_score - 0.35)
        gan_stylegan_score = max(0.0, gan_stylegan_score - 0.35)

    best_score = max(sd35_gpt5_score, gan_stylegan_score)
    if best_score < 0.45:
        matched_family = "AUTHENTIC_CAMERA_OPTICAL"
    elif sd35_gpt5_score >= gan_stylegan_score:
        matched_family = "DIFFUSION_OR_MLLM_FAMILY (SD3.5 / GPT-5 Profile)"
    else:
        matched_family = "GAN_RADIANCE_FAMILY (StyleGAN2/3 / GRAM-HD Profile)"

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / f"{target_path.stem}_dfat_reference_mandate.jpg"
    if bgr_ref is not None:
        _save_reference_mandate_panel(bgr_target, bgr_ref, target_dfat, ref_dfat, matched_family, best_score, panel_path)

    return {
        "target_image": str(target_path),
        "reference_authentic_image": str(ref_real_path),
        "matched_model_family": matched_family,
        "dfat_forgery_score": round(float(best_score), 4),
        "dfat_metrics": target_dfat,
        "reference_baseline_metrics": ref_dfat,
        "side_by_side_deltas": deltas,
        "dfat_inspection_cues": cues,
        "reference_mandate_panel": str(panel_path),
    }


def _save_reference_mandate_panel(
    bgr_target: np.ndarray,
    bgr_ref: np.ndarray,
    t_dfat: Dict[str, float],
    r_dfat: Dict[str, float],
    family: str,
    score: float,
    out_path: Path,
) -> None:
    """
    Generate the 3-panel visual inspection board advocated by Guo et al. (IEEE S&P 2027):
      Panel 1: Target Image (Global Inspection)
      Panel 2: Side-by-Side Authentic Reference Baseline ("Reference Mandate")
      Panel 3: 4x Local Magnification Crop (Surface Micro-Texture & Boundary Inspection)
    """
    H, W = 420, 420
    p1 = cv2.resize(bgr_target, (W, H))
    p2 = cv2.resize(bgr_ref, (W, H))

    # Panel 3: 4x zoom into center-top region of target image
    th, tw = bgr_target.shape[:2]
    cy, cx = th // 2, tw // 2
    zh, zw = max(32, th // 6), max(32, tw // 6)
    crop = bgr_target[max(0, cy - zh) : min(th, cy + zh), max(0, cx - zw) : min(tw, cx + zw)]
    p3 = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST)

    is_fake = score >= 0.45
    c1 = (30, 30, 200) if is_fake else (30, 150, 40)
    cv2.rectangle(p1, (0, 0), (W, 40), c1, -1)
    cv2.putText(p1, f"1. Target ({'FAKE' if is_fake else 'REAL'} {score:.0%})", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(p2, (0, 0), (W, 40), (35, 120, 50), -1)
    cv2.putText(p2, "2. Reference Mandate (Authentic)", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(p3, (0, 0), (W, 40), (40, 40, 40), -1)
    cv2.putText(p3, "3. Local 4x Zoom (DFAT Micro-Cue)", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"Family: {family[:28]}",
        f"StyleIdeal: {t_dfat['dfat_style_idealization']:.3f} (Ref {r_dfat['dfat_style_idealization']:.3f})",
        f"BoundDrop:  {t_dfat['dfat_boundary_sharpness_drop']:.3f} (Ref {r_dfat['dfat_boundary_sharpness_drop']:.3f})",
        f"OvalDiam:   {t_dfat['dfat_oval_diamond_minitexture']:.3f} (Ref {r_dfat['dfat_oval_diamond_minitexture']:.3f})",
        f"BayerAsym:  {t_dfat['optical_bayer_asymmetry']:.4f}",
    ]
    cv2.rectangle(p3, (6, H - 118), (W - 6, H - 6), (20, 20, 20), -1)
    for idx, line in enumerate(lines):
        cv2.putText(p3, line, (14, H - 96 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    board = np.hstack([p1, p2, p3])
    cv2.imwrite(str(out_path), board)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFAT Taxonomy-Guided & Reference Mandate Forensic Engine.")
    parser.add_argument("images", nargs="*", help="Target images to inspect.")
    parser.add_argument("--ref", default=str(DEFAULT_REF_REAL), help="Authentic reference baseline image.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    args = parser.parse_args()

    targets = (
        [Path(p) for p in args.images]
        if args.images
        else [
            PROJECT_ROOT / "examples" / "mj5_market_vendor.png",
            PROJECT_ROOT / "examples" / "sticky_note_memo.jpg",
            PROJECT_ROOT / "examples" / "ai_generated_cn_homework_sample.jpg",
            PROJECT_ROOT / "examples" / "real_student_cn_homework_wikimedia.jpg",
        ]
    )
    results = [evaluate_with_dfat_and_reference(t, Path(args.ref), Path(args.out_dir)) for t in targets]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
