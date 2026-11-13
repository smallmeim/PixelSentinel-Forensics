#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Unified Sample Evaluator (Signal Forensics + RDD Debiasing + Prompt Pack)
=======================================================================================
Runs the complete zero-training forensic evaluation pipeline on a target image:
1. 6-Panel Multi-Modal Signal Forensics (ELA, SRM Noise Residual, Stroke/Edge Gradient,
   Patch Complexity Map, Reference-Debiased Anomaly Map)
2. Visual + Textual Forensic Prompt Pack (Top-3 Suspicious Bounding Boxes + 2x Zoom Crop)
3. Automated Rule-Based & Statistical Verdict (combines SRM noise variance disparity,
   RDD debiased reconstruction score, and high-frequency roll-off ratio)

Usage:
    python src/evaluate_sample.py --image examples/mj5_market_vendor.png --output outputs/eval_mj5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from forensic_prompt_pipeline import generate_forensic_prompt_pack
from forensics_toolkit import analyze_image


def compute_spectral_rolloff_ratio(img_gray: np.ndarray) -> float:
    """
    Compute 2D FFT high-frequency to mid-frequency energy ratio.
    AI-generated or diffusion-inpainted images often exhibit spectral roll-off or
    periodic peaks in high frequencies compared to natural optical camera noise.
    """
    f = np.fft.fft2(img_gray - np.mean(img_gray))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx) / max(1.0, float(min(cy, cx)))

    mid_band = mag[(r >= 0.15) & (r < 0.45)]
    high_band = mag[r >= 0.55]
    ratio = float(np.mean(high_band) / (np.mean(mid_band) + 1e-6))
    return round(ratio, 4)


def compute_stroke_font_regularity(img_gray: np.ndarray, boxes: list) -> Dict[str, float]:
    """
    Quantify HUMAN-RULE-002 ("手写字太像印刷体 / Font-like Regularity in AI Handwriting"):
    In authentic handwriting, dark ink pixels inside strokes exhibit high intensity variance
    due to pen pressure changes and ink pooling at intersections. In AI-generated or
    OpenCut font-rendered handwriting, stroke interiors have unnaturally uniform intensity
    and mechanical stroke-edge gradient regularity.
    """
    if not boxes:
        return {"stroke_uniformity_index": 0.0, "ink_pressure_cv": 0.0}

    # Inspect the top suspicious text/stroke regions
    cvs = []
    for b in boxes[:3]:
        x0, y0, x1, y1 = b["bbox_xyxy"]
        patch = img_gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        # Threshold darkest 15% pixels as ink stroke foreground
        thresh = np.percentile(patch, 15)
        ink_pixels = patch[patch <= thresh]
        if ink_pixels.size > 10:
            cv = float(np.std(ink_pixels) / (np.mean(ink_pixels) + 1.0))
            cvs.append(cv)

    mean_cv = float(np.mean(cvs)) if cvs else 0.15
    # Lower ink pressure CV means flatter, more printed/font-like uniform strokes
    uniformity_index = round(float(np.clip(1.0 - (mean_cv / 0.25), 0.0, 1.0)), 4)
    return {
        "stroke_uniformity_index": uniformity_index,
        "ink_pressure_cv": round(mean_cv, 4),
    }


def evaluate_image_end_to_end(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run 6-panel forensic signal & RDD debiasing analysis
    toolkit_report = analyze_image(image_path, output_dir)

    # 2. Run Visual + Textual Forensic Prompt Pack (locks onto Top-3 suspicious boxes)
    prompt_pack = generate_forensic_prompt_pack(image_path, output_dir)

    img_gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    spectral_ratio = compute_spectral_rolloff_ratio(img_gray)

    # 3. Synthesize automated model pre-judgment & evidence summary
    meta = toolkit_report["metadata_forensics"]
    debias = toolkit_report["debias_and_complexity_analysis"]
    sig = toolkit_report["signal_statistics"]
    boxes = prompt_pack["suspicious_boxes"]
    stroke_stats = compute_stroke_font_regularity(img_gray, boxes)

    evidence_points = []
    suspicion_score = 0.0

    if not meta["has_exif"]:
        evidence_points.append("EXIF_MISSING: 无原始相机传感器 EXIF 元数据与厂商量化表指纹 (+0.20)")
        suspicion_score += 0.20

    if debias["patch_anomaly_max_ratio"] > 2.2:
        evidence_points.append(
            f"LOCAL_ANOMALY_PEAK: 局部块去偏重建异常峰值达均值 {debias['patch_anomaly_max_ratio']} 倍，"
            f"存在明显局部伪影或涂抹重绘不均匀区 (+0.30)"
        )
        suspicion_score += 0.30

    if boxes and boxes[0]["srm_noise_deviation_ratio"] > 0.35:
        b0 = boxes[0]
        evidence_points.append(
            f"SRM_NOISE_DISCONTINUITY: 候选框 #1 {b0['bbox_xyxy']} 处高频传感器噪声偏离全图中位数 "
            f"{b0['srm_noise_deviation_ratio']}x，呈现扩散模型平滑/伪影特征 (+0.25)"
        )
        suspicion_score += 0.25

    if stroke_stats["stroke_uniformity_index"] > 0.45:
        evidence_points.append(
            f"FONT_LIKE_STROKE_UNIFORMITY (HUMAN-RULE-002): 候选文字区笔画内部墨色均匀度高达 "
            f"{stroke_stats['stroke_uniformity_index']} (压感变异系数 CV={stroke_stats['ink_pressure_cv']})，"
            f"缺乏真实手写顿笔与飞白，呈‘印刷体化’AI字形特征 (+0.20)"
        )
        suspicion_score += 0.20

    if spectral_ratio < 0.72:
        evidence_points.append(
            f"SPECTRAL_ROLLOFF: 2D-FFT 高频/中频能量比偏低 ({spectral_ratio})，符合生成模型高频纹理衰减特征 (+0.20)"
        )
        suspicion_score += 0.20

    suspicion_score = min(0.99, round(suspicion_score, 2))
    verdict = "AI_GENERATED_OR_MODIFIED" if suspicion_score >= 0.45 else "LIKELY_AUTHENTIC"

    summary = {
        "image": str(image_path.resolve()),
        "model_verdict": verdict,
        "ai_forgery_probability": suspicion_score,
        "spectral_high_mid_ratio": spectral_ratio,
        "stroke_regularity_metrics": stroke_stats,
        "complexity_tier": debias["complexity_tier"],
        "debiased_rdd_score": debias["debiased_calibrated_score"],
        "top_suspicious_boxes": boxes,
        "automated_evidence_chain": evidence_points,
        "artifacts": {
            "six_panel_diagnostic_png": toolkit_report["diagnostic_report_image"],
            "visual_prompt_composite_jpg": prompt_pack["visual_prompt_image"],
            "structured_forensic_prompt_md": prompt_pack["generated_prompt_markdown"],
        },
    }

    (output_dir / f"{image_path.stem}_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end forensic evaluation on a sample image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/eval", help="Output directory")
    args = parser.parse_args()

    res = evaluate_image_end_to_end(Path(args.image), Path(args.output))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
