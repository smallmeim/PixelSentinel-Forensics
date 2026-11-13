#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Multi-Modal Image Forensics & Debiasing Toolkit
=============================================================
A unified diagnostic and localization toolkit for:
1. Global AIGC Synthesis Detection with Reference-Guided Debiasing (RDD: 90-deg Rotation + Low-Pass Calibration)
2. Localized Scene/Handwritten Text Tampering Forensics (e.g., OpenCut, AnyText, DiffSTE)
   - SRM High-Frequency Sensor Noise Residual Analysis
   - Error Level Analysis (ELA) & Double Quantization Inspection
   - Stroke Edge Micro-Gradient & Sub-Stroke Texture Discontinuity
   - Patch-wise Background Complexity Stratification (L1-L4)

Usage:
    python src/forensics_toolkit.py --image path/to/image.jpg --output outputs/
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import ExifTags, Image, ImageChops, ImageFilter


def extract_metadata_forensics(img: Image.Image, img_path: Path) -> Dict[str, Any]:
    """Inspect EXIF metadata, color profile, and JPEG quantization tables."""
    info: Dict[str, Any] = {
        "filename": img_path.name,
        "format": img.format,
        "mode": img.mode,
        "size": list(img.size),
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "has_quantization_tables": hasattr(img, "quantization") and bool(getattr(img, "quantization", None)),
        "metadata_risk_flags": [],
    }

    exif_data = img.getexif()
    if exif_data:
        info["has_exif"] = True
        tag_map = {ExifTags.TAGS.get(k, k): str(v) for k, v in exif_data.items()}
        info["camera_make"] = tag_map.get("Make")
        info["camera_model"] = tag_map.get("Model")
        info["software"] = tag_map.get("Software")
    else:
        info["metadata_risk_flags"].append(
            "EXIF_STRIPPED: No camera EXIF found (typical of web AI editors like OpenCut or social re-encoding)."
        )

    if info["software"]:
        sw_lower = info["software"].lower()
        if any(k in sw_lower for k in ["stable diffusion", "midjourney", "comfyui", "opencut", "photoshop", "canvas"]):
            info["metadata_risk_flags"].append(f"SUSPICIOUS_SOFTWARE_TAG: {info['software']}")

    return info


def compute_ela_map(img_rgb: Image.Image, quality: int = 90) -> np.ndarray:
    """
    Compute Error Level Analysis (ELA) residual map by re-encoding at a fixed JPEG quality.
    Highlights localized paste/inpainting regions that have different compression history.
    """
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img_rgb, dtype=np.float32) - np.asarray(resaved, dtype=np.float32))
    ela_gray = np.mean(diff, axis=2)
    return ela_gray


def compute_srm_noise_residual(img_gray: np.ndarray) -> np.ndarray:
    """
    Apply Spatial Rich Model (SRM) high-pass kernels to extract sensor/inpainting noise residuals.
    When tools like OpenCut erase and re-render handwritten text via diffusion/inpainting,
    the local text bounding box exhibits a distinct noise variance shift ("noise vacuum" or grid).
    """
    # 3 classic SRM high-pass filters (horizontal, vertical, 2nd-order Laplacian-like edge residual)
    k1 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32) / 2.0
    k2 = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32) / 2.0
    k3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0

    padded = np.pad(img_gray, pad_width=1, mode="reflect")
    h, w = img_gray.shape

    def conv3x3(kernel: np.ndarray) -> np.ndarray:
        out = np.zeros((h, w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if kernel[dy, dx] != 0:
                    out += kernel[dy, dx] * padded[dy : dy + h, dx : dx + w]
        return np.abs(out)

    r1 = conv3x3(k1)
    r2 = conv3x3(k2)
    r3 = conv3x3(k3)
    residual = (r1 + r2 + r3) / 3.0
    return np.clip(residual, 0.0, 30.0)


def compute_stroke_edge_gradient(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute high-frequency edge gradient and second-order Laplacian halo response.
    Useful for inspecting handwritten strokes vs AI-rendered glyphs (OpenCut / AnyText).
    """
    padded = np.pad(img_gray, pad_width=1, mode="reflect")
    h, w = img_gray.shape
    gx = (
        -padded[0:h, 0:w] + padded[0:h, 2 : w + 2]
        - 2.0 * padded[1 : h + 1, 0:w] + 2.0 * padded[1 : h + 1, 2 : w + 2]
        - padded[2 : h + 2, 0:w] + padded[2 : h + 2, 2 : w + 2]
    )
    gy = (
        -padded[0:h, 0:w] - 2.0 * padded[0:h, 1 : w + 1] - padded[0:h, 2 : w + 2]
        + padded[2 : h + 2, 0:w] + 2.0 * padded[2 : h + 2, 1 : w + 1] + padded[2 : h + 2, 2 : w + 2]
    )
    mag = np.hypot(gx, gy)
    return mag


def compute_patch_complexity_and_debiased_score(
    img_rgb: Image.Image, patch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Implements:
    1. Background Complexity Stratification (L1 Pure/Simple -> L4 Complex)
    2. Reference-Guided Debiasing (RDD Concept: 90-degree rotation + Low-Pass Filter calibration)
       to separate intrinsic image complexity from synthesis/inpainting manifold smoothness.
    """
    arr = np.asarray(img_rgb.convert("L"), dtype=np.float32) / 255.0

    # Self-reference transformation 1: 90-degree rotation (preserves complexity, perturbs directional kernels)
    img_rot90 = img_rgb.transpose(Image.Transpose.ROTATE_90)
    # Self-reference transformation 2: Low-Pass Gaussian filter (isolates low-frequency background structure)
    img_lowpass = img_rgb.filter(ImageFilter.GaussianBlur(radius=2.0))

    # Proxy reconstruction error on original, rotated, and low-pass views
    def auto_reconstruction_proxy(im: Image.Image) -> np.ndarray:
        # Down-up bottleneck + wavelet-like residual proxy when full SD VAE is not loaded
        w, h = im.size
        bottleneck = im.resize((max(8, w // 4), max(8, h // 4)), Image.Resampling.BILINEAR).resize(
            (w, h), Image.Resampling.BICUBIC
        )
        diff = np.abs(np.asarray(im, dtype=np.float32) - np.asarray(bottleneck, dtype=np.float32)).mean(axis=2)
        return diff

    err_orig = auto_reconstruction_proxy(img_rgb)
    err_rot_raw = auto_reconstruction_proxy(img_rot90)
    # Rotate back 270 degrees to align spatially with original image
    err_rot = np.rot90(err_rot_raw, k=-1)
    if err_rot.shape != err_orig.shape:
        err_rot = np.resize(err_rot, err_orig.shape)
    err_lp = auto_reconstruction_proxy(img_lowpass)

    # Debiased anomaly map: subtract low-pass baseline and rotation common-mode complexity
    debiased_diff = np.abs(err_orig - 0.5 * err_rot - 0.5 * err_lp)

    # Patch-wise aggregation for complexity map & localized anomaly detection
    h, w = arr.shape
    ph = max(1, h // patch_size)
    pw = max(1, w // patch_size)
    complexity_map = np.zeros((ph, pw), dtype=np.float32)
    anomaly_map = np.zeros((ph, pw), dtype=np.float32)

    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    edge_binary = ((gx + gy) > 0.08).astype(np.float32)

    for i in range(ph):
        for j in range(pw):
            y0, y1 = i * patch_size, min(h, (i + 1) * patch_size)
            x0, x1 = j * patch_size, min(w, (j + 1) * patch_size)
            complexity_map[i, j] = float(np.mean(edge_binary[y0:y1, x0:x1]))
            anomaly_map[i, j] = float(np.mean(debiased_diff[y0:y1, x0:x1]))

    mean_edge_density = float(np.mean(complexity_map))
    if mean_edge_density < 0.03:
        tier = "L1 (Pure/Ultra-Simple Background - High False-Positive Risk for Naive Detectors)"
    elif mean_edge_density < 0.08:
        tier = "L2 (Simple Background - Moderate Bias Risk)"
    elif mean_edge_density < 0.16:
        tier = "L3 (Medium Background Complexity)"
    else:
        tier = "L4 (Complex High-Frequency Background)"

    raw_score = float(np.mean(err_orig))
    rot_score = float(np.mean(err_rot))
    lp_score = float(np.mean(err_lp))
    debiased_score = raw_score - 0.5 * rot_score - 0.5 * lp_score

    stats = {
        "mean_edge_density": round(mean_edge_density, 5),
        "complexity_tier": tier,
        "raw_reconstruction_proxy": round(raw_score, 4),
        "rot90_reference_proxy": round(rot_score, 4),
        "lowpass_reference_proxy": round(lp_score, 4),
        "debiased_calibrated_score": round(debiased_score, 4),
        "patch_anomaly_std": round(float(np.std(anomaly_map)), 4),
        "patch_anomaly_max_ratio": round(
            float(np.max(anomaly_map) / (np.mean(anomaly_map) + 1e-6)), 4
        ),
    }
    return complexity_map, anomaly_map, stats


def save_diagnostic_panel(
    img_rgb: Image.Image,
    ela_map: np.ndarray,
    srm_map: np.ndarray,
    edge_map: np.ndarray,
    complexity_map: np.ndarray,
    anomaly_map: np.ndarray,
    stats: Dict[str, Any],
    out_png: Path,
) -> None:
    """Render a 6-panel forensic diagnostic figure if matplotlib is available, else fallback to PIL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=140)
        fig.suptitle(
            f"PixelSentinel Multi-Modal Forensic Diagnostic Report\n"
            f"Complexity Tier: {stats['complexity_tier']} | Debiased Score: {stats['debiased_calibrated_score']}",
            fontsize=12,
            fontweight="bold",
        )

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("1. Input Image (RGB)")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(ela_map, cmap="inferno")
        axes[0, 1].set_title("2. ELA Recompression Residual (Q=90)")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(srm_map, cmap="viridis")
        axes[0, 2].set_title("3. SRM High-Freq Noise Residual (OpenCut Box Detector)")
        axes[0, 2].axis("off")
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(edge_map, cmap="magma")
        axes[1, 0].set_title("4. Stroke & Edge Micro-Gradient Map")
        axes[1, 0].axis("off")
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(complexity_map, cmap="Blues", interpolation="nearest")
        axes[1, 1].set_title("5. Patch Background Complexity Map (L1-L4)")
        axes[1, 1].axis("off")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im5 = axes[1, 2].imshow(anomaly_map, cmap="jet", interpolation="bicubic")
        axes[1, 2].set_title("6. Reference-Debiased Local Anomaly Map (RDD)")
        axes[1, 2].axis("off")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Lightweight PIL fallback
        norm_srm = np.uint8(255 * (srm_map / (np.max(srm_map) + 1e-6)))
        Image.fromarray(norm_srm).save(out_png)


def analyze_image(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run full 5-dimensional forensic and debiasing pipeline on a target image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    meta = extract_metadata_forensics(img, image_path)

    img_rgb = img.convert("RGB")
    img_gray = np.asarray(img.convert("L"), dtype=np.float32)

    ela_map = compute_ela_map(img_rgb, quality=90)
    srm_map = compute_srm_noise_residual(img_gray)
    edge_map = compute_stroke_edge_gradient(img_gray)
    complexity_map, anomaly_map, debias_stats = compute_patch_complexity_and_debiased_score(img_rgb)

    out_png = output_dir / f"{image_path.stem}_forensic_report.png"
    save_diagnostic_panel(img_rgb, ela_map, srm_map, edge_map, complexity_map, anomaly_map, debias_stats, out_png)

    report = {
        "metadata_forensics": meta,
        "debias_and_complexity_analysis": debias_stats,
        "signal_statistics": {
            "ela_mean": round(float(np.mean(ela_map)), 4),
            "ela_max": round(float(np.max(ela_map)), 4),
            "srm_noise_mean": round(float(np.mean(srm_map)), 4),
            "srm_noise_std": round(float(np.std(srm_map)), 4),
            "edge_gradient_mean": round(float(np.mean(edge_map)), 4),
        },
        "diagnostic_report_image": str(out_png.resolve()),
    }

    out_json = output_dir / f"{image_path.stem}_forensic_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PixelSentinel Multi-Modal Image Forensics & Debiasing Toolkit"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image to inspect")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save diagnostic artifacts")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    report = analyze_image(img_path, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
