#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Chinese Student Homework Handwriting Dataset Generator
(现代中文手写作业「真人手写 vs AI笔迹克隆伪造」仿真数据集生成器)
=================================================================
Generates paired & diverse 512x512 training/validation/test samples of modern Chinese student
homework (横格本/方格本/A4打印纸手写解答) across two classes:
  - label 0 (AUTHENTIC_STUDENT_HANDWRITING):
      Simulates real smartphone camera capture of student gel-pen/ballpoint handwriting on ruled/grid
      paper: natural per-character muscle jitter (slant/scale/baseline micro-variance), stroke-endpoint
      ink pooling, capillary paper-fiber bleed, multiplicative blending over ruling lines, sensor PRNU
      noise, and Bayer CFA chromatic dispersion.
  - label 1 (AI_CLONED_HOMEWORK):
      Simulates a student feeding past handwriting to an AI generator (LoRA Diffusion / Few-Shot
      Handwriting Synthesis / Style-Transfer Inpainting): cloned handwriting style & paper, but
      carrying VAE 4x4/8x8 upsampling lattice, uniform stroke pressure (no endpoint ink pooling),
      low intra-class glyph variance, anti-aliasing halos, and missing Bayer chromatic asymmetry.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CN_HOMEWORK_LINES = [
    "解：由题意可知，函数 f(x) = x^2 - 4x + 3 在区间 [0, 3] 上单调递减，",
    "因此当 x = 2 时取得最小值 f(2) = -1，代入原方程得 k = 5。",
    "答：综上所述，该物体的加速度 a = 2.4 m/s^2，方向与初速度相同。",
    "证明：连接 AC 与 BD 交于点 O，因为 ABCD 为平行四边形，所以 AO = OC。",
    "1. 细胞呼吸的主要场所是线粒体，有氧呼吸第二阶段产生二氧化碳。",
    "2. 唐代诗人王维的山水田园诗体现了“诗中有画，画中有诗”的意境。",
    "实验结论：在保持电阻不变的条件下，通过导体的电流与电压成正比。",
    "设甲乙两地相距 s 千米，则 v1 * t1 + v2 * t2 = s，解得 t = 3.5 小时。",
    "化学方程式：2H2O2 ==MnO2== 2H2O + O2↑，反应前后原子种类与数目守恒。",
    "读后感：通过阅读《呐喊》，我深刻体会到鲁迅先生对国民性的深刻剖析。",
]


def _get_cn_fonts() -> List[str]:
    candidates = [
        r"C:\Windows\Fonts\simkai.ttf",
        r"C:\Windows\Fonts\STKAITI.TTF",
        r"C:\Windows\Fonts\simfang.ttf",
        r"C:\Windows\Fonts\STFANGSO.TTF",
        r"C:\Windows\Fonts\msyh.ttc",
    ]
    return [p for p in candidates if Path(p).exists()]


def _create_paper_canvas(w: int, h: int, rng: np.random.Generator, is_real: bool) -> Tuple[np.ndarray, List[int]]:
    # Warm white / cream notebook paper with subtle illumination gradient
    base_rgb = np.array([rng.uniform(228, 246), rng.uniform(224, 242), rng.uniform(212, 234)], dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1.0 - 0.08 * (((xx - w * 0.48) / (w * 0.7)) ** 2 + ((yy - h * 0.45) / (h * 0.7)) ** 2)
    paper = np.ones((h, w, 3), dtype=np.float32) * base_rgb[None, None, :] * vignette[:, :, None]

    # Paper fiber micro-texture (stronger & more organic on real paper)
    fiber_amp = rng.uniform(2.0, 3.6) if is_real else rng.uniform(0.6, 1.4)
    paper += rng.normal(0.0, fiber_amp, (h, w, 3)).astype(np.float32)

    # Ruled lines (blue/gray horizontal notebook lines)
    line_ys = list(range(54, h - 20, 44))
    line_color = np.array([rng.uniform(165, 192), rng.uniform(180, 205), rng.uniform(195, 220)], dtype=np.float32)
    for ly in line_ys:
        paper[ly : ly + 1, 18 : w - 18, :] = 0.65 * paper[ly : ly + 1, 18 : w - 18, :] + 0.35 * line_color

    return paper, line_ys


def _simulate_bayer_optical_dispersion(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Real smartphone camera lens + Bayer CFA demosaicing introduces sub-pixel R/B lateral shift & noise."""
    out = arr.copy()
    alpha_r = float(rng.uniform(0.15, 0.42))
    alpha_b = float(rng.uniform(0.12, 0.38))
    r_shift = np.roll(out[:, :, 0], 1, axis=1)
    out[:, :, 0] = (1.0 - alpha_r) * out[:, :, 0] + alpha_r * r_shift
    b_shift = np.roll(out[:, :, 2], -1, axis=0)
    out[:, :, 2] = (1.0 - alpha_b) * out[:, :, 2] + alpha_b * b_shift
    # Real CMOS sensor PRNU + shot noise
    out += rng.normal(0.0, rng.uniform(0.9, 2.4), out.shape).astype(np.float32)
    return out


def _simulate_ai_vae_artifacts(arr: np.ndarray, rng: np.random.Generator, mode: int) -> np.ndarray:
    """
    AI-cloned handwriting exhibits:
      - Subtle to moderate VAE 4x4 & 8x8 upsampling checkerboard lattice
      - Coupled R/G/B smooth residual (missing Bayer optical dispersion)
    """
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    amp4 = rng.uniform(0.22, 1.6)
    amp8 = rng.uniform(0.08, 1.2)
    lattice = (
        amp4 * np.cos(2.0 * np.pi * xx / 4.0) * np.cos(2.0 * np.pi * yy / 4.0)
        + amp8 * np.cos(2.0 * np.pi * xx / 8.0) * np.cos(2.0 * np.pi * yy / 8.0)
    )
    out = arr.copy()
    if mode == 2:
        mask = np.zeros((h, w), dtype=np.float32)
        for ly in range(54, h - 20, 44):
            mask[max(0, ly - 32) : ly + 4, 24 : w - 24] = 1.0
        out += (lattice * mask)[:, :, None]
    else:
        out += lattice[:, :, None]

    # Joint RGB latent smoothing (reduces Bayer independence) + subtle synthetic grain
    pil_sm = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.45)))
    )
    mix = float(rng.uniform(0.15, 0.30))
    out = (1.0 - mix) * out + mix * np.asarray(pil_sm, dtype=np.float32)
    out += rng.normal(0.0, rng.uniform(0.4, 1.2), (h, w, 1)).astype(np.float32)  # correlated RGB grain
    return out


def generate_single_sample(
    out_path: Path,
    label: int,
    seed: int,
    fonts: List[str],
    size: int = 512,
) -> None:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    is_real = label == 0

    paper, line_ys = _create_paper_canvas(size, size, rng, is_real=is_real)
    font_path = py_rng.choice(fonts)
    base_fsize = int(rng.integers(22, 27))
    font = ImageFont.truetype(font_path, base_fsize)

    # Ink color: dark blue/black gel pen
    if py_rng.random() < 0.55:
        ink_rgb = np.array([rng.uniform(18, 42), rng.uniform(22, 48), rng.uniform(65, 115)], dtype=np.float32)
    else:
        ink_rgb = np.array([rng.uniform(20, 42), rng.uniform(20, 42), rng.uniform(24, 46)], dtype=np.float32)

    ink_layer = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(ink_layer)

    chosen_lines = [py_rng.choice(CN_HOMEWORK_LINES) for _ in range(len(line_ys))]
    for idx, ly in enumerate(line_ys):
        text = chosen_lines[idx]
        x_cur = int(rng.integers(26, 40))
        y_base = ly - base_fsize - 3
        for ch in text[:18]:
            if x_cur > size - 42:
                break
            if is_real:
                # Real human handwriting: per-character muscle jitter (Y jitter ±2.5px, spacing jitter, pressure jitter)
                dy = int(rng.integers(-2, 3))
                dx = int(rng.integers(-1, 3))
                alpha_ch = int(rng.integers(195, 252))
                sw = 1 if py_rng.random() < 0.28 else 0
                draw.text((x_cur + dx, y_base + dy), ch, fill=alpha_ch, font=font, stroke_width=sw)
                # Real pen endpoint ink pooling dots
                if py_rng.random() < 0.45:
                    px = x_cur + dx + int(rng.integers(4, 16))
                    py = y_base + dy + int(rng.integers(6, 18))
                    draw.ellipse([px, py, px + 2, py + 2], fill=min(255, alpha_ch + 25))
                x_cur += int(base_fsize * (0.58 if ch.isascii() else 0.96)) + int(rng.integers(-1, 4))
            else:
                # AI-cloned handwriting: unnaturally uniform baseline, spacing, and constant ink saturation
                draw.text((x_cur, y_base), ch, fill=225, font=font, stroke_width=0)
                x_cur += int(base_fsize * (0.56 if ch.isascii() else 0.98))

    if is_real:
        # Subtle capillary ink bleed
        ink_arr = np.asarray(ink_layer.filter(ImageFilter.GaussianBlur(radius=0.35)), dtype=np.float32) / 255.0
    else:
        # Realistic AI anti-aliased edge (subtle, matching modern diffusion/VAE sharpness)
        ink_arr = np.asarray(
            ink_layer.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.28, 0.48)))),
            dtype=np.float32,
        ) / 255.0

    m = ink_arr[:, :, None]
    # Multiplicative ink-paper blending on real vs alpha overlay on AI
    if is_real:
        composed = paper * (1.0 - 0.85 * m) + ink_rgb[None, None, :] * (0.85 * m)
        composed = _simulate_bayer_optical_dispersion(composed, rng)
        q = int(rng.integers(88, 96))
    else:
        composed = paper * (1.0 - m) + ink_rgb[None, None, :] * m
        ai_mode = int(seed % 3)
        composed = _simulate_ai_vae_artifacts(composed, rng, mode=ai_mode)
        q = int(rng.integers(84, 94))

    out_im = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
    out_im.save(out_path, quality=q)


def build_dataset(root_dir: Path, n_train: int = 240, n_val: int = 60, n_test: int = 60) -> None:
    fonts = _get_cn_fonts()
    if not fonts:
        raise RuntimeError("No Chinese TrueType fonts found in C:\\Windows\\Fonts")

    splits = [("train", n_train, 1000), ("val", n_val, 5000), ("test", n_test, 9000)]
    manifest: Dict[str, List[Dict[str, object]]] = {}

    for split_name, count, base_seed in splits:
        split_dir = root_dir / split_name
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "ai_cloned").mkdir(parents=True, exist_ok=True)
        records = []
        half = count // 2
        for i in range(half):
            p_real = split_dir / "real" / f"cn_hw_real_{i:04d}.jpg"
            generate_single_sample(p_real, label=0, seed=base_seed + i * 2, fonts=fonts)
            records.append({"path": str(p_real.resolve()), "label": 0})

            p_ai = split_dir / "ai_cloned" / f"cn_hw_ai_{i:04d}.jpg"
            generate_single_sample(p_ai, label=1, seed=base_seed + i * 2 + 1, fonts=fonts)
            records.append({"path": str(p_ai.resolve()), "label": 1})
        manifest[split_name] = records

    (root_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset generated at {root_dir}: train={n_train}, val={n_val}, test={n_test}")


if __name__ == "__main__":
    build_dataset(Path("data/cn_homework_dataset"), n_train=240, n_val=60, n_test=60)
