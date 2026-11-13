#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic AI-Modified & Text-Tampered Dataset Generator
=======================================================
Helps build a balanced training & validation dataset for AI Image Detection:
1. Takes user-provided real images (or generates synthetic sample images for smoke testing)
2. Automatically creates localized tampered / text-modified counterparts simulating:
   - OpenCut-style local text box erasure (smooth diffusion/inpainting proxy) + new glyph rendering
   - Local patch splicing / frequency smoothing
   - Social-media (WeChat-like) JPEG recompression & resolution degradation
3. Organizes output into standard folder structure ready for `src/train.py`:
   data/dataset/
   ├── train/
   │   ├── 0_real/
   │   └── 1_fake/
   └── val/
       ├── 0_real/
       └── 1_fake/

Usage:
    # 1. Generate a quick sanity-check dataset (40 images) to test the training pipeline immediately:
    python src/data/synthesize_tampered_data.py --create_demo_data --output_dir data/demo_dataset

    # 2. Build from your own real photos folder (automatically creates OpenCut-like modified pairs):
    python src/data/synthesize_tampered_data.py --real_dir path/to/my_real_photos --output_dir data/dataset
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def simulate_opencut_text_tampering(img: Image.Image, rng: random.Random) -> Tuple[Image.Image, List[int]]:
    """
    Simulate OpenCut / Scene Text Editing on a real image:
    1. Select a rectangular text-like bounding box
    2. Erase original content via localized multi-scale smoothing + low-noise synthesis
       (destroying camera sensor PRNU/SRM high-frequency noise inside the box, just like VAE inpainting)
    3. Render synthetic handwritten/printed text strokes with smooth anti-aliased edges
    4. Blend back into the host image
    """
    out = img.convert("RGB").copy()
    w, h = out.size

    box_w = rng.randint(max(48, w // 5), max(64, w // 2))
    box_h = rng.randint(max(24, h // 10), max(40, h // 4))
    x0 = rng.randint(8, max(9, w - box_w - 8))
    y0 = rng.randint(8, max(9, h - box_h - 8))
    x1, y1 = x0 + box_w, y0 + box_h

    crop = out.crop((x0, y0, x1, y1))
    # Stage 1: Inpainting background erasure (strips camera sensor high-freq noise)
    erased = crop.filter(ImageFilter.GaussianBlur(radius=3.5))
    erased_arr = np.asarray(erased, dtype=np.float32)
    # Add subtle periodic up-sampling grid typical of diffusion VAE decoders
    yy, xx = np.mgrid[0:box_h, 0:box_w]
    vae_grid = 1.5 * np.sin(2 * np.pi * xx / 4.0) * np.cos(2 * np.pi * yy / 4.0)
    erased_arr = np.clip(erased_arr + vae_grid[..., None], 0, 255).astype(np.uint8)
    erased_patch = Image.fromarray(erased_arr)

    # Stage 2: Render synthetic replacement text/strokes inside the inpainted box
    draw = ImageDraw.Draw(erased_patch)
    sample_texts = ["PAID 980.00", "APPROVED", "2026-11-13", "Verified", "ID: 84920", "Pass #07"]
    text = rng.choice(sample_texts)
    ink_color = (rng.randint(10, 60), rng.randint(10, 60), rng.randint(20, 90))
    tx = rng.randint(4, max(5, box_w // 6))
    ty = rng.randint(4, max(5, box_h // 4))
    draw.text((tx, ty), text, fill=ink_color)

    # Simulate slight stroke jitter / hand-drawn underline
    y_line = min(box_h - 3, ty + 16)
    draw.line([(tx, y_line), (min(box_w - 4, tx + len(text) * 7), y_line)], fill=ink_color, width=2)

    # Slight post-inpainting smoothing
    erased_patch = erased_patch.filter(ImageFilter.SMOOTH_MORE)
    out.paste(erased_patch, (x0, y0))

    # Re-encode via JPEG (web editor export simulation)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(82, 94))
    buf.seek(0)
    return Image.open(buf).convert("RGB"), [x0, y0, x1, y1]


def simulate_full_aigc_image(size: Tuple[int, int], rng: random.Random) -> Image.Image:
    """
    Generate a synthetic image with diffusion-like smooth low-frequency gradients and
    periodic spectral checkerboard artifacts (unlike natural camera sensor noise).
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Smooth gradient background + synthetic geometric forms
    r = 128 + 80 * np.sin(xx / rng.uniform(30.0, 60.0)) + 40 * np.cos(yy / rng.uniform(30.0, 60.0))
    g = 120 + 70 * np.cos((xx + yy) / rng.uniform(40.0, 80.0))
    b = 140 + 60 * np.sin(yy / rng.uniform(25.0, 55.0))

    # Subtle 8x8 / 4x4 transposed-conv periodic artifact
    grid = 2.2 * np.sin(2 * np.pi * xx / 8.0) * np.sin(2 * np.pi * yy / 8.0)
    rgb = np.stack([r + grid, g + grid, b + grid], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def simulate_real_camera_image(size: Tuple[int, int], rng: random.Random) -> Image.Image:
    """
    Generate a synthetic 'camera-like' document/scene photo with realistic sensor PRNU
    Gaussian/Poisson shot noise and sharp natural strokes (for demo smoke testing).
    """
    w, h = size
    np_rng = np.random.default_rng(rng.randint(0, 100000))
    # Vary background complexity from L1 (clean white document) to L4 (textured scene)
    bg_level = rng.choice([242, 235, 200, 160])
    base = np.full((h, w, 3), bg_level, dtype=np.float32)

    if bg_level < 220:
        # Add natural multi-scale scene texture
        tex = np_rng.normal(0, 18.0, size=(h // 4, w // 4, 3)).astype(np.float32)
        tex_img = Image.fromarray(np.clip(tex + 128, 0, 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
        base = 0.7 * base + 0.3 * np.asarray(tex_img, dtype=np.float32)

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # Draw authentic sharp document lines & handwritten-like strokes BEFORE sensor noise
    for row_y in range(28, h - 24, 28):
        draw.line([(16, row_y), (w - 16, row_y)], fill=(180, 195, 215), width=1)
        draw.text((24, row_y - 14), f"Note Line #{row_y // 28}: Authentic Sample", fill=(35, 35, 45))

    # Apply realistic full-frame camera sensor noise (PRNU + shot noise) AFTER content
    arr = np.asarray(img, dtype=np.float32)
    sensor_noise = np_rng.normal(0.0, 3.8, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + sensor_noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_dataset(
    output_dir: Path,
    real_dir: Path | None = None,
    fake_dir: Path | None = None,
    create_demo_data: bool = False,
    num_demo_samples: int = 48,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    for split in ["train", "val"]:
        for cls_dir in ["0_real", "1_fake"]:
            (output_dir / split / cls_dir).mkdir(parents=True, exist_ok=True)

    real_images: List[Path] = []
    if real_dir and real_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            real_images.extend(real_dir.rglob(ext))

    manifest = {"train_real": 0, "train_fake": 0, "val_real": 0, "val_fake": 0, "tamper_boxes": {}}

    if create_demo_data or not real_images:
        total_pairs = num_demo_samples // 2
        for i in range(total_pairs):
            split = "val" if i < int(total_pairs * val_ratio) else "train"
            real_im = simulate_real_camera_image((256, 256), rng)
            real_path = output_dir / split / "0_real" / f"real_{i:04d}.jpg"
            real_im.save(real_path, quality=95)
            manifest[f"{split}_real"] += 1

            # 50% OpenCut-style local text tampering, 50% full AIGC synthesis
            if i % 2 == 0:
                fake_im, box = simulate_opencut_text_tampering(real_im, rng)
                fake_name = f"fake_opencut_{i:04d}.jpg"
                manifest["tamper_boxes"][fake_name] = box
            else:
                fake_im = simulate_full_aigc_image((256, 256), rng)
                fake_name = f"fake_aigc_{i:04d}.jpg"

            fake_path = output_dir / split / "1_fake" / fake_name
            fake_im.save(fake_path, quality=92)
            manifest[f"{split}_fake"] += 1
    else:
        rng.shuffle(real_images)
        for i, rpath in enumerate(real_images):
            split = "val" if i < int(len(real_images) * val_ratio) else "train"
            try:
                im = Image.open(rpath).convert("RGB")
            except Exception:
                continue
            target_real = output_dir / split / "0_real" / f"real_{i:04d}_{rpath.stem}.jpg"
            im.save(target_real, quality=95)
            manifest[f"{split}_real"] += 1

            # Automatically synthesize an OpenCut-like locally tampered pair from this real photo
            fake_im, box = simulate_opencut_text_tampering(im, rng)
            fake_name = f"fake_opencut_{i:04d}_{rpath.stem}.jpg"
            fake_im.save(output_dir / split / "1_fake" / fake_name, quality=92)
            manifest["tamper_boxes"][fake_name] = box
            manifest[f"{split}_fake"] += 1

        # Also copy any user-supplied full AIGC images if provided
        if fake_dir and fake_dir.exists():
            fake_files: List[Path] = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                fake_files.extend(fake_dir.rglob(ext))
            for j, fpath in enumerate(fake_files):
                split = "val" if j < int(len(fake_files) * val_ratio) else "train"
                try:
                    fim = Image.open(fpath).convert("RGB")
                    fim.save(output_dir / split / "1_fake" / f"fake_ext_{j:04d}_{fpath.stem}.jpg", quality=92)
                    manifest[f"{split}_fake"] += 1
                except Exception:
                    continue

    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Real vs AI-Generated/Modified Dataset")
    parser.add_argument("--real_dir", type=str, default=None, help="Directory of real photos")
    parser.add_argument("--fake_dir", type=str, default=None, help="Optional directory of full AIGC photos")
    parser.add_argument("--output_dir", type=str, default="data/dataset", help="Output dataset root directory")
    parser.add_argument(
        "--create_demo_data",
        action="store_true",
        help="Generate synthetic real + OpenCut-tampered + AIGC demo images for pipeline verification",
    )
    parser.add_argument("--num_demo_samples", type=int, default=48, help="Total demo images to synthesize")
    args = parser.parse_args()

    manifest = build_dataset(
        output_dir=Path(args.output_dir),
        real_dir=Path(args.real_dir) if args.real_dir else None,
        fake_dir=Path(args.fake_dir) if args.fake_dir else None,
        create_demo_data=args.create_demo_data,
        num_demo_samples=args.num_demo_samples,
    )
    print("Dataset prepared successfully:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
