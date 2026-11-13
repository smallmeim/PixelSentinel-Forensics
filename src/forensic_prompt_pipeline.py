#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Forensic Prompt Pipeline (Zero-Training Visual + Textual Prompting)
=================================================================================
Inspired by TextSleuth (MIPR 2024), X2-DFD (NeurIPS 2025), and AIGI-Holmes (ICCV 2025).

Instead of training a heavy model on local hardware, this pipeline:
1. Extracts quantitative low-level forensic signals (SRM high-frequency noise vacuum boxes,
   ELA recompression residuals, RDD 90-degree rotation + low-pass debiased scores, and
   background complexity tier L1-L4).
2. Locates the Top-K most suspicious local patches (e.g., where OpenCut erased and replaced
   handwritten text) and crops/highlights them into a Visual Prompt Composite Image.
3. Generates a tailored, data-grounded Forensic Chain-of-Thought Prompt that tells any
   Vision LLM (or Zero-Shot CLIP Forensic Prompt Ensemble) EXACTLY which coordinates and
   micro-physical clues to inspect.

Usage:
    python src/forensic_prompt_pipeline.py --image path/to/suspicious.jpg --output outputs/prompt_pack
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from forensics_toolkit import (
    compute_ela_map,
    compute_patch_complexity_and_debiased_score,
    compute_srm_noise_residual,
    compute_stroke_edge_gradient,
    extract_metadata_forensics,
)


# Zero-Shot CLIP Forensic Prompt Ensemble (contrasting camera physics vs diffusion/inpainting artifacts)
AUTHENTIC_CLIP_PROMPTS = [
    "a genuine smartphone photograph with consistent CMOS sensor grain and natural lighting",
    "authentic handwritten ink on continuous paper fiber texture with natural pen pressure",
    "an unmodified real-world photo with natural high-frequency surface detail and lens physics",
]

FORGERY_CLIP_PROMPTS = [
    "an AI-generated synthetic image with smooth diffusion textures and artificial rendering",
    "a locally inpainted photo where text was erased and replaced leaving a smooth noise vacuum",
    "digital image manipulation with inconsistent edge halos and broken background texture under text",
]


def find_top_suspicious_boxes(
    srm_map: np.ndarray,
    ela_map: np.ndarray,
    anomaly_map: np.ndarray,
    img_size: Tuple[int, int],
    top_k: int = 3,
    patch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Locate the Top-K local bounding boxes where SRM noise disparity, ELA residual,
    and Reference-Debiased anomaly co-occur (typical of OpenCut text replacement boxes).
    """
    w, h = img_size
    ph, pw = anomaly_map.shape

    # Compute local SRM variance map at patch resolution
    srm_patch_mean = np.zeros((ph, pw), dtype=np.float32)
    ela_patch_mean = np.zeros((ph, pw), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            y0, y1 = i * patch_size, min(h, (i + 1) * patch_size)
            x0, x1 = j * patch_size, min(w, (j + 1) * patch_size)
            srm_patch_mean[i, j] = float(np.mean(srm_map[y0:y1, x0:x1]))
            ela_patch_mean[i, j] = float(np.mean(ela_map[y0:y1, x0:x1]))

    # In OpenCut inpainting, SRM noise inside the box deviates strongly from the global median
    global_srm_median = float(np.median(srm_patch_mean))
    srm_deviation = np.abs(srm_patch_mean - global_srm_median) / (global_srm_median + 1e-5)

    norm_ela = ela_patch_mean / (float(np.max(ela_patch_mean)) + 1e-5)
    norm_anom = anomaly_map / (float(np.max(anomaly_map)) + 1e-5)
    norm_srm_dev = srm_deviation / (float(np.max(srm_deviation)) + 1e-5)

    combined_suspicion = 0.45 * norm_anom + 0.35 * norm_srm_dev + 0.20 * norm_ela

    flat_indices = np.argsort(combined_suspicion.ravel())[::-1]
    selected_boxes: List[Dict[str, Any]] = []

    for idx in flat_indices:
        if len(selected_boxes) >= top_k:
            break
        pi, pj = divmod(int(idx), pw)
        # Expand to a 2x2 patch neighborhood for context around the text/tampered region
        x0 = max(0, (pj - 1) * patch_size)
        y0 = max(0, (pi - 1) * patch_size)
        x1 = min(w, (pj + 2) * patch_size)
        y1 = min(h, (pi + 2) * patch_size)

        # Avoid heavily overlapping boxes
        overlap = False
        for existing in selected_boxes:
            ex0, ey0, ex1, ey1 = existing["bbox_xyxy"]
            if abs(x0 - ex0) < patch_size and abs(y0 - ey0) < patch_size:
                overlap = True
                break
        if not overlap:
            selected_boxes.append(
                {
                    "rank": len(selected_boxes) + 1,
                    "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "suspicion_score": round(float(combined_suspicion[pi, pj]), 4),
                    "srm_noise_deviation_ratio": round(float(srm_deviation[pi, pj]), 4),
                    "ela_local_mean": round(float(ela_patch_mean[pi, pj]), 4),
                }
            )
    return selected_boxes


def create_visual_prompt_image(
    img_rgb: Image.Image,
    srm_map: np.ndarray,
    edge_map: np.ndarray,
    suspicious_boxes: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Create a Visual Prompt Composite (Fused Mask Prompt style from TextSleuth):
    - Left: Original image with red bounding boxes & rank labels around Top-K suspicious regions
    - Middle: SRM High-Frequency Noise Residual map (exposing OpenCut noise vacuum boxes)
    - Right: 2x magnified crop of the #1 most suspicious region (RGB + Edge/Stroke gradient)
    """
    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)
    colors = [(255, 50, 50), (255, 165, 0), (0, 220, 255)]

    for idx, box_info in enumerate(suspicious_boxes):
        x0, y0, x1, y1 = box_info["bbox_xyxy"]
        color = colors[idx % len(colors)]
        for offset in range(2):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
        draw.text((x0 + 4, max(0, y0 - 14)), f"#{box_info['rank']} ({box_info['suspicion_score']:.2f})", fill=color)

    w, h = img_rgb.size
    srm_norm = np.uint8(255 * (srm_map / (float(np.max(srm_map)) + 1e-5)))
    srm_vis = Image.fromarray(srm_norm).convert("RGB")

    # Crop #1 suspicious region and magnify to inspect stroke/background continuity
    if suspicious_boxes:
        bx0, by0, bx1, by1 = suspicious_boxes[0]["bbox_xyxy"]
        crop_rgb = img_rgb.crop((bx0, by0, bx1, by1)).resize((w, h // 2), Image.Resampling.NEAREST)
        edge_crop = edge_map[by0:by1, bx0:bx1]
        edge_norm = np.uint8(255 * (edge_crop / (float(np.max(edge_crop)) + 1e-5)))
        crop_edge = Image.fromarray(edge_norm).convert("RGB").resize((w, h - h // 2), Image.Resampling.NEAREST)
    else:
        crop_rgb = img_rgb.resize((w, h // 2))
        crop_edge = srm_vis.resize((w, h - h // 2))

    canvas = Image.new("RGB", (w * 3, h), (15, 20, 35))
    canvas.paste(annotated, (0, 0))
    canvas.paste(srm_vis, (w, 0))
    canvas.paste(crop_rgb, (w * 2, 0))
    canvas.paste(crop_edge, (w * 2, h // 2))
    canvas.save(out_path, quality=95)


def build_structured_forensic_prompt(
    meta: Dict[str, Any],
    debias_stats: Dict[str, Any],
    suspicious_boxes: List[Dict[str, Any]],
) -> str:
    """
    Generate the complete X2-DFD + TextSleuth grounded forensic prompt with injected quantitative signals.
    """
    boxes_desc = "\n".join(
        [
            f"  - 候选框 #{b['rank']}: 像素坐标 {b['bbox_xyxy']}, 综合异常度={b['suspicion_score']}, "
            f"SRM高频噪声偏离率={b['srm_noise_deviation_ratio']}x, ELA局部残差={b['ela_local_mean']}"
            for b in suspicious_boxes
        ]
    )
    risk_flags = ", ".join(meta["metadata_risk_flags"]) if meta["metadata_risk_flags"] else "无明显元数据剥离异常"

    # Dynamically load accumulated Human Expert Forensic Rules from data/human_forensic_rules.json
    rules_path = Path(__file__).resolve().parent.parent / "data" / "human_forensic_rules.json"
    human_rules_section = ""
    if rules_path.exists():
        try:
            rules_list = json.loads(rules_path.read_text(encoding="utf-8"))
            if rules_list:
                formatted_rules = "\n".join(
                    [
                        f"  - **{r['rule_id']} [{r['category']}]**：{r['compiled_prompt_rule']}"
                        for r in rules_list
                    ]
                )
                human_rules_section = f"\n## 二（补充）、人类专家实战沉淀的优先排查法则（Human-in-the-Loop Rules）\n{formatted_rules}\n"
        except Exception:
            pass

    prompt = f"""# 数字图像微观取证与去偏审阅任务（PixelSentinel Forensic Prompt）

请作为数字图像取证专家审阅附带的图片（建议同时附带原图与 `visual_prompt_composite.jpg`）。
为了帮助你注意到普通 AI 容易忽略的微观破绽，底层信号算子已预先提取出以下定量取证线索：

## 一、底层信号定量预检结果（Grounded Forensic Signals）
- **图像分辨率与格式**：`{meta['size'][0]}x{meta['size'][1]}` ({meta['format']})
- **元数据与重编码指纹**：{risk_flags}
- **背景复杂度分层（RDD Debiasing）**：`{debias_stats['complexity_tier']}`（边缘密度 `{debias_stats['mean_edge_density']}`）
- **90°旋转+低通滤波去偏校准分**：`{debias_stats['debiased_calibrated_score']}`（原始重建代理误差 `{debias_stats['raw_reconstruction_proxy']}`）
- **局部块异常峰值倍率**：`{debias_stats['patch_anomaly_max_ratio']}x`
- **算法锁定的 Top-{len(suspicious_boxes)} 高疑局部区域（已在视觉提示图中用红/橙框标出）**：
{boxes_desc}
{human_rules_section}
---

## 二、请你严格按以下 4 个微观视角进行核查（重点关注候选框 #{suspicious_boxes[0]['rank'] if suspicious_boxes else 1}）

1. **【视角 A：排查 OpenCut 等局部 AI 换字/重绘痕迹】**
   请重点放大查看上方列出的高疑坐标框（尤其是框内的手写字、金额、姓名、印章或标签）：
   - **字下纸纹与横格线连续性**：文字正下方的纸张纤维颗粒、横格线或背景阴影是否连续穿过笔画？还是在文字周围出现了轻微的“雾状平滑块”、横格线断裂或错位？
   - **手写墨水物理叠加**：在笔画起收笔和**两笔交叉点**，是否有真实的双倍墨水深浅沉积？还是像 AI 渲染字一样内部颜色完全均匀、边缘呈数码羽化光晕？
2. **【视角 B：核对 SRM 高频传感器噪声断层】**
   对照视觉提示图中间的 SRM 噪声图与右侧 2x 放大图：高疑框内部是否出现了与周围真实相机传感器噪点不一致的“噪声真空区”或规则网格？
3. **【视角 C：排查全图 AIGC 生成伪影】**
   检查全局光影消失点、边缘高频细节（发丝、织物、微小背景文字）是否存在扩散模型的涂抹粘连或无意义乱码笔画。
4. **【视角 D：背景复杂度防误报护栏（Debiasing Guardrail）】**
   ⚠️ 当前图像复杂度分级为 `{debias_stats['complexity_tier']}`。如果该图属于 L1/L2（白墙、白纸文档等简单背景），**严禁仅仅因为背景干净平滑就将其误判为 AI 生成**！必须依据真实的传感器噪点、物理墨迹压痕或局部噪声断层下结论。

---

## 三、请输出结构化取证结论
- **最终判定**：【真实实拍】 / 【全图 AI 生成】 / 【真实底图 + 局部 AI 篡改（如 OpenCut 改字）】
- **伪造置信度**：0% ~ 100%
- **定位到的篡改位置与微观证据链**：逐条说明你在上述 4 个视角下观察到的具体证据。
"""
    return prompt


def generate_forensic_prompt_pack(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    meta = extract_metadata_forensics(img, image_path)
    img_rgb = img.convert("RGB")
    img_gray = np.asarray(img.convert("L"), dtype=np.float32)

    ela_map = compute_ela_map(img_rgb, quality=90)
    srm_map = compute_srm_noise_residual(img_gray)
    edge_map = compute_stroke_edge_gradient(img_gray)
    _, anomaly_map, debias_stats = compute_patch_complexity_and_debiased_score(img_rgb)

    suspicious_boxes = find_top_suspicious_boxes(srm_map, ela_map, anomaly_map, img_rgb.size, top_k=3)

    visual_prompt_path = output_dir / f"{image_path.stem}_visual_prompt.jpg"
    create_visual_prompt_image(img_rgb, srm_map, edge_map, suspicious_boxes, visual_prompt_path)

    prompt_text = build_structured_forensic_prompt(meta, debias_stats, suspicious_boxes)
    prompt_md_path = output_dir / f"{image_path.stem}_forensic_prompt.md"
    prompt_md_path.write_text(prompt_text, encoding="utf-8")

    result = {
        "input_image": str(image_path.resolve()),
        "visual_prompt_image": str(visual_prompt_path.resolve()),
        "generated_prompt_markdown": str(prompt_md_path.resolve()),
        "complexity_tier": debias_stats["complexity_tier"],
        "debiased_calibrated_score": debias_stats["debiased_calibrated_score"],
        "suspicious_boxes": suspicious_boxes,
    }
    (output_dir / f"{image_path.stem}_prompt_pack.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Visual + Textual Forensic Prompt Pack (Zero Local GPU Training)"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image to analyze")
    parser.add_argument("--output", type=str, default="outputs/prompt_pack", help="Output folder")
    args = parser.parse_args()

    res = generate_forensic_prompt_pack(Path(args.image), Path(args.output))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
