#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel WeChat / Alipay Transfer Screenshot Forensics Engine
(中国社交与支付平台：微信支付账单/聊天转账/支付宝回单 PS 与 AI 伪造专项取证)
====================================================================================================
Specifically engineered for Chinese social & payment platform screenshots (微信支付转账账单详情、
微信聊天转账卡片、支付宝电子回单、手机银行转账截图) forged via Photoshop (PS 贴字/改数/涂抹覆盖)
or AI Inpainting (OpenCut / 豆包 / 扩散改字).

Core Physical & Digital Compositor Laws for WeChat / Alipay UI Screenshots:
1. **WeChat Dual-Column UI Grid Law (微信账单双列坐标轴严格对齐定律)**:
   - Genuine WeChat Pay bill details (`账单详情`) enforce a strict two-column layout grid:
     * Left Label Rail (`当前状态`, `转账说明`, `转账时间`, `收款时间`, `支付方式`, `转账单号`)
       shares an exact left X-coordinate (`X_label_rail`, 0px horizontal jitter).
     * Right Value Rail (`对方已收钱`, `2026年11月13日...`, `零钱通`, `1000050001...`)
       shares an exact second-column left X-coordinate (`X_value_rail`, 0px horizontal jitter).
   - Manual PS layer dragging causes **Left Label Drift** or **Right Value Column X-Coordinate Drift**.
2. **Hero Amount Sub-Pixel Baseline Law (微信大额金额同行基线零抖动定律)**:
   - In genuine WeChat amount headers (e.g., `-88,500.00`), the minus/currency sign, integer digits,
     and decimal `.00` sit on an identical pixel baseline. PS pasting of modified digits (`88,500`)
     leaves a `2~6 px` vertical baseline step (`PS_BASELINE_MISALIGNMENT`).
3. **WeChat Sans / System Font Anti-Aliasing & Stroke Weight Law (微信专属字体与抗锯齿斜率一致性)**:
   - Genuine WeChat screenshots render amounts in `WeChat Sans Std` and body rows in uniform `PingFang SC`
     or `Microsoft YaHei`. Splicing digits/timestamps with `SimSun (宋体)`, `Arial`, or rescaled layers
     creates sharp intra-line stroke density CV and edge gradient slope anomalies (`WECHAT_FONT_MISMATCH`).
4. **UI Canvas Erase-Box Purity Law (微信纯白画布涂抹覆盖框残差定律)**:
   - Covering old digits in PS with an eyedropper rectangle (`#FCFCFA` vs `#FFFFFF`) or JPEG re-saving
     leaves a localized luminance/noise step around the tampered text (`PS_ERASE_BOX_RESIDUAL`).

Usage:
    D:\\aigi\\venv\\Scripts\\python.exe src/screenshot_receipt_forensics.py --image examples/forged_transfer_receipt.jpg --output outputs/eval_transfer_receipt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_cjk_font(size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
        ]
        if bold
        else [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\Deng.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _segment_text_lines_and_glyphs(
    gray: np.ndarray, rgb: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """
    Segment horizontal text rows, glyph/word clusters, and WeChat two-column (Label / Value) structure.
    """
    h, w = gray.shape
    bg_lum = float(np.percentile(gray, 85))
    ink_mask = (bg_lum - gray) > 26.0

    # Ignore outer 4% border and top status/header bar or decorative icon circles
    pad_x, pad_y = int(w * 0.04), int(h * 0.05)
    ink_mask[:pad_y, :] = False
    ink_mask[h - pad_y :, :] = False
    ink_mask[:, :pad_x] = False
    ink_mask[:, w - pad_x :] = False

    # Also ignore thin 1px horizontal divider lines (where >55% of width is ink on 1-2 rows)
    row_ink_raw = np.sum(ink_mask, axis=1)
    divider_rows = row_ink_raw > int(w * 0.50)
    ink_mask[divider_rows, :] = False

    row_ink = np.sum(ink_mask, axis=1)
    active_rows = row_ink > max(4, int(w * 0.008))

    lines: List[Tuple[int, int]] = []
    in_line = False
    y_start = 0
    for y in range(h):
        if active_rows[y] and not in_line:
            in_line = True
            y_start = y
        elif not active_rows[y] and in_line:
            in_line = False
            # Support both standard WeChat table rows (12~26px) and large WeChat amount rows (up to 9% height)
            if 10 <= (y - y_start) <= int(h * 0.095):
                lines.append((y_start, y))

    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = np.hypot(gx, gy)

    line_features: List[Dict[str, Any]] = []
    for idx, (y0, y1) in enumerate(lines):
        line_mask = ink_mask[y0:y1, :]
        col_ink = np.sum(line_mask, axis=0) > 0

        # Group contiguous or closely-spaced columns into glyph/token clusters (gap <= 4px)
        glyphs: List[Dict[str, Any]] = []
        in_g = False
        x0 = 0
        gap_count = 0
        for x in range(w):
            if col_ink[x]:
                if not in_g:
                    in_g = True
                    x0 = x
                gap_count = 0
            else:
                if in_g:
                    gap_count += 1
                    if gap_count > 4:
                        x1 = x - gap_count + 1
                        if (x1 - x0) >= 5:
                            sub_m = line_mask[:, x0:x1]
                            ys_on = np.where(np.any(sub_m, axis=1))[0]
                            if ys_on.size > 0:
                                gy0 = int(y0 + ys_on[0])
                                gy1 = int(y0 + ys_on[-1] + 1)
                                sub_gray = gray[gy0:gy1, x0:x1]
                                sub_grad = grad[gy0:gy1, x0:x1]
                                sub_ink = sub_m[ys_on[0] : ys_on[-1] + 1, :]
                                contrast = float(bg_lum - np.percentile(sub_gray, 10)) + 1e-4
                                edge_sharpness = float(np.percentile(sub_grad, 90)) / contrast
                                stroke_density = float(np.mean(sub_ink))
                                glyphs.append(
                                    {
                                        "bbox": [int(x0), gy0, int(x1), gy1],
                                        "top_y": gy0,
                                        "bottom_y": gy1,
                                        "center_y": 0.5 * (gy0 + gy1),
                                        "height": gy1 - gy0,
                                        "edge_sharpness": round(edge_sharpness, 4),
                                        "stroke_density": round(stroke_density, 4),
                                    }
                                )
                        in_g = False
                        gap_count = 0

        if not glyphs:
            continue

        # Filter out circular icon rows (single square cluster centered horizontally near top)
        if len(glyphs) == 1:
            g0 = glyphs[0]
            gw, gh = g0["bbox"][2] - g0["bbox"][0], g0["bbox"][3] - g0["bbox"][1]
            if 0.75 <= (gw / float(max(1, gh))) <= 1.35 and gh >= 36:
                continue

        # Detect WeChat Two-Column structure: Left Label vs Right Value column
        # If row starts on the left (< 25% width) and has a wide horizontal gap (>= 18px) to a value column
        value_col_left_x: Optional[int] = None
        value_col_bbox: Optional[List[int]] = None
        if glyphs[0]["bbox"][0] < w * 0.25 and len(glyphs) >= 2:
            for g_i in range(1, len(glyphs)):
                prev_r = glyphs[g_i - 1]["bbox"][2]
                curr_l = glyphs[g_i]["bbox"][0]
                if (curr_l - prev_r) >= 18 and (w * 0.20 <= curr_l <= w * 0.52):
                    value_col_left_x = int(curr_l)
                    value_col_bbox = [int(curr_l), int(y0), int(glyphs[-1]["bbox"][2]), int(y1)]
                    break

        # Baseline jitter analysis (focus on main character clusters of height >= 10px, ignoring tiny commas/dots)
        tall_glyphs = [g for g in glyphs if g["height"] >= max(10, int((y1 - y0) * 0.50))]
        ref_glyphs = tall_glyphs if len(tall_glyphs) >= 2 else glyphs

        bottoms = np.array([g["bottom_y"] for g in ref_glyphs], dtype=np.float32)
        tops = np.array([g["top_y"] for g in ref_glyphs], dtype=np.float32)
        centers = np.array([g["center_y"] for g in ref_glyphs], dtype=np.float32)
        sharp_vals = np.array([g["edge_sharpness"] for g in ref_glyphs], dtype=np.float32)
        dens_vals = np.array([g["stroke_density"] for g in ref_glyphs], dtype=np.float32)

        # Reference top/bottom from the anchor glyphs (first & last glyph of the line, e.g. currency sign & .00)
        anchor_Pool = [ref_glyphs[0], ref_glyphs[-1]] if len(ref_glyphs) >= 3 else [ref_glyphs[0]]
        ref_top = float(np.median([g["top_y"] for g in anchor_Pool]))
        ref_bot = float(np.median([g["bottom_y"] for g in anchor_Pool]))
        med_sharp = float(np.median(sharp_vals))
        med_dens = float(np.median(dens_vals))

        # True PS vertical baseline shift translates BOTH top_y and bottom_y in the same direction (>= 2.5px)
        co_shifts: List[float] = []
        for g in ref_glyphs:
            dt = float(g["top_y"]) - ref_top
            db = float(g["bottom_y"]) - ref_bot
            if dt * db > 0 and min(abs(dt), abs(db)) >= 2.5:
                co_shifts.append(0.5 * (abs(dt) + abs(db)))
            else:
                co_shifts.append(0.0)
        max_baseline_shift = float(max(co_shifts)) if len(co_shifts) >= 2 else 0.0
        font_sharpness_cv = float(np.std(sharp_vals) / (med_sharp + 1e-4)) if sharp_vals.size >= 2 else 0.0
        stroke_density_cv = float(np.std(dens_vals) / (med_dens + 1e-4)) if dens_vals.size >= 2 else 0.0

        # Measure local background purity & erase-box step around this line
        pad_box_y0, pad_box_y1 = max(0, y0 - 5), min(h, y1 + 5)
        pad_box_x0 = max(0, glyphs[0]["bbox"][0] - 8)
        pad_box_x1 = min(w, glyphs[-1]["bbox"][2] + 8)
        bg_patch = gray[pad_box_y0:pad_box_y1, pad_box_x0:pad_box_x1]
        bg_only = bg_patch[np.abs(bg_patch - bg_lum) < 18.0]
        bg_noise_std = float(np.std(bg_only)) if bg_only.size > 10 else 0.0

        line_features.append(
            {
                "line_index": len(line_features),
                "y_range": [int(y0), int(y1)],
                "left_x": int(glyphs[0]["bbox"][0]),
                "right_x": int(glyphs[-1]["bbox"][2]),
                "value_col_left_x": value_col_left_x,
                "value_col_bbox": value_col_bbox,
                "glyph_count": len(glyphs),
                "max_baseline_shift_px": round(max_baseline_shift, 2),
                "median_edge_sharpness": round(med_sharp, 4),
                "intra_line_font_sharpness_cv": round(font_sharpness_cv, 4),
                "intra_line_stroke_density_cv": round(stroke_density_cv, 4),
                "local_bg_noise_std": round(bg_noise_std, 4),
                "glyphs": glyphs,
                "ref_glyphs": ref_glyphs,
            }
        )

    return lines, line_features


def analyze_transfer_receipt(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Perform full PS + AI forensic inspection on a WeChat Pay / Alipay / Bank transfer screenshot.
    Detects:
      1. Hero amount / intra-line character baseline misalignment (微信金额/同行文字 PS 贴字基线上下错位)
      2. WeChat Left Label Rail & Right Value Rail coordinate drift (微信双列网格左列/右列 X 坐标不齐)
      3. WeChat Sans vs spliced font anti-aliasing & stroke weight mismatch (微信专属字体与抗锯齿斜率突变)
      4. Local PS erase-box background residual on pure UI canvas (PS 涂白覆盖框底色残差)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    h, w = gray.shape

    # Check if the image has a flat digital UI/document canvas (at least 42% of pixels within ±8 of background mode)
    bg_lum = float(np.percentile(gray, 85))
    flat_bg_fraction = float(np.mean(np.abs(gray - bg_lum) <= 8.0))
    is_ui_canvas = flat_bg_fraction >= 0.42

    _, line_feats = _segment_text_lines_and_glyphs(gray, arr) if is_ui_canvas else ([], [])

    anomalies: List[Dict[str, Any]] = []
    flags: List[str] = []
    receipt_score = 0.0

    # 1. Check Intra-Line Baseline Shifts (微信金额或同行文字 PS 贴字上下错位)
    for lf in line_feats:
        if lf["max_baseline_shift_px"] >= 2.5 and len(lf["ref_glyphs"]) >= 2:
            r_glyphs = lf["ref_glyphs"]
            anchor_pool = [r_glyphs[0], r_glyphs[-1]] if len(r_glyphs) >= 3 else [r_glyphs[0]]
            ref_top = float(np.median([g["top_y"] for g in anchor_pool]))
            ref_bot = float(np.median([g["bottom_y"] for g in anchor_pool]))
            shifted_boxes: List[List[int]] = []
            shifts: List[float] = []
            for g in r_glyphs:
                dt = float(g["top_y"]) - ref_top
                db = float(g["bottom_y"]) - ref_bot
                if dt * db > 0 and min(abs(dt), abs(db)) >= 2.5:
                    shifted_boxes.append(g["bbox"])
                    shifts.append(0.5 * (dt + db))
            if shifted_boxes:
                bx0 = min(b[0] for b in shifted_boxes)
                by0 = min(b[1] for b in shifted_boxes)
                bx1 = max(b[2] for b in shifted_boxes)
                by1 = max(b[3] for b in shifted_boxes)
                shift_px = round(float(np.mean(shifts)), 1)
                anomalies.append(
                    {
                        "type": "BASELINE_MISALIGNMENT",
                        "bbox": [bx0, by0, bx1, by1],
                        "label": f"PS贴字基线错位 {shift_px:+.1f}px",
                        "severity": abs(shift_px) * 4.0,
                    }
                )
            receipt_score += 0.35
            flags.append(
                f"WECHAT_BASELINE_MISALIGNMENT: 第 {lf['line_index']+1} 行 (Y={lf['y_range']}) 内部字符/金额存在 "
                f"{lf['max_baseline_shift_px']}px 的上下基线错位，系 PS 手工贴字改数未对齐微信原生基线所致 (+0.35)"
            )

    # 2A. Check WeChat Left Label Column X-Coordinate Alignment Rail (微信账单左列标签轴偏移)
    left_xs = [(lf["line_index"], lf["left_x"], lf["y_range"]) for lf in line_feats if lf["left_x"] < w * 0.28]
    left_rail_x: Optional[float] = None
    if len(left_xs) >= 3:
        xs_arr = np.array([x for _, x, _ in left_xs], dtype=np.float32)
        left_rail_x = float(np.median(xs_arr))
        for l_idx, lx, yr in left_xs:
            drift = lx - left_rail_x
            if 2.0 <= abs(drift) <= 22.0:
                lf_match = next(x for x in line_feats if x["line_index"] == l_idx)
                anomalies.append(
                    {
                        "type": "LEFT_RAIL_X_DRIFT",
                        "bbox": [lx, yr[0], lf_match["right_x"], yr[1]],
                        "label": f"左列标签轴偏移 {drift:+.1f}px",
                        "severity": abs(drift) * 1.3,
                    }
                )
                receipt_score += 0.30
                flags.append(
                    f"WECHAT_LEFT_RAIL_DRIFT: 第 {l_idx+1} 行左侧标签起点 X={lx}px 偏离微信账单左列标准轴线 "
                    f"(X={left_rail_x:.0f}px) 达 {drift:+.1f}px，暴露 PS 图层移动痕迹 (+0.30)"
                )

    # 2B. Check WeChat Right Value Column X-Coordinate Alignment Rail (微信账单右列数值轴偏移)
    val_xs = [
        (lf["line_index"], lf["value_col_left_x"], lf["value_col_bbox"])
        for lf in line_feats
        if lf["value_col_left_x"] is not None and lf["value_col_bbox"] is not None
    ]
    value_rail_x: Optional[float] = None
    if len(val_xs) >= 3:
        v_arr = np.array([vx for _, vx, _ in val_xs], dtype=np.float32)
        value_rail_x = float(np.median(v_arr))
        for l_idx, vx, vbbox in val_xs:
            drift = float(vx) - value_rail_x
            if 2.0 <= abs(drift) <= 24.0:
                anomalies.append(
                    {
                        "type": "RIGHT_VALUE_X_DRIFT",
                        "bbox": vbbox,
                        "label": f"右列数值轴偏移 {drift:+.1f}px",
                        "severity": abs(drift) * 1.4,
                    }
                )
                receipt_score += 0.30
                flags.append(
                    f"WECHAT_VALUE_RAIL_DRIFT: 第 {l_idx+1} 行右侧字段起点 X={vx}px 偏离微信账单右列标准对齐轴 "
                    f"(X={value_rail_x:.0f}px) 达 {drift:+.1f}px，系篡改时间/单号/状态时坐标未对齐 (+0.30)"
                )

    # 3. Check WeChat Font Anti-Aliasing & Stroke Weight Consistency (微信专属字体 WeChat Sans 与系统字体拼贴冲突)
    if len(line_feats) >= 3:
        all_sharp = np.array([lf["median_edge_sharpness"] for lf in line_feats], dtype=np.float32)
        global_med_sharp = float(np.median(all_sharp))
        global_mad_sharp = float(np.median(np.abs(all_sharp - global_med_sharp))) + 0.015

        for lf in line_feats:
            z_sharp = abs(lf["median_edge_sharpness"] - global_med_sharp) / (1.4826 * global_mad_sharp)
            cv_s = lf["intra_line_font_sharpness_cv"]
            cv_d = lf["intra_line_stroke_density_cv"]
            if z_sharp >= 2.8 or cv_s >= 0.20 or cv_d >= 0.24:
                anomalies.append(
                    {
                        "type": "FONT_AA_MISMATCH",
                        "bbox": [lf["left_x"], lf["y_range"][0], lf["right_x"], lf["y_range"][1]],
                        "label": f"字体/抗锯齿冲突 (CV={max(cv_s, cv_d):.2f})",
                        "severity": float(max(z_sharp, cv_s * 12.0, cv_d * 10.0)),
                    }
                )
                receipt_score += 0.30
                flags.append(
                    f"WECHAT_FONT_RENDERING_MISMATCH: 第 {lf['line_index']+1} 行字体边缘抗锯齿斜率 "
                    f"(Sharpness={lf['median_edge_sharpness']}, 笔画/锐度变异系数 CV={max(cv_s, cv_d):.3f}) "
                    f"与微信原生 UI 字体 (WeChat Sans / 系统默认字体) 不一致，存在跨字体拼贴或 AI 合成文字 (+0.30)"
                )

    # 4. Check Local UI Background Purity Violation (PS 涂白覆盖框底色残差)
    if len(line_feats) >= 3:
        bg_stds = np.array([lf["local_bg_noise_std"] for lf in line_feats], dtype=np.float32)
        med_bg = float(np.median(bg_stds))
        for lf in line_feats:
            if lf["local_bg_noise_std"] > max(0.75, med_bg * 2.4):
                anomalies.append(
                    {
                        "type": "BACKGROUND_PASTE_BOX",
                        "bbox": [lf["left_x"] - 6, lf["y_range"][0] - 4, lf["right_x"] + 6, lf["y_range"][1] + 4],
                        "label": f"PS涂白覆盖残差 (σ={lf['local_bg_noise_std']:.2f})",
                        "severity": lf["local_bg_noise_std"] * 4.0,
                    }
                )
                receipt_score += 0.25
                flags.append(
                    f"WECHAT_CANVAS_ERASE_RESIDUAL: 第 {lf['line_index']+1} 行文字底色区域检出非零涂抹/重压缩残差 "
                    f"(σ={lf['local_bg_noise_std']} vs 微信原生纯白画布 σ={med_bg:.2f})，锁定 PS 矩形涂白改字框 (+0.25)"
                )

    out_vis = output_dir / f"{image_path.stem}_receipt_forensic_map.jpg"
    _save_receipt_forensic_visual(im, line_feats, anomalies, left_rail_x, value_rail_x, out_vis)

    return {
        "image": image_path.name,
        "receipt_forgery_score": round(min(0.99, receipt_score), 2),
        "is_forged_receipt": receipt_score >= 0.35,
        "detected_line_count": len(line_feats),
        "left_label_rail_x": left_rail_x,
        "right_value_rail_x": value_rail_x,
        "anomaly_boxes": anomalies,
        "receipt_forensic_flags": flags,
        "receipt_visual_path": str(out_vis.resolve()),
    }


def _save_receipt_forensic_visual(
    im: Image.Image,
    line_feats: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    left_rail_x: Optional[float],
    value_rail_x: Optional[float],
    out_path: Path,
) -> None:
    """
    Render a 2-panel WeChat/Alipay Forensic Diagnostic Map:
    - Left: Original WeChat transfer screenshot overlaid with Left Label Rail (cyan),
            Right Value Rail (sky-blue), row baselines (green), and red/orange anomaly boxes.
    - Right: 3x magnified pixel-level inspection of the top forged regions with baseline & rail rulers.
    """
    w, h = im.size
    ruler_im = im.copy()
    rdraw = ImageDraw.Draw(ruler_im)
    font_sm = _load_cjk_font(13, bold=False)
    font_bd = _load_cjk_font(14, bold=True)
    font_title = _load_cjk_font(16, bold=True)

    # Draw WeChat Left Label Rail on ruler_im
    if left_rail_x is not None:
        rx = int(round(left_rail_x))
        rdraw.line([(rx, int(h * 0.36)), (rx, int(h * 0.90))], fill=(0, 210, 180), width=2)

    # Draw WeChat Right Value Rail on ruler_im
    if value_rail_x is not None:
        vx = int(round(value_rail_x))
        rdraw.line([(vx, int(h * 0.36)), (vx, int(h * 0.90))], fill=(30, 144, 255), width=2)

    # Draw horizontal baseline reference lines on each detected text row on ruler_im
    for lf in line_feats:
        r_glyphs = lf["ref_glyphs"]
        anchor_pool = [r_glyphs[0], r_glyphs[-1]] if len(r_glyphs) >= 3 else [r_glyphs[0]]
        med_bot = int(round(float(np.median([g["bottom_y"] for g in anchor_pool]))))
        rdraw.line([(lf["left_x"], med_bot), (lf["right_x"], med_bot)], fill=(6, 214, 160), width=1)

    # Draw red/orange boxes on ruler_im (without text labels so 3x crops stay clean)
    for a in anomalies:
        bx0, by0, bx1, by1 = a["bbox"]
        is_coord = a["type"] in ("BASELINE_MISALIGNMENT", "LEFT_RAIL_X_DRIFT", "RIGHT_VALUE_X_DRIFT")
        color = (235, 30, 70) if is_coord else (230, 125, 0)
        rdraw.rectangle([bx0 - 3, by0 - 3, bx1 + 3, by1 + 3], outline=color, width=2)

    annotated = ruler_im.copy()
    draw = ImageDraw.Draw(annotated)
    if left_rail_x is not None:
        rx = int(round(left_rail_x))
        draw.text((rx + 4, int(h * 0.36) - 18), f"左列基准轴 X={rx}px", fill=(0, 150, 130), font=font_sm)
    if value_rail_x is not None:
        vx = int(round(value_rail_x))
        draw.text((vx + 4, int(h * 0.36) - 18), f"右列数值轴 X={vx}px", fill=(20, 110, 220), font=font_sm)

    used_label_ys: List[int] = []
    for a in anomalies:
        bx0, by0, bx1, by1 = a["bbox"]
        is_coord = a["type"] in ("BASELINE_MISALIGNMENT", "LEFT_RAIL_X_DRIFT", "RIGHT_VALUE_X_DRIFT")
        color = (235, 30, 70) if is_coord else (230, 125, 0)
        lbl_y = max(4, by0 - 19)
        if any(abs(lbl_y - uy) < 15 for uy in used_label_ys):
            lbl_y = min(h - 18, by1 + 4)
        used_label_ys.append(lbl_y)
        draw.text((bx0, lbl_y), a["label"], fill=color, font=font_bd)

    # Create right-side zoom panel showing top 3 anomaly crops from clean ruler_im
    panel_w = max(500, int(w * 0.92))
    canvas = Image.new("RGB", (w + panel_w + 20, h), (12, 18, 32))
    canvas.paste(annotated, (0, 0))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text(
        (w + 18, 16),
        "微信/社交平台转账凭证 PS 与 AI 像素级取证放大镜 (3x Zoom)",
        fill=(0, 245, 212),
        font=font_title,
    )

    top_anoms = sorted(anomalies, key=lambda x: x["severity"], reverse=True)[:3]
    y_cursor = 54
    for idx, a in enumerate(top_anoms):
        bx0, by0, bx1, by1 = a["bbox"]
        x0 = max(0, bx0 - 26)
        x1 = min(w, bx1 + 26)
        y0 = max(0, by0 - 16)
        y1 = min(h, by1 + 16)
        cw = max(1, x1 - x0)
        ch = max(1, y1 - y0)
        scale = min((panel_w - 32) / float(cw), 210.0 / float(ch))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))

        crop = ruler_im.crop((x0, y0, x1, y1)).resize((nw, nh), Image.Resampling.NEAREST)
        cdraw.text(
            (w + 18, y_cursor),
            f"证据 #{idx+1} [{a['type']}] {a['label']}",
            fill=(255, 105, 125),
            font=font_bd,
        )
        canvas.paste(crop, (w + 18, y_cursor + 24))
        cdraw.rectangle([w + 18, y_cursor + 24, w + 18 + nw, y_cursor + 24 + nh], outline=(56, 189, 248), width=1)
        y_cursor += nh + 52

    canvas.save(out_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WeChat/Alipay transfer screenshot for PS/AI forgery")
    parser.add_argument("--image", type=str, required=True, help="Input screenshot path")
    parser.add_argument("--output", type=str, default="outputs/eval_receipt", help="Output directory")
    args = parser.parse_args()

    res = analyze_transfer_receipt(Path(args.image), Path(args.output))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
