#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Web Platform Server (AI 图像真伪与局部篡改取证网站后端)
=====================================================================
A zero-dependency (stdlib HTTP + PIL/NumPy) full-stack web application that integrates:
1. Real-time image upload & multi-modal signal forensics (`evaluate_sample.py`)
2. Underlying physical & generative framework inspection (`deep_framework_forensics.py`)
3. Pre-loaded 4-level benchmark gallery (MJ5 Vendor, Lab Note, Sticky Note, Room Corner)
4. Interactive Human-in-the-Loop Forensic Rule submission (`human_ai_feedback_loop.py`)

Usage:
    D:\\aigi\\venv\\Scripts\\python.exe src/web_app.py --port 8080
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from deep_framework_forensics import run_deep_framework_inspection
from evaluate_sample import evaluate_image_end_to_end
from human_ai_feedback_loop import add_human_observation, load_rules
import sys
from local_edit_detector import analyze_local_ai_edit
from opencut_forensics import analyze_opencut_tampering
from screenshot_receipt_forensics import analyze_transfer_receipt
from universal_forensic_algorithm import run_universal_forensic_algorithm
from vlm_api_client import (
    call_third_party_vlm_forensics,
    get_public_config,
    load_api_config,
    save_api_config,
)

ROOT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent
)
STATIC_DIR = ROOT_DIR / "src" / "static"
EXAMPLES_DIR = ROOT_DIR / "examples"
OUTPUTS_DIR = ROOT_DIR / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "web_uploads"


PRESET_CASES = [
    {
        "id": "authentic_real",
        "title": "对照组 0：100% 真实相机原图 (未改真图基准)",
        "subtitle": "真实实拍对照：具备真实镜头 Bayer 色散非对称度 (0.0425) 且无局部差分异常 (Z=4.17σ)",
        "image_rel": "/files/examples/authentic_camera_real.png",
        "file_path": EXAMPLES_DIR / "authentic_camera_real.png",
        "eval_dir": OUTPUTS_DIR / "eval_authentic_real",
        "human_note": "真图防误判基准：尽管天空与树林交界存在自然对比度，但跨通道 Bayer 光学色散非对称度高达 0.0425，无 VAE 8x8 步长周期，综合判定为真实原图。",
    },
    {
        "id": "mj5",
        "title": "等级 1：集市老人肖像 (Midjourney v5)",
        "subtitle": "全图 AI 生成：宏观高频雕刻纹路与背景传感器底噪真空",
        "image_rel": "/files/examples/mj5_market_vendor.png",
        "file_path": EXAMPLES_DIR / "mj5_market_vendor.png",
        "eval_dir": OUTPUTS_DIR / "eval_mj5",
        "human_note": "HUMAN-RULE-001：肉眼可见人物皱纹与草帽编织纹呈现过度雕刻的油画涂抹感，背景货架缺乏真实传感器噪点。",
    },
    {
        "id": "lab_note",
        "title": "等级 2：教室课桌手写实验单 (文档/手写伪造)",
        "subtitle": "全图 AI 生成：课桌木纹逼真，但手写字呈‘电脑字库印刷体化’",
        "image_rel": "/files/examples/handwritten_lab_note.jpg",
        "file_path": EXAMPLES_DIR / "handwritten_lab_note.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_lab_note",
        "human_note": "HUMAN-RULE-002：桌面木纹很真，但手写字母数字（OK、3、1）同字同形、等宽等高，且右上角出现 Stoctus 拼写幻觉。",
    },
    {
        "id": "sticky_note",
        "title": "等级 3：脏键盘与涂改便利贴 (高拟真局部涂改)",
        "subtitle": "高拟真生成：指纹与饼干碎屑极真，靠摆拍构图与底层框架破局",
        "image_rel": "/files/examples/sticky_note_memo.jpg",
        "file_path": EXAMPLES_DIR / "sticky_note_memo.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_sticky_note",
        "human_note": "HUMAN-RULE-003：微距斜切+单反虚化不符合拍便利贴的真实意图；算法锁定 850 涂改处存在 9.18x 局部异常与 6 条平行悬空划痕。",
    },
    {
        "id": "room_corner",
        "title": "等级 4：无文字房间角落插座 (全图零文字高隐蔽性)",
        "subtitle": "高隐蔽生成：无文字可查，靠一维电线拓扑断裂 + Bayer/VAE 铁证识别",
        "image_rel": "/files/examples/mundane_room_corner.jpg",
        "file_path": EXAMPLES_DIR / "mundane_room_corner.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_room_corner",
        "human_note": "HUMAN-RULE-005：两个充电头走出的两根线半路合并消失（另一个充电头的线不见了），且绕椅腿自成死循环；底层 Bayer 色散非对称度仅 0.0061。",
    },
    {
        "id": "washbasin_mirror",
        "title": "等级 5：宿舍洗手台镜面随手拍 (无字无电线终极难度)",
        "subtitle": "三维自遮挡与镜面几何：放大见折叠眼镜腿与镜框重合穿模，镜中杯柄消失且牙刷朝向反转",
        "image_rel": "/files/examples/washbasin_mirror_snapshot.jpg",
        "file_path": EXAMPLES_DIR / "washbasin_mirror_snapshot.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_washbasin_mirror",
        "human_note": "HUMAN-RULE-007：梳子阴影极真、无法靠生活常识判断；但放大后可见折叠眼镜架的镜腿与镜框互相重合粘连，且镜子反射中右侧杯柄消失、左下角眼镜反射位置违背几何光学。",
    },
    {
        "id": "opencut_edit",
        "title": "专项 A：OpenCut 同笔迹无痕改字 (FAIL → PASS)",
        "subtitle": "OpenCut 双阶段擦除与同风格扩散改字：同页 59 个字块横向互校锁定 [806, 326, 894, 374] (Z=44.55σ)",
        "image_rel": "/files/examples/opencut_modified_note.jpg",
        "file_path": EXAMPLES_DIR / "opencut_modified_note.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_opencut_edit",
        "human_note": "OpenCut 无痕改字破局：肉眼看 PASS 与上下行 OK 具有完全相同的深蓝圆珠笔墨色与 2px 手写笔锋；但通过『同页字块横向互校 (Z=44.55σ) + 字下横格线 44% 衰减断层 + 4x4/8x8 潜空间谐波』精准锁定篡改框。",
    },
    {
        "id": "real_inpaint",
        "title": "专项 B：100% 真实风景照局部 AI 抹除重绘",
        "subtitle": "真实照片 + 局部 AI 改图：97% 区域具真实 Bayer 色散，精准揪出河面重绘框 [192, 304, 288, 368]",
        "image_rel": "/files/examples/real_photo_ai_inpaint.jpg",
        "file_path": EXAMPLES_DIR / "real_photo_ai_inpaint.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_real_inpaint",
        "human_note": "真图改图最难防：全图 97% 保留真实相机传感器特征（全局易漏检），但河面局部抹除石头处暴露 Z=28.11σ 的中心-周边底噪跳变与解码器网格。",
    },
    {
        "id": "wechat_receipt",
        "title": "专项 C：微信支付转账账单详情 PS/AI 改数取证",
        "subtitle": "社交平台转账凭证伪造：检出 88500 贴字基线错位 (+4.5px)、双列对齐轴偏移 (+7px/+10px) 与字体冲突",
        "image_rel": "/files/examples/forged_transfer_receipt.jpg",
        "file_path": EXAMPLES_DIR / "forged_transfer_receipt.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_transfer_receipt",
        "human_note": "微信账单防诈核心：原生微信 UI 遵循严格的同行零基线抖动与左右双列对齐轴定律。本例精准抓出 88500 贴字下沉 +4.5px、收款时间右列偏移 +10px 及宋体/WeChat Sans 抗锯齿冲突。",
    },
]


def build_case_payload(image_path: Path, eval_dir: Path, case_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    stem = image_path.stem
    summary_file = eval_dir / f"{stem}_evaluation_summary.json"
    if not summary_file.exists():
        eval_summary = evaluate_image_end_to_end(image_path, eval_dir)
    else:
        eval_summary = json.loads(summary_file.read_text(encoding="utf-8"))

    local_edit_map_file = eval_dir / f"{stem}_local_edit_map.jpg"
    local_edit_json_file = eval_dir / f"{stem}_local_edit_stats.json"
    if not local_edit_map_file.exists() or not local_edit_json_file.exists():
        local_edit_stats = analyze_local_ai_edit(image_path, eval_dir)
        local_edit_json_file.write_text(json.dumps(local_edit_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        local_edit_stats = json.loads(local_edit_json_file.read_text(encoding="utf-8"))

    receipt_map_file = eval_dir / f"{stem}_receipt_forensic_map.jpg"
    receipt_json_file = eval_dir / f"{stem}_receipt_forensic_stats.json"
    if not receipt_map_file.exists() or not receipt_json_file.exists():
        receipt_stats = analyze_transfer_receipt(image_path, eval_dir)
        receipt_json_file.write_text(json.dumps(receipt_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        receipt_stats = json.loads(receipt_json_file.read_text(encoding="utf-8"))

    opencut_map_file = eval_dir / f"{stem}_opencut_forensic_map.jpg"
    opencut_json_file = eval_dir / f"{stem}_opencut_forensic_stats.json"
    if not opencut_map_file.exists() or not opencut_json_file.exists():
        opencut_stats = analyze_opencut_tampering(image_path, eval_dir)
        opencut_json_file.write_text(json.dumps(opencut_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        opencut_stats = json.loads(opencut_json_file.read_text(encoding="utf-8"))

    deep_res = run_deep_framework_inspection(image_path)
    univ_res = run_universal_forensic_algorithm(image_path)
    prompt_md_file = eval_dir / f"{stem}_forensic_prompt.md"
    prompt_text = prompt_md_file.read_text(encoding="utf-8") if prompt_md_file.exists() else ""

    rel_dir = eval_dir.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    img_rel = image_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()

    # Combine universal mathematical algorithm, deep framework confidence, signal prob, local edit, receipt & OpenCut forensics
    sig_prob = float(eval_summary.get("ai_forgery_probability", 0.5))
    fw_prob = float(deep_res.get("framework_forgery_confidence", 0.5))
    univ_prob = float(univ_res.get("universal_algo_forgery_score", 0.5))
    is_local_edit = bool(local_edit_stats.get("is_localized_ai_edit", False))
    local_z = float(local_edit_stats.get("max_patch_zscore", 0.0))
    is_forged_receipt = bool(receipt_stats.get("is_forged_receipt", False))
    receipt_prob = float(receipt_stats.get("receipt_forgery_score", 0.0))
    is_opencut = bool(opencut_stats.get("is_opencut_tampered", False))
    opencut_z = float(opencut_stats.get("max_peer_zscore", 0.0))

    if is_forged_receipt and receipt_prob >= 0.60:
        combined_prob = round(max(receipt_prob, 0.96), 2)
        verdict_str = "微信/社交平台转账凭证 PS 或 AI 篡改伪造 (基线错位/坐标轴偏移/字体冲突)"
    elif is_opencut and opencut_z >= 8.0:
        combined_prob = 0.99
        oc_bbox = opencut_stats.get("primary_opencut_token", {}).get("bbox", [0, 0, 0, 0])
        verdict_str = f"OpenCut 局部无痕换字 / 同笔迹扩散篡改 (锁定字框 {oc_bbox}, 同页互校 Z={opencut_z}σ)"
    elif is_local_edit and local_z >= 8.0:
        combined_prob = round(min(0.99, max(0.92, 0.4 * univ_prob + 0.3 * fw_prob + 0.3 * sig_prob + 0.25)), 2)
        bbox = local_edit_stats.get("primary_edit_bbox", [0, 0, 0, 0])
        verdict_str = f"局部 AI 改图 / 涂抹重绘 (锁定篡改框 {bbox})"
    elif univ_prob >= 0.50 or fw_prob >= 0.45:
        combined_prob = round(min(0.99, max(univ_prob, 0.50 * univ_prob + 0.30 * fw_prob + 0.20 * sig_prob)), 2)
        verdict_str = "全图 AI 生成或伪造 (UMSF 通用数理算法裁定)"
    else:
        combined_prob = round(0.50 * univ_prob + 0.30 * fw_prob + 0.20 * sig_prob, 2)
        verdict_str = "真实未改原图 (LIKELY_AUTHENTIC)"

    return {
        "id": case_meta["id"] if case_meta else stem,
        "title": case_meta["title"] if case_meta else f"自定义上传检测：{image_path.name}",
        "subtitle": case_meta["subtitle"] if case_meta else "UMSF 通用数理晶格算法 + OpenCut同页字块互校 + 微信账单基线字体取证",
        "human_note": case_meta.get("human_note", "") if case_meta else "无需依赖画面具体物体个例，直接由 UMSF 2x2晶格奇偶破缺、OpenCut同页横向互校与微信双列坐标轴算子完成裁定。",
        "combined_probability": combined_prob,
        "verdict": verdict_str,
        "images": {
            "original": f"/files/{img_rel}",
            "six_panel": f"/files/{rel_dir}/{stem}_forensic_report.png",
            "visual_prompt": f"/files/{rel_dir}/{stem}_visual_prompt.jpg",
            "local_edit_map": f"/files/{rel_dir}/{stem}_local_edit_map.jpg",
            "receipt_forensic_map": f"/files/{rel_dir}/{stem}_receipt_forensic_map.jpg",
            "opencut_forensic_map": f"/files/{rel_dir}/{stem}_opencut_forensic_map.jpg",
        },
        "universal_algorithm": univ_res,
        "opencut_analysis": opencut_stats,
        "receipt_analysis": receipt_stats,
        "signal_evaluation": eval_summary,
        "deep_framework": deep_res,
        "local_edit_analysis": local_edit_stats,
        "forensic_prompt_markdown": prompt_text,
        "vlm_api_report": None,
    }


def run_vlm_judge_on_payload(case_payload: Dict[str, Any], override_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    orig_rel = case_payload["images"]["original"].lstrip("/")
    if orig_rel.startswith("files/"):
        orig_rel = orig_rel[len("files/") :]
    orig_path = (ROOT_DIR / orig_rel).resolve()

    diag_rel = case_payload["images"].get("local_edit_map") or case_payload["images"].get("visual_prompt")
    diag_path = None
    if diag_rel:
        clean_diag = diag_rel.lstrip("/")
        if clean_diag.startswith("files/"):
            clean_diag = clean_diag[len("files/") :]
        diag_path = (ROOT_DIR / clean_diag).resolve()

    df = case_payload.get("deep_framework", {})
    le = case_payload.get("local_edit_analysis", {})
    se = case_payload.get("signal_evaluation", {})
    metrics_digest = {
        "local_signal_verdict": case_payload.get("verdict"),
        "local_combined_probability": case_payload.get("combined_probability"),
        "bayer_chromatic_asymmetry": df.get("test3_rgb_highfreq_coupling", {}).get("bayer_chromatic_asymmetry"),
        "vae_to_bayer_periodicity_ratio": df.get("test2_vae_stride_vs_bayer_cfa", {}).get("vae_to_bayer_periodicity_ratio"),
        "sensor_noise_non_uniformity": df.get("test1_sensor_noise_physics", {}).get("noise_non_uniformity_ratio"),
        "local_edit_max_zscore": le.get("max_patch_zscore"),
        "local_edit_primary_bbox": le.get("primary_edit_bbox"),
        "is_localized_ai_edit": le.get("is_localized_ai_edit"),
        "debiased_rdd_score": se.get("debiased_rdd_score"),
    }

    return call_third_party_vlm_forensics(
        original_img_path=orig_path,
        diagnostic_img_path=diag_path,
        forensic_prompt_md=case_payload.get("forensic_prompt_markdown", ""),
        local_metrics_summary=metrics_digest,
        override_config=override_cfg,
    )


class ForensicHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Quiet clean logging
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
            return

        if path == "/api/presets":
            cases = []
            for c in PRESET_CASES:
                if c["file_path"].exists():
                    cases.append(build_case_payload(c["file_path"], c["eval_dir"], c))
            self._send_json({"presets": cases, "human_rules": load_rules(), "api_config": get_public_config()})
            return

        if path == "/api/config":
            self._send_json(get_public_config())
            return

        if path == "/api/rules":
            self._send_json({"human_rules": load_rules()})
            return

        if path.startswith("/files/"):
            rel = path[len("/files/") :]
            target = (ROOT_DIR / rel).resolve()
            if str(target).startswith(str(ROOT_DIR.resolve())):
                self._serve_file(target)
                return
            self.send_error(403, "Forbidden")
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if parsed.path == "/api/config":
            save_api_config(payload)
            self._send_json({"status": "ok", "api_config": get_public_config()})
            return

        if parsed.path in ("/api/analyze", "/api/v1/detect"):
            # Expects {"filename": "xxx.jpg", "image_base64": "data:image/...;base64,...", "call_vlm": true/false}
            b64_str = payload.get("image_base64", "")
            orig_name = payload.get("filename", "upload.jpg")
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)

            digest = hashlib.md5(raw_bytes).hexdigest()[:10]
            ext = Path(orig_name).suffix.lower() or ".jpg"
            if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                ext = ".jpg"

            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            saved_img = UPLOADS_DIR / f"upload_{digest}{ext}"
            saved_img.write_bytes(raw_bytes)
            eval_out = UPLOADS_DIR / f"eval_{digest}"

            try:
                result = build_case_payload(saved_img, eval_out, None)
                cfg = load_api_config()
                should_call_vlm = payload.get("call_vlm", cfg.get("auto_call_on_upload", True))
                if should_call_vlm and (cfg.get("api_key") or payload.get("api_key")):
                    vlm_res = run_vlm_judge_on_payload(result, override_cfg=payload.get("api_override"))
                    result["vlm_api_report"] = vlm_res
                    if vlm_res.get("status") == "ok" and vlm_res.get("parsed_report"):
                        pr = vlm_res["parsed_report"]
                        vlm_conf = float(pr.get("confidence_score", result["combined_probability"]))
                        result["combined_probability"] = round(0.5 * result["combined_probability"] + 0.5 * vlm_conf, 2)
                        if pr.get("verdict_zh"):
                            result["verdict"] = f"{pr['verdict_zh']} (本地物理 + {vlm_res.get('model')} 联合裁定)"
                self._send_json({"status": "ok", "result": result})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if parsed.path == "/api/vlm_judge":
            # Runs third-party VLM API on an existing case payload or image path
            case_data = payload.get("case_data")
            override_cfg = payload.get("api_override")
            if not case_data:
                self._send_json({"error": "Missing case_data"}, status=400)
                return
            try:
                vlm_res = run_vlm_judge_on_payload(case_data, override_cfg=override_cfg)
                self._send_json({"status": "ok", "vlm_api_report": vlm_res})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if parsed.path == "/api/add_rule":
            entry = add_human_observation(
                image_name=payload.get("image_name", "custom_sample"),
                category=payload.get("category", "Human Forensic Observation"),
                human_observation=payload.get("observation", ""),
                compiled_prompt_rule=payload.get("prompt_rule", ""),
                model_missed_why=payload.get("missed_why", ""),
            )
            self._send_json({"status": "ok", "rule": entry, "all_rules": load_rules()})
            return

        self.send_error(404, "Endpoint not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start PixelSentinel AI Image Forensics Standalone Program")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8088, help="Port number")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open default browser on startup")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), ForensicHTTPRequestHandler)
    print("=" * 72)
    print(f"🛡️  PixelSentinel AI 图像伪造与局部改图取证程序已启动！")
    print(f"🌐  访问地址: {url}")
    print(f"🔌  支持一键接入第三方多模态 API (硅基流动 / 通义千问 Qwen-VL / 智谱 GLM-4V / GPT-4o / Gemini)")
    print("=" * 72)

    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    server.serve_forever()


if __name__ == "__main__":
    main()
