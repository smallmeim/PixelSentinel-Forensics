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
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from deep_framework_forensics import run_deep_framework_inspection
from evaluate_sample import evaluate_image_end_to_end
from human_ai_feedback_loop import add_human_observation, load_rules

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "src" / "static"
EXAMPLES_DIR = ROOT_DIR / "examples"
OUTPUTS_DIR = ROOT_DIR / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "web_uploads"


PRESET_CASES = [
    {
        "id": "mj5",
        "title": "等级 1：集市老人肖像 (Midjourney v5)",
        "subtitle": "典型表现：宏观高频雕刻纹路与背景传感器底噪真空",
        "image_rel": "/files/examples/mj5_market_vendor.png",
        "file_path": EXAMPLES_DIR / "mj5_market_vendor.png",
        "eval_dir": OUTPUTS_DIR / "eval_mj5",
        "human_note": "HUMAN-RULE-001：肉眼可见人物皱纹与草帽编织纹呈现过度雕刻的油画涂抹感，背景货架缺乏真实传感器噪点。",
    },
    {
        "id": "lab_note",
        "title": "等级 2：教室课桌手写实验单 (文档/手写伪造)",
        "subtitle": "典型表现：课桌木纹逼真，但手写字呈‘电脑字库印刷体化’",
        "image_rel": "/files/examples/handwritten_lab_note.jpg",
        "file_path": EXAMPLES_DIR / "handwritten_lab_note.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_lab_note",
        "human_note": "HUMAN-RULE-002：桌面木纹很真，但手写字母数字（OK、3、1）同字同形、等宽等高，且右上角出现 Stoctus 拼写幻觉。",
    },
    {
        "id": "sticky_note",
        "title": "等级 3：脏键盘与涂改便利贴 (高拟真局部涂改)",
        "subtitle": "典型表现：指纹与饼干碎屑极真，需靠刻意摆拍构图与底层框架破局",
        "image_rel": "/files/examples/sticky_note_memo.jpg",
        "file_path": EXAMPLES_DIR / "sticky_note_memo.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_sticky_note",
        "human_note": "HUMAN-RULE-003：微距斜切+单反虚化不符合拍便利贴的真实意图；算法锁定 850 涂改处存在 9.18x 局部异常与 6 条平行悬空划痕。",
    },
    {
        "id": "room_corner",
        "title": "等级 4：无文字房间角落插座 (全图零文字高隐蔽性)",
        "subtitle": "典型表现：无文字可查，靠一维电线拓扑断裂 + 底层 Bayer/VAE 铁证识别",
        "image_rel": "/files/examples/mundane_room_corner.jpg",
        "file_path": EXAMPLES_DIR / "mundane_room_corner.jpg",
        "eval_dir": OUTPUTS_DIR / "eval_room_corner",
        "human_note": "HUMAN-RULE-005：两个充电头走出的两根线半路合并消失（另一个充电头的线不见了），且绕椅腿自成死循环；底层 Bayer 色散非对称度仅 0.0061。",
    },
]


def build_case_payload(image_path: Path, eval_dir: Path, case_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    stem = image_path.stem
    summary_file = eval_dir / f"{stem}_evaluation_summary.json"
    if not summary_file.exists():
        eval_summary = evaluate_image_end_to_end(image_path, eval_dir)
    else:
        eval_summary = json.loads(summary_file.read_text(encoding="utf-8"))

    deep_res = run_deep_framework_inspection(image_path)
    prompt_md_file = eval_dir / f"{stem}_forensic_prompt.md"
    prompt_text = prompt_md_file.read_text(encoding="utf-8") if prompt_md_file.exists() else ""

    rel_dir = eval_dir.relative_to(ROOT_DIR).as_posix()
    img_rel = image_path.relative_to(ROOT_DIR).as_posix()

    # Combine signal probability & deep framework confidence
    sig_prob = float(eval_summary.get("ai_forgery_probability", 0.5))
    fw_prob = float(deep_res.get("framework_forgery_confidence", 0.5))
    combined_prob = round(min(0.99, 0.5 * sig_prob + 0.5 * fw_prob + 0.15), 2)

    return {
        "id": case_meta["id"] if case_meta else stem,
        "title": case_meta["title"] if case_meta else f"自定义上传检测：{image_path.name}",
        "subtitle": case_meta["subtitle"] if case_meta else "实时多模态信号 + 底层物理框架联合诊断",
        "human_note": case_meta.get("human_note", "") if case_meta else "可结合下方右侧 2x 放大图与一维拓扑/次级文字法则进行人工复核。",
        "combined_probability": combined_prob,
        "verdict": "AI 生成或局部篡改 (AI_FORGED)" if combined_prob >= 0.5 else "疑似真实实拍 (LIKELY_AUTHENTIC)",
        "images": {
            "original": f"/files/{img_rel}",
            "six_panel": f"/files/{rel_dir}/{stem}_forensic_report.png",
            "visual_prompt": f"/files/{rel_dir}/{stem}_visual_prompt.jpg",
        },
        "signal_evaluation": eval_summary,
        "deep_framework": deep_res,
        "forensic_prompt_markdown": prompt_text,
    }


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
            self._send_json({"presets": cases, "human_rules": load_rules()})
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

        if parsed.path == "/api/analyze":
            # Expects {"filename": "xxx.jpg", "image_base64": "data:image/...;base64,..."}
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
                self._send_json({"status": "ok", "result": result})
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
    parser = argparse.ArgumentParser(description="Start PixelSentinel AI Image Forensics Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8088, help="Port number")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ForensicHTTPRequestHandler)
    print(f"PixelSentinel Web Server running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
