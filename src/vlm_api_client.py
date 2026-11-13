#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Third-Party Multimodal VLM API Client (第三方大模型 API 接入引擎)
=============================================================================
Supports zero-dependency (urllib) multimodal forensic inspection calls to:
1. OpenAI-Compatible Vision APIs:
   - SiliconFlow (硅基流动: Qwen/Qwen2.5-VL-72B-Instruct, Pro/Qwen/Qwen2.5-VL-7B-Instruct)
   - Alibaba Cloud DashScope (阿里云百炼: qwen-vl-max, qwen-vl-plus)
   - Zhipu AI (智谱开放平台: glm-4v-plus, glm-4v)
   - OpenAI (gpt-4o, gpt-4o-mini)
   - OpenRouter / OneAPI / NewAPI / Local Ollama / vLLM
2. Google Gemini Native API (gemini-2.5-flash, gemini-2.5-pro)

Sends BOTH:
  - Image 1: Original suspicious image
  - Image 2: PixelSentinel Local Edit & Visual Prompt composite image (with bounding boxes & 3x zoom)
Along with the quantitative signal-injected forensic prompt (`forensic_prompt.md`),
so the third-party API can see both the raw image and our underlying signal heatmaps.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_config.json"

PROVIDER_PRESETS: List[Dict[str, str]] = [
    {
        "id": "siliconflow",
        "name": "硅基流动 SiliconFlow (推荐·支持 Qwen2.5-VL)",
        "protocol": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "placeholder_key": "sk-...",
    },
    {
        "id": "dashscope",
        "name": "阿里云百炼 DashScope (通义千问 Qwen-VL)",
        "protocol": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-vl-max",
        "placeholder_key": "sk-...",
    },
    {
        "id": "zhipu",
        "name": "智谱清言 BigModel (GLM-4V-Plus)",
        "protocol": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4v-plus",
        "placeholder_key": "填写智谱 API Key",
    },
    {
        "id": "openai",
        "name": "OpenAI / 兼容中转站 (GPT-4o / OneAPI)",
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "placeholder_key": "sk-...",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter (聚合多模态 API)",
        "protocol": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "qwen/qwen-2.5-vl-72b-instruct",
        "placeholder_key": "sk-or-v1-...",
    },
    {
        "id": "gemini",
        "name": "Google Gemini 官方 API (Gemini 2.5 Flash/Pro)",
        "protocol": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-2.5-flash",
        "placeholder_key": "AIzaSy...",
    },
    {
        "id": "custom",
        "name": "自定义 OpenAI 兼容接口 (本地 Ollama / vLLM / 其他)",
        "protocol": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5vl:7b",
        "placeholder_key": "ollama",
    },
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "provider_id": "siliconflow",
    "protocol": "openai",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "",
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "auto_call_on_upload": True,
    "temperature": 0.1,
}


def load_api_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_api_config(new_cfg: Dict[str, Any]) -> Dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    current = load_api_config()
    for k in ("provider_id", "protocol", "base_url", "model", "auto_call_on_upload", "temperature"):
        if k in new_cfg and new_cfg[k] is not None:
            current[k] = new_cfg[k]
    # Only overwrite api_key if user provided a non-masked key
    raw_key = str(new_cfg.get("api_key", "")).strip()
    if raw_key and "***" not in raw_key:
        current["api_key"] = raw_key
    elif "clear_key" in new_cfg and new_cfg["clear_key"]:
        current["api_key"] = ""

    CONFIG_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}***{key[-4:]}"


def get_public_config() -> Dict[str, Any]:
    cfg = load_api_config()
    pub = dict(cfg)
    pub["has_api_key"] = bool(cfg.get("api_key", "").strip())
    pub["api_key_masked"] = mask_api_key(cfg.get("api_key", ""))
    pub.pop("api_key", None)
    pub["presets"] = PROVIDER_PRESETS
    return pub


def _encode_image_to_jpeg_b64(img_path: Path, max_side: int = 1280) -> str:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract structured JSON block from VLM markdown/text response, or fallback gracefully."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if not candidate:
        brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        candidate = brace.group(1) if brace else None

    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Fallback heuristic parser if model returned plain text
    verdict = "AI_GENERATED_OR_TAMPERED"
    if any(k in text for k in ("真实未改", "AUTHENTIC", "真实实拍", "未见明显伪造")):
        if not any(k in text for k in ("局部篡改", "AI生成", "伪造")):
            verdict = "AUTHENTIC"
    return {
        "final_verdict": verdict,
        "forgery_type": "见下方详细分析",
        "confidence_score": 0.85,
        "visual_clues_found": ["模型已返回自由文本取证报告（详见下方完整推理链）"],
        "reasoning_summary": text.strip()[:800],
    }


def call_third_party_vlm_forensics(
    original_img_path: Path,
    diagnostic_img_path: Optional[Path],
    forensic_prompt_md: str,
    local_metrics_summary: Dict[str, Any],
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calls the configured third-party VLM API with:
      - Image 1: Original suspicious image
      - Image 2: Diagnostic composite image (Local Edit Map or Visual Prompt)
      - Text: Quantitative forensic prompt + JSON output instruction
    """
    cfg = load_api_config()
    if override_config:
        for k, v in override_config.items():
            if v is not None and v != "" and "***" not in str(v):
                cfg[k] = v

    api_key = str(cfg.get("api_key", "")).strip()
    base_url = str(cfg.get("base_url", "")).strip().rstrip("/")
    model = str(cfg.get("model", "")).strip()
    protocol = str(cfg.get("protocol", "openai")).strip().lower()

    if not api_key:
        return {
            "status": "no_api_key",
            "error": "尚未配置第三方 API Key。请在页面顶部「🔌 第三方 API 接入配置」中填入 API Key（如硅基流动、阿里云百炼、智谱 GLM-4V、OpenAI 或 Gemini）。",
        }

    orig_b64 = _encode_image_to_jpeg_b64(original_img_path, max_side=1280)
    diag_b64 = (
        _encode_image_to_jpeg_b64(diagnostic_img_path, max_side=1400)
        if diagnostic_img_path and diagnostic_img_path.exists()
        else None
    )

    system_instruction = (
        "你是一位国家级信息安全竞赛的「AI 图像伪造与局部篡改取证专家 (PixelSentinel VLM Judge)」。"
        "你将同时收到：\n"
        "1. 第一张图：原始待检图像；\n"
        "2. 第二张图：底层信号检测器生成的「局部 AI 改图定位与放大诊断图」（红框为算法锁定的局部篡改/重绘高度可疑区，右侧为 3 倍放大图）；\n"
        "3. 文本：底层物理框架指标（Bayer 色散非对称度、VAE 步长周期比、局部滑窗差分峰值 Z-Score）以及人类专家总结的六大取证法则。\n\n"
        "请结合你的视觉观察（文字乱码、手写印刷体化、一维电线拓扑断裂、局部改图边缘横线截断、微距摆拍意图）与底层信号指标，给出最终裁定。\n"
        "请务必在回复中包含如下 JSON 格式结论（可放在 ```json 代码块中）：\n"
        "{\n"
        '  "final_verdict": "FULL_AI_GENERATED | LOCAL_AI_TAMPERED | AUTHENTIC",\n'
        '  "verdict_zh": "全图 AI 生成伪造 | 局部 AI 改图/涂抹重绘 | 真实未改原图",\n'
        '  "confidence_score": 0.0 到 1.0 之间的小数,\n'
        '  "tampered_region_desc": "若为局部改图或有具体破绽，指出具体位置与内容（如：B201行PASS字样、河面石头抹除区、充电器第2根线消失等）",\n'
        '  "visual_clues_found": ["破绽1...", "破绽2...", "底层物理框架佐证3..."],\n'
        '  "reasoning_summary": "详细的图文结合取证推理过程（150~300字）"\n'
        "}"
    )

    user_text = (
        f"### 底层物理信号与局部差分预检摘要\n"
        f"```json\n{json.dumps(local_metrics_summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"### 结构化取证检查指令与六大人类法则\n"
        f"{forensic_prompt_md}\n\n"
        f"请仔细审阅所附原图与局部放大诊断图，按照要求的 JSON 格式输出最终取证结论。"
    )

    try:
        if protocol == "gemini" or "generativelanguage.googleapis.com" in base_url:
            raw_reply = _call_gemini_api(base_url, api_key, model, system_instruction, user_text, orig_b64, diag_b64)
        else:
            raw_reply = _call_openai_compatible_api(
                base_url, api_key, model, system_instruction, user_text, orig_b64, diag_b64, float(cfg.get("temperature", 0.1))
            )

        parsed_json = _extract_json_from_text(raw_reply)
        return {
            "status": "ok",
            "provider": cfg.get("provider_id", "custom"),
            "model": model,
            "parsed_report": parsed_json,
            "raw_response": raw_reply,
        }
    except urllib.error.HTTPError as he:
        err_body = ""
        try:
            err_body = he.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return {
            "status": "error",
            "error": f"第三方 API HTTP {he.code} 错误: {he.reason}. 详情: {err_body}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"调用第三方 API 失败: {str(e)}",
        }


def _call_openai_compatible_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    orig_b64: str,
    diag_b64: Optional[str],
    temperature: float = 0.1,
) -> str:
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    content_list: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
    ]
    if diag_b64:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{diag_b64}"}})
    content_list.append({"type": "text", "text": user_text})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list},
        ],
        "temperature": temperature,
        "max_tokens": 1200,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "PixelSentinel-Forensics/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=65) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def _call_gemini_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    orig_b64: str,
    diag_b64: Optional[str],
) -> str:
    clean_model = model.replace("models/", "")
    url = f"{base_url}/models/{clean_model}:generateContent?key={api_key}"

    parts: List[Dict[str, Any]] = [
        {"text": system_prompt + "\n\n" + user_text},
        {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}},
    ]
    if diag_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": diag_b64}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "PixelSentinel-Forensics/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=65) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]
