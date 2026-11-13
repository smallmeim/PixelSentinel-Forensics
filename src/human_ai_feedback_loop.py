#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-in-the-Loop Forensic Knowledge Compiler (人机协同取证规则沉淀器)
======================================================================
Records human expert observations on AI-forged images (e.g., where the human eye
spotted unnatural details that standard models missed) and automatically compiles
them into:
1. `data/human_forensic_rules.json` (structured knowledge base of human forensic heuristics)
2. Updated prompt clauses for `forensic_prompt_pipeline.py` so future VLM/AI inspections
   explicitly check the human-discovered cues.

Usage:
    python src/human_ai_feedback_loop.py --image examples/mj5_market_vendor.png \
        --category "Physical/Optical & Semantic Detail" \
        --observation "帽檐编织纹理在右侧断裂，背景货架上的商品包装文字退化为不可读乱码，眼镜框左右鼻托不对称" \
        --prompt_rule "放大检查帽檐/织物周期性编织纹理是否在中途发生拓扑错位，以及背景货架/瓶身标签文字是否退化为乱码符号"
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List


RULES_FILE = Path(__file__).resolve().parent.parent / "data" / "human_forensic_rules.json"


def load_rules() -> List[Dict[str, Any]]:
    if RULES_FILE.exists():
        return json.loads(RULES_FILE.read_text(encoding="utf-8"))
    return []


def add_human_observation(
    image_name: str,
    category: str,
    human_observation: str,
    compiled_prompt_rule: str,
    model_missed_why: str = "",
) -> Dict[str, Any]:
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    entry = {
        "rule_id": f"HUMAN-RULE-{len(rules) + 1:03d}",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "sample_image": image_name,
        "category": category,
        "human_observation": human_observation,
        "model_missed_why": model_missed_why,
        "compiled_prompt_rule": compiled_prompt_rule,
    }
    rules.append(entry)
    RULES_FILE.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Record human forensic observation into rule base")
    parser.add_argument("--image", type=str, required=True, help="Sample image name or path")
    parser.add_argument("--category", type=str, default="Visual & Physical Anomaly", help="Clue category")
    parser.add_argument("--observation", type=str, required=True, help="How the human spotted the AI trace")
    parser.add_argument("--prompt_rule", type=str, required=True, help=" actionable prompt check derived from human")
    parser.add_argument("--missed_why", type=str, default="", help="Why the baseline AI/signal model missed it")
    args = parser.parse_args()

    entry = add_human_observation(
        image_name=args.image,
        category=args.category,
        human_observation=args.observation,
        compiled_prompt_rule=args.prompt_rule,
        model_missed_why=args.missed_why,
    )
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
