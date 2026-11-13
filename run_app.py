#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelSentinel Standalone One-Click Launcher (一键启动程序入口)
=============================================================
Run `python run_app.py` or double-click `start_pixelsentinel.bat` to launch
the PixelSentinel Forensics Program and automatically open the UI in your browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_app import main  # noqa: E402

if __name__ == "__main__":
    if "--open-browser" not in sys.argv and "--no-browser" not in sys.argv:
        sys.argv.append("--open-browser")
    main()
