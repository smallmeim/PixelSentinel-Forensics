@echo off
chcp 65001 >nul
title PixelSentinel AI 图像伪造与局部改图取证程序
cd /d "%~dp0"

echo ========================================================================
echo   🛡️  正在启动 PixelSentinel AI 图像伪造与局部篡改取证程序...
echo   🔌  启动后可在顶部直接填入第三方 API Key (硅基流动/千问/智谱/OpenAI/Gemini)
echo ========================================================================

if exist "D:\aigi\venv\Scripts\python.exe" (
    "D:\aigi\venv\Scripts\python.exe" run_app.py
) else (
    python run_app.py
)
pause
