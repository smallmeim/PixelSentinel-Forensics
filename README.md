# 🛡️ PixelSentinel-Forensics

> **信息安全竞赛 · AI 图像伪造识别、局部文字篡改取证与检测器去偏研究**  
> *AI-Generated Image (AIGI) Detection, Scene/Handwritten Text Tampering Forensics & Detector Debiasing Research*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch cu128](https://img.shields.io/badge/PyTorch-2.7%2B%20(cu128)-EE4C2C.svg)](https://pytorch.org/)
[![Track](https://img.shields.io/badge/Competition-InfoSec%20Content%20Security-00F5D4.svg)](#)

---

## 📌 一、项目背景与核心问题

随着扩散模型（Diffusion Models）与流匹配（Flow Matching）生成技术的演进，视觉内容安全面临两大极具挑战性的现实威胁：

1. **全图合成与检测器「背景复杂度偏差」（Background Complexity Bias）**  
   主流免训练/训练型 AI 图像检测器（如 AEROBLADE、DIRE、CNNSpot）依赖自编码器重建误差或频域残差判别真伪。然而在**中文手机实拍场景**（如白墙人像、纸质文档、干净天空、微信压缩图像）中，由于真实图像本身背景极其简单，自编码器重建误差天然偏低，导致**大量真实实拍照片被误判为 AI 生成图（高假阳性率 FPR）**。
2. **以 OpenCut 为代表的「AI 局部无痕改字 / 手写字篡改」（Localized AI Text & Handwriting Tampering）**  
   新一代在线 AI 视觉编辑工具（如 **OpenCut** 的 AI 图片无痕改字、阿里 **AnyText**、**DiffSTE** 等）能够在保留 95% 以上真实相机底图的前提下，仅对图像中的**手写字、收据金额、笔记、路牌或证件文字**进行局部擦除重绘与风格迁移。此类「真底图 + 局部微区域扩散重绘」直接击穿了传统全局 AIGC 检测器，形成图像取证领域的**「预判悖论」（Pre-judgment Paradox）**。

本项目面向信息安全竞赛（内容安全与安全检测赛道），构建一套**「检测器偏差体检 + 参照变换去偏 + 局部文字/手写篡改多维取证」**的研究与工具框架。

---

## 🧭 二、四大取证赛道与「预判悖论」

受 NeurIPS 2025 [ForensicHub](https://github.com/scu-zjz/ForensicHub) 统一取证基准启发，我们将 AI 视觉伪造取证划分为四个正交又相互交织的维度：

```mermaid
flowchart TD
    A["未知来源待检图像 (Input Image)"] --> B{"取证预判悖论\n(Pre-judgment Paradox)"}
    B --> C["赛道 1: 全图 AIGC 合成\n(SD / Midjourney / Flux)"]
    B --> D["赛道 2: 场景与手写文字局部篡改\n(OpenCut / AnyText / DiffSTE)"]
    B --> E["赛道 3: 通用局部篡改定位 (IMDL)\n(拼接 / 复制移动 / 擦除重绘)"]
    B --> F["赛道 4: 人脸深度伪造与美颜掩蔽\n(Deepfake / Face Retouching)"]

    C --> G["核心痛点: 白墙/文档等低复杂度真图被误判"]
    D --> H["核心痛点: 95% 真实底图掩护 5% 局部扩散改字"]
    G --> I["PixelSentinel 解决方案:\n1. 旋转/低通滤波参照去偏 (RDD)\n2. 多尺度噪声残差 + 块级重建热力图 + 笔画边缘取证"]
    H --> I
```

---

## 🔍 三、重点研究：如何识破 OpenCut 等 AI 手写字/文字篡改与全图伪造？

针对 **OpenCut（AI 图片无痕改字）** 等可以替换现实照片中手写字或印刷字的工具，以及主流扩散生成模型，我们总结并实现了 **5 大核心鉴别与取证方法**（详见 [`docs/opencut_and_text_tampering_forensics.md`](docs/opencut_and_text_tampering_forensics.md)）：

| 维度 | 取证方法 | 针对的造假机理 | 适用场景（全图 AIGC / OpenCut 局部改字） |
| :--- | :--- | :--- | :--- |
| **1. 高频传感器噪声残差** | **SRM 高通滤波与噪声指纹（NoisePrint / Bayar Conv）** | 真实手机拍照含全图一致的 CMOS 传感器噪声（PRNU）与 ISP 滤波痕迹；OpenCut 在文字框内用扩散/LaMa 擦除背景再重绘文字，会在文字矩形框内形成**「高频相机噪声真空区」**。 | ⭐⭐⭐⭐⭐ 极易框出 OpenCut 改字边界 |
| **2. 块级自编码器去偏重建** | **Patch-wise AEROBLADE + 参照去偏（RDD）** | 经过扩散模型重绘的文字区域已经落在 VAE 流形上，其局部 LPIPS 重建误差显著低于周围未篡改的真实手写字；结合 **90° 旋转与低通滤波参照变换**，可剔除白纸/白墙背景带来的共模偏差。 | ⭐⭐⭐⭐⭐ 兼顾全图去偏与局部扩散检测 |
| **3. 笔画微观边缘与纸张纹理** | **多尺度介观特征（Mesorch）+ 边缘感知（EAN）** | 真实手写墨迹在纸张纤维上有自然的渗透梯度与笔画压感连续性；AI 拟合的手写字在笔画边缘存在**高频频谱截断（Spectral Roll-off）**、微弱光晕（Halo）以及字下纸张纹理被平滑涂抹的断层。 | ⭐⭐⭐⭐⭐ 专攻手写体/文档金额篡改 |
| **4. 压缩双重量化与重编码** | **ELA（误差水平分析）+ 8×8 DCT 量化表核验** | 局部修图或网页端工具（OpenCut）导出时，篡改区域的 8×8 JPEG DCT 块残差与原图背景存在压缩代数差（Double JPEG Compression），且量化表从手机原生（Apple/Xiaomi）变为 Web 标准库（Skia/libjpeg）。 | ⭐⭐⭐⭐☆ 快速初筛与低成本取证 |
| **5. 频域周期性伪影分析** | **2D-FFT / DCT 高频谱异常检测** | 上采样层（Upsampling / ConvTranspose）与潜空间解码器会在频谱图高频四角或轴向留下栅格状周期性伪影。 | ⭐⭐⭐⭐☆ 全图 AIGC 生成检测 |

---

## 📂 四、仓库目录结构

```text
PixelSentinel-Forensics/
├── start_pixelsentinel.bat                                    # 🚀 Windows 双击一键启动程序（自动启动服务并打开界面）
├── run_app.py                                                 # 🚀 跨平台一键启动入口（python run_app.py）
├── README.md                                                  # 项目总览与研究导航
├── requirements.txt                                           # 依赖包配置
├── data/
│   └── human_forensic_rules.json                              # ⭐ 人机协同沉淀的六大人工取证专家规则库 (HUMAN-RULE-001~006)
├── examples/                                                  # 六大伪造与局部改图基准测试图集
├── docs/
│   ├── forensic_prompt_engineering_and_sota_training.md       # ⭐ 让AI注意到微观破绽的专用提示词库 + 顶会低算力训练方案调研
│   ├── opencut_and_text_tampering_forensics.md                # 深度调研：OpenCut 手写字篡改机理与五维鉴别法
│   ├── detector_bias_and_rdd_debiasing.md                     # 理论与实验：AI 检测器背景复杂度偏差诊断与去偏方案
│   └── related_works_matrix.md                                # 2024–2026 顶会开源仓库对标矩阵（Benchmark & SOTA）
└── src/
    ├── web_app.py                                             # 🌐 PixelSentinel 独立取证程序后端服务
    ├── universal_forensic_algorithm.py                        # 🧮 UMSF 通用非个例数理与晶格奇偶取证算法（NPR 2x2奇偶破缺 + GGD + 马尔可夫熵）
    ├── vlm_api_client.py                                      # 🔌 第三方多模态大模型 API 接入客户端（支持硅基流动/千问/智谱/GPT-4o/Gemini）
    ├── static/
    │   └── index.html                                         # 🌐 交互式前端界面（内置 API 接入配置栏、六大基准对比、3x局部篡改放大镜）
    ├── local_edit_detector.py                                 # 🎯 局部 AI 改图/涂抹重绘定位器（32x32 滑窗中心-周边差分 + 2D 离轴 FFT 谐波）
    ├── deep_framework_forensics.py                            # 🔬 图片底层物理与生成框架检测（Bayer色散非对称度 + VAE 8x8步长周期）
    ├── evaluate_sample.py                                     # 端到端多模态信号 + 去偏评估器
    ├── forensic_prompt_pipeline.py                            # 免训练：自动生成「视觉提示拼图 + 定量信号注入取证 Prompt」
    ├── forensics_toolkit.py                                   # 多维图像取证与去偏诊断工具（输出六合一热力图报告）
    ├── human_ai_feedback_loop.py                              # 人机协同取证规则沉淀器
    ├── models/
    │   └── clip_forensic_classifier.py                        # 冻结CLIP-ViT + Top-K局部块异常池化 + SRM侧信道轻量级模型定义
    └── data/
        └── synthesize_tampered_data.py                        # OpenCut局部改字与AIGC配对样本合成器
```

---

## 🧠 四（补充）、人机协同沉淀的「六级 AI 伪造与局部改图鉴别体系」（Human-in-the-Loop Rules）

通过多轮真实案例的人机协同对抗实测，我们将 AI 伪造与局部改图痕迹归纳为**「表观语义拓扑 + 局部滑窗差分 + 底层物理框架」六级鉴别法则**（已内置于 [`data/human_forensic_rules.json`](data/human_forensic_rules.json) 与 Web 平台）：

| 规则编号 | 维度名称 | 人工与算法联合判别要诀 | 对应典型场景 |
| :--- | :--- | :--- | :--- |
| **HUMAN-RULE-001** | **宏观雕刻感纹路** | 人物皮肤皱纹、织物、毛发呈现过度锐化雕刻感与局部油画般平滑涂抹并存的伪影。 | 早期 Midjourney / SD 肖像与风景 |
| **HUMAN-RULE-002** | **手写字「印刷体化」** | 背景桌面/纸张极真，但手写字母数字（如多个 `OK`、`3`、`1`）同字同形、等宽等高、笔画无顿笔压感方差，呈「电脑手写字库」质感。 | AI 伪造手写实验单、收据、OpenCut 改字 |
| **HUMAN-RULE-003** | **拍摄动机与光学景深悖论** | 现实中手机拍记录/日常物多为平视或正拍且小底景深很深；AI 倾向于采用斜切微距视角、刻意堆砌背景道具（杯子/外设）并伴随单反级大光圈虚化。 | 高拟真桌面摆拍、微距静物伪造 |
| **HUMAN-RULE-004** | **次级场景小字/车牌乱码** | 主招牌大字（如「停车位」「超市」）已能写对，但放大车牌前缀汉字、挡泥板小字、告示牌副标题时，笔画退化为似字非字的序列乱码。 | 城市街景、门店招牌、路牌随手拍 |
| **HUMAN-RULE-005** | **一维细长物体拓扑断裂 + 底层框架兜底** | 无文字图中，顺着电线/数据线/鞋带追踪可发现「两头合并消失」「有头无尾死循环」；若表观无失误，直接由底层 **Bayer 色散非对称度 (`<0.015`)** 与 **VAE `8×8` 步长自相关** 一票否决。 | 全图无文字日常角落、高阶变色龙级伪造 |
| **HUMAN-RULE-006** | **局部 AI 改图/涂抹重绘边界（Center-Surround 差分）** | 当底图 95% 以上为真实照片仅局部改字（如 `FAIL` 改 `PASS`）或局部抹除重绘时，通过 `32×32` 滑窗检测 **2D 离轴 FFT 网格谐波 (`[8,8] / [4,4]`)** 与 **中心-周边 8 邻域底噪跳变**，精准框出篡改坐标 (`Z > 6σ`)。 | 真实照片局部 Inpainting、OpenCut 无痕改字 |
| **HUMAN-RULE-007** | **三维折叠细框自遮挡穿模 & 镜面反射几何悖论** | 当画面既无文字也无电线、连阴影都极逼真时，放大检查**多层薄结构自遮挡处**（如折叠平放的眼镜架，透过镜片看到的后方镜腿与前框会发生平面化重合粘连），并比对镜面内外物体的配件（如杯子把手消失、牙刷刷毛朝向反转、远端眼镜在镜底的反射位置异常）。 | 无字无电线日常室内照、镜面/玻璃反射场景 |
| **HUMAN-RULE-008** | **微信/社交平台转账账单 PS 与 AI 改数取证** | 原生微信支付账单详情遵循严格的**双列坐标对齐轴定律**（左列标签轴 `X_label` 与右列数值轴 `X_value` 零抖动）、**大额金额同行零基线错位定律**以及 **`WeChat Sans` 专属字体边缘抗锯齿斜率一致性**；PS 贴字改数或 AI 改字会暴露 `+3~6px` 同向基线错位、左右列坐标偏移及跨字体抗锯齿冲突。 | 微信转账账单详情、聊天转账气泡、支付宝电子回单伪造 |

---

## 🌐 五、一键启动独立取证程序 & 接入第三方大模型 API

### 1. 双击一键启动程序（自动拉起服务并打开取证操作台）

- **Windows 双击启动**：直接双击仓库根目录下的 [`start_pixelsentinel.bat`](start_pixelsentinel.bat)
- **命令行一键启动**：
  ```bash
  python run_app.py
  ```
启动后程序会自动打开浏览器进入 `http://127.0.0.1:8088` 操作台。

### 2. 即插即用接入第三方视觉大模型 API（无需本地算力）

在界面顶部的 **「🔌 第三方多模态大模型 API 接入配置」** 栏中：
1. **选择服务商预设**：内置支持 **硅基流动 SiliconFlow (`Qwen/Qwen2.5-VL-72B-Instruct`)**、**阿里云百炼 (`qwen-vl-max`)**、**智谱清言 (`glm-4v-plus`)**、**OpenAI (`gpt-4o`)**、**OpenRouter**、**Google Gemini (`gemini-2.5-flash`)** 以及本地 **Ollama / vLLM**；
2. **填入 API Key 并点击「💾 保存配置」**：密钥仅保存在本地 `config/api_config.json`（已被 `.gitignore` 排除，绝不上传仓库）；
3. **自动物理显影 + API 联合裁定**：上传任意待检图片（或点击顶部的「🤖 调用第三方 API 深度鉴别当前图」），程序会自动：
   - 第一步（本地毫秒级）：计算底层 **Bayer 色散非对称度**、**VAE 步长周期**、**32×32 滑窗局部改图差分热力图**，并生成 **3 倍放大诊断图**；
   - 第二步（API 协同）：将**「原始待检图 + 局部放大诊断图 + 注入定量信号的六大专家规则 Prompt」**一并发送给第三方多模态大模型 API，自动解析 JSON 鉴别结论并与本地物理得分加权融合。

`src/forensics_toolkit.py` 提供了开箱即用的多维图像取证脚本，无需重训大模型即可对任意待检图片（实拍图、AIGC 合成图、OpenCut 局部改字图）生成**六合一可视化诊断报告**：

1. **原图与元数据检查（EXIF / 量化表指纹）**
2. **ELA 重压缩误差热力图（Error Level Analysis）**
3. **SRM 高频噪声残差图（暴露局部擦除与重绘边界）**
4. **笔画/边缘梯度不一致性图（Sobel/Laplacian 边缘取证）**
5. **块级图像复杂度分布图（Background Complexity Map）**
6. **旋转+低通参照去偏异常定位图（Reference-Debiased Anomaly Map）**

### 运行方式

```bash
# 安装基础依赖
pip install numpy pillow scipy matplotlib

# 对单张可疑图片运行多维取证诊断（自动生成可视化报告与 JSON 诊断结论）
python src/forensics_toolkit.py --image path/to/suspicious_image.jpg --output report_out/
```

---

## 📚 六、参考与对标开源仓库（2024–2026）

本仓库在架构设计与方法研究上重点参考并致敬以下前沿工作（详细对比见 [`docs/related_works_matrix.md`](docs/related_works_matrix.md)）：

- **评测与统一框架**：
  - [Ekko-zn/AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark) — 统一集成 CNNSpot、UnivFD、DIRE、LGrad 等 8 种检测器的评测基准
  - [scu-zjz/ForensicHub](https://github.com/scu-zjz/ForensicHub) *(NeurIPS 2025)* — 覆盖 AIGC、IMDL、文档篡改、Deepfake 四赛道的统一取证框架
  - [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) *(ICLR 2025)* — *A Sanity Check for AI-generated Image Detection*
- **免训练重建与去偏**：
  - [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) *(CVPR 2024)* — 基于扩散模型自编码器 LPIPS 重建距离的免训练检测
  - *RDD (CVPR 2026)* — *A Debiased Reconstruction-based Framework*（参照变换去偏核心思想来源）
- **文本/手写篡改与局部取证定位**：
  - [scu-zjz/Mesorch](https://github.com/scu-zjz/Mesorch) *(AAAI 2025)* — 多尺度介观架构，在文本篡改与通用篡改定位上表现优异
  - [MrHuan3/MUN](https://github.com/MrHuan3/MUN) *(AAAI 2025)* — 噪声指纹（NoisePrint）+ 双向解码图像篡改定位
  - [Kamichanw/Loupe](https://github.com/Kamichanw/Loupe) *(IJCAI 2025)* — DDL 挑战赛冠军轻量化篡改检测方案
  - *TextSleuth (MIPR 2024)* & *FFDN (ECCV 2024)* — 面向场景文字编辑（Scene Text Editing）与数字文档篡改的可解释检测

---

## 📄 License

本项目遵循 [MIT License](LICENSE) 开源协议，仅用于信息安全竞赛防御性技术研究与学术交流。
