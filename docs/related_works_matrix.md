# AI 图像伪造检测与篡改定位开源仓库对标矩阵（2024–2026）

本文件整理了与本项目高度相关的前沿开源仓库，按**「统一评测基准」「免训练与去偏检测」「文本/手写与局部篡改定位」「轻量级工程工具」**四大类归档，供算法复现与基线对比使用。

---

## 一、统一评测框架与基准（Benchmarks）

| 仓库名称 | 发表会议/年份 | 核心覆盖范围 | 关键特点与借用价值 |
| :--- | :---: | :--- | :--- |
| [Ekko-zn/AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark) | Benchmark | 8 种主流 AIGC 检测器（CNNSpot, FreDect, Fusing, Gram-Net, LGrad, LNP, DIRE, UnivFD） | 提供统一 `eval_all.py` 与预训练权重，自带 `--noise_type jpg/blur/resize` 扰动评测，非常适合改造成**微信社交传播链退化实验**。 |
| [scu-zjz/ForensicHub](https://github.com/scu-zjz/ForensicHub) | **NeurIPS 2025** | 四赛道合一：AIGC 全图合成 + IMDL 局部篡改 + **Doc 文档/文字篡改** + Deepfake | 破解「取证预判悖论」的代表性框架，统一了 10+ 种基线与 6 种骨干网，支持跨赛道交叉评测（3×3 误判矩阵）。 |
| [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) | **ICLR 2025** | *A Sanity Check for AI-Generated Image Detection* + Chameleon 数据集 | 「先对现有检测器做体检、再提双专家改进（高频噪声 SRM + 视觉语义 CLIP）」的标杆范式。 |
| [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | NeurIPS 2023/2025 | 人脸深度伪造与通用伪造评测 | 模块化程度高，适合作为人脸与人像美颜伪造分支的参考。 |

---

## 二、免训练检测、去偏与泛化方法（Training-Free & Generalization）

| 仓库名称 | 发表会议/年份 | 核心技术路线 | 对本项目的意义 |
| :--- | :---: | :--- | :--- |
| [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) | **CVPR 2024** | **Training-Free**：利用扩散模型自编码器（VAE）计算 LPIPS 重建误差 | 本项目核心基线之一；无需训练即可在 RTX 5060 (8GB) 上跑通端到端推理，且天然支持块级（Patch-wise）误差热力图输出。 |
| **RDD** *(Debiased Reconstruction)* | **CVPR 2026** | **Training-Free 去偏**：90° 旋转 + 低通滤波自参照校准 + 图像域/潜空间双分支融合 | 直接解决 AEROBLADE 等方法把「白墙/文档真照片」误判成 AI 图的致命缺陷。 |
| [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) | **ICML 2025 Oral** | 正交子空间分解（SVD 保留主语义空间 + 残差子空间学伪影） | 强泛化能力代表作，防止分类器过拟合特定生成器的表层特征。 |
| [mever-team/spai](https://github.com/mever-team/spai) | **CVPR 2025** | 频谱学习（Spectral Learning），支持任意原生分辨率输入 | 处理高分辨率手机实拍原图（无需中心裁剪或暴力缩放破坏高频伪影）时极具参考价值。 |
| [wyczzy/AIGI-Holmes](https://github.com/wyczzy/AIGI-Holmes) | **ICCV 2025** | 多模态大模型（MLLM）可解释伪造检测 | 不仅输出真假概率，还能给出人类可读的伪造证据解释。 |

---

## 三、场景文字、手写文档与局部篡改定位（Text & Local Manipulation Forensics）

针对 **OpenCut**（AI 替换现实图片中的手写字/印刷字）等局部篡改场景，以下仓库提供了从噪声指纹到多尺度分割的完整技术栈：

| 仓库 / 论文 | 发表会议/年份 | 核心技术路线 | 针对 OpenCut / 手写改字的价值 |
| :--- | :---: | :--- | :--- |
| [scu-zjz/Mesorch](https://github.com/scu-zjz/Mesorch) | **AAAI 2025** | **多尺度介观架构（Mesoscopic Architecture）**：CNN + Transformer 混合建模微观噪声与宏观上下文 | 官方在**文本篡改图片**上做过专门复现验证，能够精准分割出被 AI 替换或擦除重绘的文字区域。 |
| [MrHuan3/MUN](https://github.com/MrHuan3/MUN) | **AAAI 2025** | **噪声指纹（NoisePrint）** + 双向解码器 | 专门捕捉局部扩散重绘（Inpainting）导致的相机传感器背景噪声断层。 |
| **EAN** *(Edge-Aware Network)* | **TCSVT 2025** | 边缘感知模块（EAM）+ 特征融合模块（FFM） | 锁定拼接、擦除重绘与文字替换留下的边界梯度异常。 |
| [Kamichanw/Loupe](https://github.com/Kamichanw/Loupe) | **IJCAI 2025** | 轻量化篡改检测与定位（DDL 挑战赛冠军方案） | 无需庞大算力即可在消费级显卡上实现高精度局部篡改定位。 |
| **TextSleuth** | **MIPR 2024** | 面向 AI 篡改文本（Scene Text Editing）的可解释检测 (`arXiv:2412.14816`) | 专门针对 AnyText / DiffSTE 等扩散改字模型设计，分析笔画纹理抖动与背景差异。 |
| **FFDN** | **ECCV 2024** | 频域与空域融合的数字文档篡改检测网络 | 针对文档、票据、手写签名与翻拍文字篡改的高鲁棒性检测。 |
| [SZAISEC/CLUE](https://github.com/SZAISEC/CLUE) | **arXiv 2025** | 利用扩散模型特征反推局部图像伪造区域 | 用魔法打败魔法：用扩散特征捕捉扩散局部重绘痕迹。 |

---

## 四、开箱即用的工程与离线检测工具（Engineering Tools）

| 仓库名称 | 语言/栈 | 功能亮点 |
| :--- | :---: | :--- |
| [lynote-ai/ai-image-detector](https://github.com/lynote-ai/ai-image-detector) | Python (Web UI + CLI) | 自带命令行、REST API 与 Web 演示界面，适合作为竞赛可执行程序的前端交互参考。 |
| [MatrixA/aicheck](https://github.com/MatrixA/aicheck) | Rust | 免 GPU 极速离线检查：解析 C2PA 签名、EXIF 异常与隐式元数据指纹。 |
| [ant-research/Awesome-AIGC-Image-Video-Detection](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) | 论文索引 | 蚂蚁集团维护的 AIGC 图像与视频伪造检测最新论文全集。 |
