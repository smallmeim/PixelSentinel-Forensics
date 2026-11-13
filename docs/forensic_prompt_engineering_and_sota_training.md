# 提示词驱动的 AI 图像取证（Forensic Prompting）与主流仓库低算力训练方案调研

> **核心思路**：不依赖本机高算力训练大模型，而是通过**「专家级取证提示词（Forensic Chain-of-Thought Prompts）+ 视觉提示增强（Visual Prompting）」**引导多模态视觉 AI 注意到此前忽略的微观物理破绽（如 OpenCut 换字留下的字下纸纹断裂、墨迹交叉无叠加、传感器噪声真空），或采用顶会主流的**「冻结骨干 + 提示词微调（Prompt Tuning）」**。

---

## 一、其他前沿仓库是怎么「不拼算力」做训练和识别的？（2024–2026 顶会方案）

我们调研了 GitHub 上最新的 6 个代表性仓库，它们解决「算力受限」和「普通 AI 看不出微观伪影」的方法主要有以下三类：

### 1. 「视觉掩码提示 + 专家问题引导」路线（代表作：TextSleuth & FakeShield）
- **出处**：*TextSleuth: Towards Explainable Tampered Text Detection* (`arXiv:2412.14816`)、[FakeShield](https://arxiv.org/pdf/2410.02761)
- **为什么普通 AI 看不出改字？**  
  直接把一张用 **OpenCut** 改过手写字的图片喂给多模态大模型（MLLM），模型默认做「全局语义理解」（看文字通不通顺、画面合不合理），完全不会去放大看某个字下面的纸张纤维和边缘噪声。
- **TextSleuth 的做法（Fused Mask Prompt + Grounding Prompt）**：
  1. 先用极轻量的传统信号/滤波算法（如 SRM 高频噪声残差、ELA 重压缩残差）生成一张**疑似篡改区域高亮图（Visual Prompt）**；
  2. 把「原图 + 局部放大框 + 噪声残差图」拼在一起喂给视觉 AI，并用**结构化取证提示词**强制 AI 逐项检查笔画边缘、背景纹理连续性和光照方向。

### 2. 「专家知识注入提示词」路线（代表作：X²-DFD & AIGI-Holmes）
- **出处**：NeurIPS 2025 *X²-DFD*（王波老师分享材料第 09 篇）、ICCV 2025 [wyczzy/AIGI-Holmes](https://github.com/wyczzy/AIGI-Holmes)
- **核心做法**：
  - **X²-DFD** 先用轻量级取证算子提取定量指标（如高频能量比、局部噪声方差比、背景复杂度），将这些**先验数值直接写进自然语言 Prompt** 中（例如：*“底层信号检测显示该图块 (x,y) 处存在 3.4 倍的高频噪声断层，请重点核查该区域的笔画与背景过渡”*），让 AI 从「盲猜」变成「带着线索审讯」。
  - **AIGI-Holmes** 使用多个开源 MLLM 组成「多专家陪审团（Multi-Expert Jury）」，分别使用专门针对**物理光影、几何透视、微观纹理、手写/印刷文字结构**的提示词独立审图，最后汇总证据链。

### 3. 「图像自适应提示学习（Prompt Learning）」路线（代表作：IAPL & FatFormer）
- **出处**：CVPR 2026 [liyih/IAPL](https://github.com/liyih/IAPL) (*Image-Adaptive Prompt Learning*)、CVPR 2024 [Michel-liu/FatFormer](https://github.com/Michel-liu/FatFormer)
- **是怎么训练的？**
  - **完全冻结**预训练好的 CLIP（`ViT-B/16` 或 `ViT-L/14`，占 99.9% 参数，一丁点都不训）；
  - 只训练输入端的 **16 个可学习提示词向量（Learnable Prompt Tokens, 仅几千到几万个参数）**，把空域特征和频域（DWT/FFT）伪影作为视觉提示注入进 CLIP。
  - **算力需求**：因为只更新极少量的 Prompt 向量，哪怕在免费的 **Google Colab (T4)**、**Kaggle (双 T4)** 或 **ModelScope 免费 Notebook** 上，十来分钟就能训完，完全不需要消耗本地笔记本算力。

---

## 二、核心设计：让图像识别 AI 注意到「先前注意不到的破绽」的专用提示词库

普通提示词（如 *“请判断这张图是不是 AI 生成或修改的”*）之所以失效，是因为存在**三大盲区**：
1. **语义盲区**：AI 看到手写字语义通顺，就忽略了笔画底下的纸张纹理已经断了；
2. **局部稀释盲区**：OpenCut 只改了 3% 面积的文字，AI 扫一眼整张图觉得 97% 都很真实就放过了；
3. **简单背景误判盲区**：AI 看到白墙、白纸背景过于干净平滑，就误以为是 AI 生成的（背景复杂度偏差）。

为此，我们设计了以下 **四套结构化取证提示词（Forensic Prompts）**，可直接用于任何多模态视觉大模型（GPT-4o、Gemini、Qwen2.5-VL、InternVL 等）或取证 Agent：

---

### 📌 提示词模版 1：全能五步法数字图像取证专家提示词（System Prompt）

> **适用场景**：通用待检图片（同时防范「全图 AIGC 生成」与「OpenCut 局部改字/P图篡改」，并内置「白墙/文档防误报护栏」）。

```markdown
你是一位顶级的数字图像取证与视觉内容安全专家（Digital Image Forensics Investigator）。
在审阅图片时，普通视觉 AI 往往只关注宏观语义合理性，从而漏掉局部微小的 AI 擦除重绘痕迹（如 OpenCut 局部无痕改字），或者把背景简单的真实手机实拍图（如白墙、白纸文档）误判为 AI 生成。

请你严格放弃“凭第一印象猜真假”的习惯，必须按照以下【五步微观取证清单】逐项排查图像中普通 AI 容易忽略的物理与信号破绽：

### 【第一步：排查“真底图 + 局部 AI 换字/重绘”（针对 OpenCut / AnyText 等局部篡改）】
请逐一放大审视画面中的所有文字（尤其是手写字、签名、数字金额、日期、路牌、标签）：
1. **字下背景纹理连续性（Sub-Stroke Texture Continuity）**：
   观察文字笔画正下方和紧邻四周的纸张纤维、横格线、木纹或墙面颗粒。如果原图有横格线或纸张折痕，它们是否自然穿过笔画？还是在某几个字周围出现了隐蔽的“矩形涂抹框”、纸纹突然变平滑、或者横格线在字下发生微小的断裂、弯曲与重绘错位？
2. **手写墨迹物理交互（Ink-Paper Physics）**：
   真实手写笔画在起笔、顿笔和**两笔交叉点（Stroke Intersections）**会有双倍墨水沉积（颜色明显变深），且边缘有顺着纸张纤维的微小毛细渗透；而 AI 生成/替换的手写字（如 OpenCut）内部颜色往往过于均匀，交叉处没有墨水叠加深度变化，且笔画边缘带有平滑的数码抗锯齿光晕（Halo）。
3. **同行笔迹与光影孤岛（Intra-Line Consistency）**：
   对比同一行、同一页中的不同文字：是否存在某几个字（特别是关键数字、姓名、金额）的锐度、边缘噪点颗粒感、或受光阴影方向与旁边其他文字存在微妙断层？

### 【第二步：排查高频传感器噪声断层（Sensor Noise & Compression Consistency）】
1. 真实手机实拍照片在全图（包括暗部与纯色区）都有均匀的 CMOS 传感器光子散粒噪声（Sensor Grain）。
2. 检查画面中是否存在局部区域像被“熨斗烫平”一样的**高频噪声真空区（Noise Vacuum）**，或局部块与周围环境存在清晰度/压缩网格断层（常见于 Diffusion Inpainting 局部修补）。

### 【第三步：排查全图 AIGC 生成的微观物理与几何悖论】
1. **高频纹理坍塌与伪周期性**：放大观察发丝末端、织物编织纹、远景树叶、复杂阴影交界处，是否存在扩散模型特有的“水彩涂抹感”、粘连或规则网格伪影？
2. **物理光学与透视矛盾**：高光反射点（如眼球反光、金属反光）、投射阴影的消失点是否指向同一光源？不可读的微小背景文字是否退化成了无意义的“外星文/乱码笔画（Gibberish Glyphs）”？

### 【第四步：背景复杂度去偏护栏（Debiasing Guardrail —— 严禁误判白墙/白纸真图！）】
⚠️ **防误报铁律**：
- 绝不能仅仅因为图片背景是“大面积白墙、纯色桌面、晴朗蓝天、或干净白纸文档”就判定为 AI 生成！
- 面对简单背景图片时，你必须去寻找**正面的真实物理证据**（如纸张边缘的微小毛刺、真实镜头暗角、手机广角边缘色差、真实灰尘颗粒、墨水压痕）。只要存在真实的传感器噪点与物理压痕，即使背景极简也应判定为【真实实拍】。

### 【第五步：结构化输出你的取证报告】
请严格按以下格式输出结论：
1. **最终判定结论**：【真实实拍 (Authentic)】 / 【全图 AI 生成 (Full AIGC)】 / 【真实底图 + 局部 AI 篡改/改字 (Locally AI-Manipulated / Text-Edited)】
2. **伪造/篡改概率得分**：0% ~ 100%
3. **可疑区域精确定位**：指出画面中具体的可疑坐标或文字内容（例如：“第 2 行右侧金额数字 `980.00` 区域”）。
4. **关键微观证据链**：列出你在【字下纹理】【墨迹交叉】【噪声连续性】【物理光影】【背景复杂度去偏】五个维度观察到的具体细节证据。
```

---

### 📌 提示词模版 2：专攻「OpenCut 手写字/文档单据篡改」的精细对比提示词

> **适用场景**：当怀疑一张包含手写笔记、请假条、实验数据记录、收据发票、白板签名的真实照片被 **OpenCut** 等工具偷偷换掉了其中的几个字时使用。

```markdown
请作为文档与笔迹取证专家，对这张包含文字/手写字的图片进行【同行基准比对取证（Intra-Document Forensic Comparison）】：

已知像 OpenCut 这类 AI 无痕改字工具会保留 95% 的真实照片背景，仅框选部分文字进行“背景擦除（Inpainting）+ 笔迹风格模仿重绘”。请你不要看整张图像不像真的，而是**用图内的文字自己比自己**：

1. **建立图内真值基准（Anchor Baseline）**：
   先在图中找出最自然、与纸张纹理/折痕/横格线融合得最紧密的普通文字区域，记录其：① 墨水边缘颗粒感、② 笔画起收笔锋、③ 字下纸张底纹噪点强度。
2. **逐词异常扫描（Anomaly Scanning）**：
   重点检查具有高篡改价值的字段（姓名、日期、金额数字、结论词、签名）：
   - **看字底**：哪个词底下的纸张颜色出现轻微的“雾状平滑块”或色温偏移？哪个词跨过横格线时，横格线变淡或错位了？
   - **看笔锋与交叉**：哪个词的笔画粗细过于恒定（缺乏书写速度变化带来的飞白与顿笔墨团）？
   - **看几何对齐**：手写行通常有自然的基线起伏（Baseline Drift），哪个词排列得过于水平死板，或者没有顺应纸张表面的轻微弯曲褶皱？
3. **给出定位结论**：
   明确指出哪几个字是原图真笔迹，哪几个字疑似被 AI 局部擦除后重绘替换，并给出微观对比依据。
```

---

### 📌 提示词模版 3：「视觉提示 + 定量信号注入」协同提示词（X²-DFD + TextSleuth 混合范式）

在我们的代码 [`src/forensic_prompt_pipeline.py`](../src/forensic_prompt_pipeline.py) 中，程序会自动先算好 **SRM 噪声异常比**、**ELA 最大残差位置** 和 **RDD 去偏复杂度档位**，并将这些数字自动填入下面这段 Prompt 连同六合一热力图一起喂给视觉 AI：

```markdown
【底层信号取证预检数据（由 PixelSentinel 算子自动提取）】
- 背景复杂度分层：{complexity_tier}（边缘密度：{mean_edge_density}）
- 旋转+低通参照去偏得分（RDD Score）：{debiased_calibrated_score}
- SRM 高频传感器噪声均值 / 标准差：{srm_noise_mean} / {srm_noise_std}
- 局部块最大异常倍率（Patch Anomaly Max Ratio）：{patch_anomaly_max_ratio}x
- 高疑局部块坐标网格（Top-3 Suspicious Patch Boxes）：{suspicious_boxes}
- 元数据核验结果：{metadata_risk_flags}

【指令】
请结合上方底层信号提取出的【高疑局部块坐标】以及附带的【SRM 高频噪声残差图 / 局部放大图】，重点核查高疑坐标框内的像素（尤其是文字或物体边缘）是否存在 AI 局部擦除重绘（如 OpenCut 改字）或生成伪影，并综合给出最终真伪判定。
```

---

### 📌 提示词模版 4：免训练 CLIP 零样本对比提示词组（Zero-Shot CLIP Forensic Prompt Ensemble）

如果你想直接用本机已有的 CLIP 模型做零样本打分（完全不训练），普通做法是用 `"a real photo"` vs `"a fake photo"`，准确率极低（因为 CLIP 不懂什么是 fake）。  
参考 **DeCLIP** 与 **FatFormer** 的设计，应当使用**描述微观取证物理特征的正负提示词组（Prompt Ensemble）**计算余弦相似度均值：

- **真实相机物理特征提示词组（Authentic Physical Prompts）**：
  - `"a camera photograph with natural CMOS sensor grain and consistent ISO noise across the frame"`
  - `"authentic handwritten ink strokes bleeding naturally into continuous paper fiber texture"`
  - `"unmodified smartphone photo with sharp natural edges and authentic optical lens aberration"`
- **AI 生成与局部重绘伪影提示词组（AI Synthesis & Inpainting Prompts）**：
  - `"an AI-generated synthetic image with over-smoothed diffusion textures and unnatural lighting"`
  - `"a locally inpainted image where text or handwriting was erased and replaced with smooth synthetic strokes"`
  - `"digital forgery with inconsistent background noise vacuum and blurred paper texture under replaced text"`

> 💡 这套零样本 CLIP 取证提示词组 + 块级滑动窗口打分，已经直接集成在 [`src/forensic_prompt_pipeline.py`](../src/forensic_prompt_pipeline.py) 中，无需任何训练即可直接运行！
