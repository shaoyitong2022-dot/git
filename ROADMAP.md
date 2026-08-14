# ROADMAP — AI4Science × Quantum ML 博士申请规划

> 作者：中山大学物理系本科生（准大二 → 大四申请）
> 目标：Fall 2029 入学 PhD，方向 AI4Science / 量子信息 / Quantum+AI 交叉
> 目标院校：HYPSM、UCB、ETH、港三（HKU/CUHK/HKUST）、新二（NUS/NTU）、TUM、Caltech、Columbia、Penn 等藤校

---

## 0. 总体定位与策略

**核心叙事（贯穿整个申请包）**：「物理训练打底 + 量子信息视角 + 现代深度学习工具，用 AI 求解量子多体与分子科学问题」。
所有项目、选课、套磁、推荐信都向这条主线收敛，避免"什么都做过一点"的散点画像。

**三条铁律**：
1. **代码可跑 > 什么都碰**：每个项目仓库必须有可一键复现的 demo + README + 一张结果图。招生委员会点开能跑就赢一半。
2. **先复现后原创**：Phase 4 之前不要追新意，先把经典结果复现到位（数字对得上），再谈扩展。
3. **早进组**：大三上（最早大二暑假）必须进一个本校或海外教授的组。项目和论文最好挂在课题组里，有导师背书。

---

## 1. 四阶段时间线总览

| 阶段 | 时间 | 主题 | 本仓库目录 | 核心产出 |
|------|------|------|-----------|---------|
| Phase 1 | 大二上 (2026 秋) | 量子机器学习基础 | `01-quantum-ml/` | 3-4 个 QML 复现 + 1 篇 demo 笔记 |
| Phase 2 | 大二下 (2027 春) | PINN 与神经量子态 | `02-pinn-nqs/` | NQS 复现 + PINN 解 Schrödinger |
| Phase 3 | 大三上 (2027 秋) | 等变神经网络 / 分子 | `03-equivariant-nn/` | NequIP/MACE 复现 + 进组 |
| Phase 4 | 大三下-暑假 (2028) | 论文复现 + 原创扩展 | `04-paper-repro/` | 1 篇 workshop/预印本 + 海外科研 |
| 申请季 | 大四上 (2028 秋) | SOP / 套磁 / 推荐信 | — | 12 月前提交全部申请 |

---

## 2. Phase 1 — 量子机器学习基础（大二上，2026 秋）

### 2.1 技能目标
- **数学**：Dirac 记号与 Hilbert 空间、密度矩阵、张量积、偏迹、谱分解（重读 Griffiths《量子力学》前 4 章 + Nielsen & Chuang 第 1-2 章）。
- **量子计算**：量子门集合、Grover/QPE 原理、变分量子算法框架（VQA）。
- **ML 基础**：梯度下降、自动微分（PyTorch autograd）、交叉熵/MSE 损失。
- **工具**：PennyLane（主）、Qiskit（辅），会用 `lightning.qubit` 设备。

### 2.2 项目（对应 `01-quantum-ml/`）
1. **VQE 求解 H₂ / LiH 基态**（起步脚本已生成：`vqe_h2.py`）
   - 用 STO-3G 基组构造分子 Hamiltonian（`qiskit-nature` 或手写）→ UCCSD / Hardware-efficient ansatz → 优化 → 与 FCI 对比。
   - 产出：能量随键长曲线 + 误差表。
2. **变分量子分类器（VQC）**：Iris / 双月牙数据集， amplitude / angle encoding，对比经典 SVM。
3. **量子核 SVM**：复现 PennyLane 量子核 demo，测 kernel alignment，与 RBF 核对比。
4. **量子 GAN（qGAN）**：学习一个简单离散分布（如高斯混合）。

### 2.3 必读论文
- Peruzzo et al., *A variational eigenvalue solver on a photonic quantum processor*, Nature Comm 2014（VQE 原始论文）。
- Benedetti et al., *A generative modeling approach for benchmarking quantum learning*, Sci Rep 2019.
- Schuld, *Supervised quantum machine learning models are kernel methods*, PRR 2021.
- Huang et al., *Power of data in quantum machine learning*, Nature Comm 2021.

### 2.4 里程碑（12 月底）
- 仓库 `01-quantum-ml/` 有 3 个可跑脚本 + 结果图 + README。
- 能脱稿讲清楚 VQE 数学推导。
- 完成 PennyLane 官方 demo 的 50%（`pennylane.ai/qml/demos`）。

---

## 3. Phase 2 — PINN 与神经量子态（大二下，2027 春）

### 3.1 技能目标
- **PDE 数值**：有限差分法基线（用来对比 PINN）、Schrödinger 方程本征值问题。
- **变分原理**：Rayleigh-Ritz、变分 Monte Carlo（VMC）。
- **JAX**：`jit`/`vmap`/`grad`，用 NetKet 的 sampler。
- **现代 DL 训练**：学习率调度、AdamW、梯度 clipping、checkpointing。

### 3.2 项目（对应 `02-pinn-nqs/`）
1. **PINN 解 1D Schrödinger 方程**：谐振子 + 有限深势阱，硬约束 + 软约束两种边界处理。
2. **PINN 解 2D 热方程**：条件输入（参数化 PDE）。
3. **神经量子态（RBM）解横场 Ising 模型**：复现 Carleo & Troyer 2017 关键图（基态能量 vs 横场强度）。
4. **NQS 解 Heisenberg 链**：能量 + 自旋-自旋关联函数，与精确对角化（ED）对比。

### 3.3 必读论文
- Raissi et al., *Physics-informed neural networks*, JCP 2019（PINN 原始）。
- Carleo & Troyer, *Solving the quantum many-body problem with artificial neural networks*, Science 2017（NQS 开山）。
- Pfau et al., *Ab initio solution of the many-electron Schrödinger equation with deep neural networks*, PRL 2020（FermiNet）。
- Sharir et al., *Deep learning based quantum state approximation*, 2020。

### 3.4 里程碑 + 进组
- 复现 Carleo-Troyer 的 Ising 能量曲线（误差 < 1e-3）。
- **大二下结束前必须联系本校做 AI4Sci / 量子计算的导师**（物理系或计算机系均可），争取大三上进组。
  - 中大可关注：物理学院的凝聚态理论、量子信息方向；AI 学院做科学机器学习的组。

---

## 4. Phase 3 — 等变神经网络（大三上，2027 秋）

### 4.1 技能目标
- **群论基础**：SO(3)、不可约表示（irreps）、Clebsch-Gordan 系数。
- **图神经网络**：消息传递、邻接、等变消息传递。
- **分子动力学**：MD17 数据集、力场训练、能量/力守恒（梯度对输入求导）。
- **产出意识**：开始写实验报告（LaTeX），习惯记录超参 + 复现脚本。

### 4.2 项目（对应 `03-equivariant-nn/`）
1. **e3nn 入门**：在甲烷上做等变卷积，理解 irreps 张量。
2. **NequIP 训练 MD17**：乙醇 / 阿司匹林，对比 MAE 与论文 Table。
3. **等变扩散模型（EDM-lite）**：复现 EDM 在 QM9 上的小分子生成（采样几十个分子）。
4. **MACE benchmark**：在自选小数据集上对标 MACE。

### 4.3 必读论文
- Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*, Nature Comm 2022（NequIP）。
- Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks*, NeurIPS 2022。
- Höllrigl et al. / Hoogeboom et al., *Equivariant Diffusion for Molecule Generation*, ICML 2022。
- Thomas et al., *Tensor field networks*, 2018（e3nn 理论源头）。

### 4.4 里程碑
- 至少 1 个 NeurIPS workshop 级别的复现（不一定投稿，但质量到位）。
- **进组满 3 个月**，开始有自己负责的子问题。

---

## 5. Phase 4 — 论文复现 + 原创扩展（大三下-暑假，2028）

### 5.1 策略
- 选 1 篇近 2 年的顶会/顶刊论文（QML / NQS / 等变 NN / 量子纠错+ML 任一），先 100% 复现图，再提出 1 个可验证的扩展。
- 扩展不一定要"大"，但必须自洽、有 ablation、能讲故事。

### 5.2 候选方向（按优先级）
1. **量子纠错 + ML**：神经解码器（surface code），最贴合"量子信息 + AI"，且 ETH/Caltech 极看重。
2. **NQS 新架构**：把 Transformer/扩散思想引入 NQS，复现 + 在新 lattice 上测试。
3. **等变势函数**：MACE 在固态或材料体系的新应用。

### 5.3 海外科研（大三暑假，2028 6-8 月）
- **目标**：拿到 1 封海外强推。优先级：ETH > 港新 > 美国藤校暑研项目。
- 途径：直接套磁带论文/代码附件；申请官方暑研（Amgen、MIT MISTI、ETH 等暑期项目）。
- 套磁信三件套：1 段自我介绍 + 1 段对该组最近论文的理解 + 1 段你能贡献什么（附 GitHub 链接）。

---

## 6. 申请季（大四上，2028 秋，12 月截止）

### 6.1 申请包构成
| 项目 | 权重 | 准备要点 |
|------|------|---------|
| 推荐信 ×3 | 极高 | 2 封科研导师（1 中大 + 1 海外）+ 1 封课程/项目导师。每封信都要有"具体细节+对比" |
| SOP | 高 | 800-1000 词，主线统一：物理→量子信息→AI4Sci，点出 2-3 个具体研究经历 |
| 论文/预印本 | 高（如有） | 哪怕是 workshop/poster 也写上，arXiv 编号优先 |
| GitHub 作品集 | 中高 | 本仓库即作品集，保证 4 个 phase 都能跑 |
| GPA / 排名 | 高 | 物理核心课 + 数学课保持高分，排名前 10% |
| GRE | 中 | 部分藤校仍要；数学最好 168+（量子方向数学权重大） |
| 英语 | 高 | TOEFL 100+（口语 23+），港新要雅思 7.0 |

### 6.2 目标院校分组与策略
| 梯队 | 院校 | 策略 |
|------|------|------|
| 冲刺 | MIT、Harvard、Stanford、Princeton、Caltech | 必须有海外强推 + 论文；套磁要早（8 月起） |
| 主申 | UCB、Columbia、Penn、ETH、Yale | 重点投入；ETH 偏好有明确研究方向者 |
| 保底 | 港三（HKU/CUHK/HKUST）、新二（NUS/NTU）、TUM | 早申（港新 9-10 月就开），保底兼冲刺奖学金 |

### 6.3 关键教授方向参考（套磁前先读其近 2 年论文）
- **QML**：Nathan Wiebe (UBC/ Toronto)、Maria Schuld (Stellenbosch)、Swee Goh (Waterloo)。
- **NQS / AI4Sci**：Giuseppe Carleo (EPFL)、Frank Noé (FU Berlin)、David Pfau (DeepMind)。
- **等变 NN**：Tess Smidt (MIT)、Boris Kozinsky (Harvard, NequIP)、Noa Maranan (Cambridge, MACE)。
- **量子纠错+ML**：Torsten Hoefler (ETH)、Steven Girvin (Yale)。

---

## 7. 风险与备选

| 风险 | 应对 |
|------|------|
| RTX 3050 4GB 显存不够跑大模型 | Phase 3 起用 Colab Pro / 校内 HPC / 申请 ASU/SDSC 免费学生算力 |
| Python 3.10 在 Phase 3 后可能偏旧 | Phase 3 前评估是否升级到 3.12（新建 conda env，不动 pytorch_env） |
| 没进到海外组 | 大三暑假去港新/中科院做暑研，质量同样被认可 |
| 论文未产出 | 至少保证 1 篇高质量复现 + workshop poster，SOP 里讲清"为什么没发"（科研周期长）|

---

## 8. 每周节奏建议

- **周一-周五**：白天上课，晚上 1.5h 编码 + 0.5h 读论文。
- **周六**：3-4h 深度项目时间（跑实验、调参）。
- **周日**：写周报（记进 `04-paper-repro/notes/` 或个人 blog），整理本周 commit。
- 每月：在 GitHub 维护一个 `progress.md`，列出本月完成项 + 下月目标（招生时能展示持续力）。

---

## 9. 立即可做（本周）

1. [ ] 运行 `python 01-quantum-ml/vqe_h2.py`，理解每一行。
2. [ ] 注册 PennyLane 示例账号，跑通 [VQE tutorial](https://pennylane.ai/qml/demos/tutorial_vqe)。
3. [ ] 安装 VS Code + Jupyter + Python 扩展，配置 conda env 解析器。
4. [ ] 在本仓库建 `progress.md`，写下本周目标。
5. [ ] 读完 Griffiths 第 1 章 + Nielsen-Chuang 第 1.1-1.3 节。
