# MEA (arXiv:2508.11204) Novelty vs SEIL / MILES — rebuttal notes

- **日期**:2026-07-27  ·  T-RO 投稿 25-1536,Reviewer 7 提出重合
- **产出**:定制 workflow 深读 SEIL/MILES 全文 + 扫描其它先例(agent 已 WebFetch 原文)
- **一句话结论**:**裸 claim「first non-isometric augmentation for multi-step manipulation」守不住,必须收窄到"具体机制的合取 + RL/POMDP 理论"。** MILES 是弱重合(易反驳),SEIL 是硬重合(必须正面处理,建议加为 baseline)。

---

## 0. 诚实的暴露评估

Reviewer 7 只引了 MILES/SEIL,但「first」claim 的暴露面**比他指出的更大**。以下先例合起来覆盖了 MEA 的各个碎片:

| 先例 | 覆盖了 MEA 的哪一块 | 但做不到 |
|---|---|---|
| **SEIL** (ICRA'23, 2211.00194) | 等变网 + 正交投影 + **接触帧不增强** + 自由空间**改变相对位姿** | RL/POMDP;结构化群变换;多群理论 |
| **Fourier Transporter** (ICLR'24, 2401.12046) | **对 pick/place 用两个独立群**(= "multi-group / 独立变换物体")| 单步开环 BC;架构等变而非轨迹增强;无相位/无时序内相对角 |
| **Equivariant Transporter** (RSS'22, 2202.09400) | pick×place 积群 SO(2)×SO(2),独立群 | 同上,单步 |
| **MimicGen / DexMimicGen** (CoRL'23, 2310.17596) | **多步、接触丰富任务的分段 SE(3) 变换** | 每段**刚性**移动(不改段内相对位姿、不衰减收敛到同一抓取);IL 数据生成非 RL;无等变理论 |
| approach-angle 随机化(抓取综述 2207.02556) | 「改变自由空间接近角」本身是老做法 | 无原则随机、非群结构、无 POMDP 不变性理论 |

→ **不能再说「first non-isometric / multi-group / 分相位」**——这三个词各自都有在先工作。**「非等距 / 多群」这个术语本身也不是 MEA 首创**(Wang/Platt 等变操作线里早有 "two independent groups on object and goal" 的措辞)——不要 claim 造了这个词。

---

## 1. MILES(弱重合,可正面反驳)

**MILES = 单示教、真机自监督 BC**:从 1 条示教,在每个 waypoint 附近(~4cm/4°)把末端扰动到随机位姿、物理执行**直线回归轨迹**并录 RGB+力反馈,训练普通 ResNet-18+LSTM BC。**完全没有等变/群论,连 "isometric" 这个词都不出现。**

**审稿人把它称作 "non-isometric augmentation" 是不准确的**——它没有任何群作用。唯一重合是**动机/结果层面**(数据高效、训练分布里有偏离示教的 gripper↔物体相对位姿),但那是**真机小范围局部探索的副产品**,不是设计的变换。

**反驳要点(可直接写进 response)**:
1. MILES 是 **BC + 真机自监督采数**,MEA 是 **RL(Equi-RSAC)+ POMDP/RNN**,增强进的是 replay buffer 的**离线解析变换**——MILES 每个数据点都要**真机执行**,无法离线增强已有 buffer。
2. MILES **零群论 / 零等变 / 无 isometric 框架**;MEA 核心是 group-invariant MDP → 多群 POMDP 不变性定理。
3. MILES **从不独立变换物体与夹爪**(物体固定、只扰末端),没有 multi-group;其相对位姿散布是**无收敛调度的局部物理扰动**,不是 MEA 的**衰减接近角 α → 收敛到同一抓取**的设计变换。
4. MILES 的自身卖点是"免去人工 reset 的自主采数",与 MEA 的增强变换**正交**。

→ MILES 应当**引用 + 一句话区分**(paradigm/机制/理论都不同),无需实验对比。

---

## 2. SEIL(硬重合,必须正面处理)

**SEIL = 少样本 O(2)-等变 BC**,两个增强通道:
- **Transition Simulation (TS)**:点云 C **固定**,把末端位姿加各向同性高斯噪声 `p̄ = p' + (a⁻¹+ε), ε~N(0,σ=0.4)`(含 Δθ),用**正交投影 Π 重渲**"仿佛夹爪在 p̄"的观测;校正动作指回示教路径(DAgger 式回归漏斗)。**只在无接触且未夹持的 transition 上做**(validity filter),否则复制原 transition。
- 另一路:每对 (obs,action) 做 **64× 全局 SO(2)** 刚性等距增强。

**和 MEA 的真实重合(要如实承认)**:
1. 都用**正交投影重渲**固定点云(同一渲染 trick),不重跑物理;
2. 都用**平面对称的等变网**;
3. 都**在接触/夹持帧扣掉增强**,只增强自由空间(← Reviewer 7 的 Weakness 2,属实);
4. **TS 确实在自由空间改变 gripper↔物体相对位姿(含相对接近角)**,且和 MEA 一样**收敛回同一抓取**。

**⚠️ 我起初以为的硬差异"SEIL 只做段内刚性变换、不改相对位姿"是错的——SEIL 确实改相对位姿。** 所以 novelty 不能落在这。

**可辩护的差异(收窄到这几条)**:
| 维度 | SEIL | MEA |
|---|---|---|
| 范式 | **BC / 监督 MSE**(RL 仅列为 future work) | **RL(Equi-RSAC,model-free+model-based)+ POMDP/RNN belief + replay** |
| 相对位姿变换的**性质** | **无结构各向同性高斯噪声**(DAgger 回归漏斗,动作是"纠错"指回示教) | **结构化群元 α**(绕定向夹爪转 target 方位、**衰减到 0 收敛同一抓取**),增强轨迹本身是"新接近角下的最优示教" |
| 多群 | 否——只扰夹爪、场景固定,单一扰动 | **对不同实体施不同群元**(夹爪 identity / 物体旋转),多群非等距 |
| 理论 | 无 | **多群非等距 POMDP 的 value-invariance / policy-equivariance 定理** |

**关键动作**:Reviewer 7 明确要求"对比 SEIL 的增强或给理由不比"。**最有说服力的做法是把 SEIL 的 TS 增强移植进 RL buffer 作为 baseline**,证明"结构化群-α + 多群"在 RL 设定下优于"高斯 TS 噪声"。若不做对比,必须给出**强**理由(如 TS 是 BC 专用的纠错漏斗、动作语义不同——它产的是纠错动作而非同任务的最优新示教)。
- ⚠️ 风险自评:若移植后 SEIL-TS ≈ MEA,novelty 会塌缩到"理论 + RL 框架"。上手前先想清楚这条。

---

## 3. 收窄后的、可辩护的 novelty 陈述(可直接替换原文)

> 现有等变增强要么(a)对整场景施**单一全局等距**群(SO(2)-Equi-RL),要么(b)对 pick/place **独立**施刚性群但仍是**单步、段内刚性**(Equivariant / Fourier Transporter),要么(c)按段施刚性 SE(3) 合成多步示教但**不改段内相对位姿、无收敛**且用于 **IL 数据生成**(MimicGen),要么(d)在**无接触段**用**无结构高斯扰动**改变相对位姿以做 **BC 纠错**(SEIL)。
> **MEA 是首个将下述三者同时实现的方法**:(i) 在**单个时间步内**以**结构化群元**改变 gripper↔物体**相对接近角 α 并衰减收敛到同一抓取**(非刚性、非高斯);(ii) 置于 **model-free RL 的 replay buffer、POMDP/RNN belief** 之下(非 BC);(iii) 由**多群非等距 POMDP 的 value-invariance / policy-equivariance 定理**支撑,并对接触相位显式退化到平凡群。

即:**不 claim「首个非等距/多群」,改 claim「首个把 *衰减的、分相位的、逐实体* 非等距增强用于 model-free RL 下的接触丰富多步操作,并配多群 POMDP 不变性保证」**。

---

## 4. Rebuttal 待办清单

1. **补引 + 区分**:SEIL、MILES、Fourier Transporter、Equivariant Transporter、MimicGen/DexMimicGen(Reviewer 7 只点了前两个,主动补齐余下,展示 awareness)。
2. **删「first non-isometric」裸表述**,替换为 §3 的收窄陈述。
3. **加 SEIL baseline**(TS 移植进 RL buffer)或写强理由不比 —— 直接回应 Weakness 1/Suggestion 1。
4. **术语更正**(呼应 Reviewer 2/3):"non-isometric" 与 "trivial representation" 用法被多位审稿人质疑;要么严格定义(多群 = 不同实体不同群元 → 联合作用非等距),要么换词(如 "multi-frame / per-entity group augmentation")。Reviewer 7 还质疑有值域上限的变换集**不闭合 → 不是群**,需在定义上回应。
5. **定位性引用**(帮框定 novelty,非威胁):
   - 潜在/部分对称的危害与"错配对称有害"—— NeurIPS'22 **2211.09231**(为"接触帧扣增强"提供原则性依据);
   - 等变 RL under POMDP —— **2408.14336**(MEA 的多群 POMDP 是其非等距推广)。

---

## 参考(agent 已读原文;引用前请复核 arXiv 号)

- SEIL — https://arxiv.org/abs/2211.00194 (ICRA 2023)
- MILES — https://arxiv.org/abs/2410.19693 (CoRL 2024; proceedings v270 papagiannis25a)
- Fourier Transporter — https://arxiv.org/abs/2401.12046 (ICLR 2024)
- Equivariant Transporter Net — https://arxiv.org/abs/2202.09400 (RSS 2022 / IJRR 2024)
- MimicGen — https://arxiv.org/abs/2310.17596 (CoRL 2023);DexMimicGen — https://arxiv.org/abs/2410.24185
- Effectiveness of Equivariant Models w/ Latent Symmetry — https://arxiv.org/abs/2211.09231 (NeurIPS 2022)
- Equivariant RL under Partial Observability — https://arxiv.org/abs/2408.14336
- 抓取 approach-angle 随机化综述 — https://arxiv.org/abs/2207.02556
