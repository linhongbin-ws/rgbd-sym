# MEA v2 → ICLR:让 novelty 与「超过 baseline」是同一个机制

- **日期**:2026-07-27  ·  分支 `meav2`
- **背景**:v1(arXiv:2508.11204)只挂 arХiv、未 formulate contextual condition;v2 补上「Conditionally Equivariant POMDP」([[new_idea]])重投 ICLR。
- **依据**:本会话训练结果([[training-status-meav2]])、机制更正([[mea-paper-and-winning-mechanism]] §9)、先例扫描([[novelty_rebuttal_notes]])、近似/部分等变理论。

---

## 0. 核心论点(一句话)

> **不要让「novelty = 理论」和「涨点 = 增强」分家。让二者是同一件事:context-conditioned(gauge)equivariance——有效对称群/锚点随推断出的接触相位 context 变化,并把这个 context 喂进 policy/critic;它既是 v2 相对 v1 与所有先例的新意,也正是能超过全局等变 baseline 的原因。**

**为什么这能同时成立**:baseline(C4 等变 + seq_rot)对**每一帧**强加**同一个全局对称**,包括接触/负载相位——而那里对称已被物理破坏。对破缺处强加对称 = **extrinsic equivariance,有可证的误差下界、甚至有害**(2303.04745;Curie 原理 2312.09016)。**一个按 context 放松/重锚等变的网络能逃离这个下界 → 这就是相对 baseline 的头寸**;而均匀增强做不到(它还在同一个全局群里打转,与 C4 冗余,2203.04439,实测也没涨)。→ **novelty(把对称条件化)= 涨点来源(逃离 extrinsic-equivariance floor)。**

---

## 1. 为什么这是新的(相对 v1 与全部先例)

| 工作 | 对称是 | 缺的正是 v2 的点 |
|---|---|---|
| **v1 (MEA)** | 多群**数据增强**,但**未 formulate context**,网络仍单一全局群 | context 的形式化 + 喂进网络 |
| SEIL / Fourier&Equi-Transporter | **静态**群(单一或 pick/place 两个固定群) | 群随 context **动态**变化 |
| MimicGen | 分段刚性 SE(3),IL 数据生成 | 等变理论 + context 网络 |
| Equivariant RL under POMDP (2408.14336) | **单一全局**群 + POMDP | 群的**非等距 / 上下文条件化** |
| Partial/Approx-Equi RL (2512.00915, 2411.04225) | 学习式部分等变,generic control | **物理接触相位驱动**的 gauge + 匹配的数据增强 + 操作 POMDP |

→ **v2 的独占交集**:*由推断的接触/相位 context 决定的、群+锚点+表示都随之变化的 gauge equivariance,同时落在(a)数据增强侧与(b)网络条件化侧,并配多群非等距 POMDP 的不变性定理。* 没有先例同时占这几点。**这正是 ICLR 要的「一个干净的新机制 + 理论」。**

---

## 2. 为什么能超过 baseline(机制 = §0 的涨点)

两条**耦合**的杠杆,但**主涨点在网络侧(b)**:

- **(a) 数据侧(必要不充分)**:分相位增强——自由空间做**结构化相对接近角 α**(绕定向夹爪转 target、衰减收敛同一抓取,§9),接触帧退化平凡群。**注意:本会话的 v2-aug-only 实测没超过 baseline**(反射没兑现、mea_v2<baseline)。→ 光靠增强不够,且 ≈ SEIL,novelty 也薄。
- **(b) 网络侧(充分,且是真 novelty)**:把推断的 context c 喂进 equivariant policy/critic(design_mea_v2 §3 的 layer1 gauge 通道 / new_idea 的 MoEE gauge estimator)。让网络在自由空间保持等变、在接触/负载相位**放松或重锚**等变 → **分离"同一抽象变换对应矛盾动力学"(new_idea §5 的 high-variance/不收敛根因)→ 逃离 extrinsic floor**。**这是 v1→v2 真正的增量,也是最可能把负结果翻正的一招——但目前 UNTESTED。**

---

## 3. 一次证明「novelty=涨点」的消融阶梯(ICLR 因果叙事)

| 臂 | 说明 | 检验 |
|---|---|---|
| **B0** baseline | Equi-RSAC,全局群,无 context | 参照 |
| **B1** v2-aug-only | 分相位增强,网络不变 | 数据侧单独有没有用(本会话≈无)|
| **B2** v2-context-only | c 喂网络,**不加额外增强** | **架构侧 gauge 单独的增益(关键)** |
| **B3** v2-full | context + 匹配增强 | 完整方法 |

**目标叙事**:B3 ≫ B0,且增益**可归因于 context(B2 相对 B1 的跳变)**,而非更多增强 → **因果证明「把对称条件化(v2 的新意)才是超过 baseline 的原因」**。这直接回应 v1 审稿人的"就是换个增强"质疑,也把 novelty 钉死在 context 上而非增强上。

---

## 4. 诚实的风险(必须先 de-risk,否则没论文)

1. **B2(context 进网络)从未测过**;整篇论文押在它上。**第一优先级 = 尽快跑 B2**,若它也不涨,方向要改(或换更强的 gauge:MoEE / 显式重锚,而非单标量通道)。
2. **sim 头寸可能太薄**:本仓 sim 在接触帧用 **GT mask 预填**遮挡([[mea-paper-and-winning-mechanism]] / env.py:112-117),且当前任务**全程近精确对称**(pull/pick/drawer)→ 对称破缺被抹平 → context 杠杆在现有 sim 任务上增益可能很小(**这正解释了本会话的 null 结果**)。
   - → **选/加一个 gauge 真正剧烈切换的任务**:place-to-goal(锚点 object→goal + 负载动力学切换,new_idea Phase-3)、或带**障碍/接触丰富**破坏旋转对称的任务。这同时回应 Reviewer 3「任务太简单」。
3. **归因风险**:审稿人会问"涨点来自 context 还是来自更好的增强"。→ 消融阶梯 B1 vs B2 必须干净隔离;context 通道要做成**可开关**、其余全同。
4. **真机 vs sim**:context/gauge 的价值在真机(无 GT 预填、真遮挡、真接触)更大;若 sim 头寸薄,**真机小实验**可能是 novelty 的关键证据(呼应 v1 审稿人对真机的偏好)。

---

## 5. 建议路线(按优先级)

1. **先跑 B2(context-only,最小版)**:c = 可观测相位信号(contact / gripper_close / z)写入 layer1 常数平面,网络其余不变,**先在 place-to-goal 或 block_push(有 goal、gauge 会切)上**,3 seed,AUC 判据。—— 这是**一步定生死**的 de-risk。
2. B2 若有信号 → 补 B0/B1/B3 阶梯 + 加一个 gauge 剧烈切换的任务,凑因果叙事。
3. B2 若无信号 → 升级 gauge(MoEE / 显式重锚 / 学习式 gauge estimator c_θ(b)),或把主场景移到真机。
4. **理论**:把 new_idea 的 Contextual Equivariant POMDP 定理(per-context value-invariance / policy-equivariance)与 partial-equivariance 的误差界(2411.04225 的 (ε_R+γρε_P)/(1−γ))对接——**用它形式化"逃离 extrinsic floor"的增益**,让理论和实验说同一件事。
5. **定位**:引 2211.09231(错配对称有害)、2512.00915 / 2411.04225(部分等变)、2408.14336(等变 POMDP)、SEIL/Transporter/MimicGen(见 [[novelty_rebuttal_notes]])——把 v2 摆成"**物理 context 条件化的部分等变**,首个同时做数据侧+网络侧+操作 POMDP 理论"的位置。

---

## 6. 一句话 ICLR 定位(可作 abstract 骨架)

> 全局等变 RL 在多步接触操作里把**单一对称**强加于**所有相位**,在接触/负载相位对称被物理破坏处付出 extrinsic-equivariance 代价。我们提出 **Contextual Equivariant POMDP**:有效对称群、锚点与表示是**推断接触相位 context 的函数**,既作为**分相位数据增强**、又作为**context 条件化的等变 policy/critic**,并证明 per-context 的 value-invariance / policy-equivariance。它在**对称被破坏的相位**恢复表达力从而超过全局等变 baseline——这是均匀增强与静态对称的先例(SEIL / Transporter / MimicGen / 等变 POMDP)都留在桌上的头寸。
