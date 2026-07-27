# 条件化对称性增强:哪些「可能超过 baseline」(pull / pick / push / drawer)

- **日期**:2026-07-27
- **分支**:`meav2`
- **产出方式**:定制 deep-research workflow(4 任务本地代码 grounding + 5 路文献检索 + 对抗核验 + 逐任务综合,27 agents)。
- **依据**:理论 [[new_idea]];设计与负结果 [[design_mea_v2]]、遮挡调研 [[occlusion_aug_survey]];本会话训练结果(§1)。

---

## 0. 核心发现(一句话)

**「能超过 baseline 的条件化对称增强」不是待发现的东西——它就是本项目自己已发表的 MEA。**

> **arXiv:2508.11204 — "Multi-Group Equivariant Augmentation for Reinforcement Learning in Robot Manipulation", Hongbin Lin, Juan Rojas, Kwok Wai Samuel Au.**
> 在 **Equi-RSAC**(= 本仓 C4 等变 + RSAC)之上做**多群 / 分相位增强**:把轨迹切成 *pre-interaction(自由空间)/ interaction(接触抓取)/ termination* 三段,对「不完美」的 interaction 段**只用平凡群(trivial group = 恒等,即不旋转)**增强 → **block-pull +25%、block-pick +45%、drawer-open +40%**(对抗核验通过,已读全文确认 baseline、机制、数字)。

`new_idea.md` 的「条件化等变 POMDP」正是它的理论推广(Phase-2 CONTACT → 群退化到 SE(2)/平凡)。

**为什么最近三批 mea_v2 A/B 没能复现这个正结果**(§1 机理):它们测的是**均匀(uniform)配方**——`global` 连续旋转 + `reflect`,对**每一帧**施加同一个变换。这恰恰是与 C4 网络**冗余**的那部分。而论文真正取胜的**分相位配方(interaction 段用平凡群 = 抓取帧不旋转)从来没进过 mea_v2 的 A/B 臂**。更糟的是:design_mea_v2 §8 把 `conditional` 模式**当作**「绕 target 转夹爪的 approach-gauge」而搁置(它在夹爪居中相机下确实是 no-op)——但那**不是**论文的机制。论文的机制是**时间维上的群退化(接触帧扣掉旋转)**,是**减法、且相机安全**,跟 approach-gauge 是两回事。**mea_v2 搁置错了东西。**

---

## 1. 为什么均匀增强超不过 baseline(已核验的机理链)

| 证据 | 结论 | 引用 |
|---|---|---|
| SO(2) 数据增强会**拖累** RAD/DrQ,等变架构 > 带 SO(2) 增强的它们 | 对已内建的对称做增强 = 冗余、甚至有害 | SO(2)-Equi-RL, ICLR 2022, **2203.04439** ✓ |
| relaxed/approx 等变**仅在对称被破坏时**才领先;对称精确时与严格等变持平(无损无益) | 我们的任务全程近似 E(2) 对称 → 均匀增强没有头寸 | Approx-Equi-RL, **2411.04225** ✓;Partial-Equi-RL, **2512.00915** ✓ |
| 严格等变**无法在对称输入上破缺对称**(Curie 原理,G_φ(x) ≥ G_x);extrinsic 等变可**严格有害** | 接触相位是对称破缺处,正是等变网被迫吃 error floor 的地方 | Symmetry-Breaking-ENN, **2312.09016** ✓;General Theory, **2303.04745** ✓ |
| 夹爪居中俯视相机把夹爪钉在图心;接触帧 target 被夹爪遮挡、预填均值深度 | 对称破缺帧**同时**是相机 no-op / 遮挡帧 → 在这些帧做旋转增强双重浪费 | 本仓 env.py:112-117 / design_mea_v2 §8 |

**本会话实测(block_pull, d15, 3 seed, 同 pipeline)佐证**:均匀反射 OFF(final 0.77 / AUC 0.257)≥ 反射 ON(0.43 / 0.244);mea_v2(rot+reflect)0.27 < baseline 0.37。→ 均匀配方确实没用,与上表机理一致。

---

## 2. 能超过 baseline 的机制:分相位增强(接触帧退化到平凡群)

> **⚠️ 2026-07-27 作者(2508.11204 一作)更正,见 §9:本节把「取胜配方」写成了「分相位**刚性**旋转 + 接触帧扣旋转」,这是不完整/误导的。刚性旋转(整场景连夹爪一起转)只改变视角,gripper↔target 的 egocentric 相对位姿不变 ≡ 相机在轨迹中途转一次(弱)。论文真正的非冗余增益来自 approach 阶段**改变 gripper↔target 相对接近角 α**(收敛到同一抓取),需逐实体分割 + 深度重投影。本节的「扣接触帧」只是安全那一半。以 §9 为准。**
- **为何非冗余**:C4 网络与 seq_rot 都**对每一帧**均匀施旋转(`seq_rot.py:68` "Same for the entire history")。本配方注入的新 bit 是「接触相位是对称正则、不该被旋转增强」——把 `num_aug_episode=4` 的预算从被遮挡/no-op 的接触帧,**重新分配**到真正有信息的自由空间帧。这正是 2508.11204 的取胜机制,也是逃离 extrinsic-equivariance error floor 的操作。
- **物理/相机安全**:纯减法,不合成新帧 → 不可能穿桌、不改 reward;是「调度」而非「换锚点」→ 不受夹爪居中相机中和(它改的是**哪些帧**加群,不是**在哪转**)。
- **实现**:`SeqRotBuffer._augment_and_add_episodes` 里把单个 θ 换成逐帧 θ 数组(接触帧置 0),`set_trans_zero=True` 不动。极廉价、可隔离 A/B。

---

## 3. 逐任务候选(综合判定 + 相机核验)

综合 agent 对**每个任务的总体判定都给了 `UNLIKELY`**——但那是**相对于「已经带连续 seq_rot 的当前 pipeline」**的保守判断(近似等变理论:近精确对称 → 头寸小)。放在「本项目已有 +25/40/45% 正结果」的背景下,**分相位扣接触帧**仍是**证据最硬、最该先试**的一招。

| 任务 | 首选杠杆(有直接证据) | 该任务独有 / 正交的非冗余杠杆 | 总体判定 |
|---|---|---|---|
| **block_pull** | 分相位旋转门控(接触帧扣旋转) | 分相位**反射**(自由空间帧,**用修正后的动作标签**) | unlikely,但廉价必测 |
| **block_pick** | 分相位旋转门控(`_isHolding` 现成) | **独立扰动干扰块**(蓝色 distractor 单独 SE(2),正交于所有全局对称,穿相机) | unlikely,distractor 杠杆最有意思 |
| **block_push** | **只动干扰块 object1**(原地转 yaw,reward 无关,正交于一切) | 修正标签的**全局反射**(goal 同步镜像,`‖goal−obj2‖` 不变;旧反射负结果因动作 bug 作废,**实为未测**) | unlikely,两条都廉价 |
| **drawer_open** | 分相位旋转门控(论文直接 +40%) | 锁死抽屉 distractor 去耦;自由空间门控反射 | unlikely,增益主要来自去噪/预算重分配(drawer 全程近精确对称,论文前提在此偏弱) |

**要点**:
- **approach-gauge(绕 target 转夹爪,new_idea Phase-1)在 4 个任务上一律被夹爪居中相机中和 = no-op**(pull 已证,pick/push/drawer 综合确认)。**不要照搬**。push 的 conditional 版若要救,必须给 θ 一个**随 t→k 衰减到 0 的调度**(否则在 handoff 帧产生瞬移/非法帧),且可增强窗口只有几帧,期望收益小。
- 反射是唯一**结构性**非冗余的全局杠杆(C4 `flip_symmetry=false`),但**均匀施加已在 block_pull 实测失败**。唯一没测过的变体 = **分相位(只在自由空间)+ 修正后的动作标签**(`a[2]→-a[2], a[4]→-a[4]`,旧 runs 错翻 a[1] 且差 2θ,`sym_v2.py:95-98` 已修)。→ 低期望,但因旧结果被 bug 污染,值得作为廉价复核。

---

## 4. 一个真正正交的新杠杆:逐实体「干扰物去相关」

pick / push / drawer 都有一个 **reward 无关的第二实体**(pick 的蓝干扰块、push 的 object1、drawer 的锁死抽屉)。**单独**给它施加随机 SE(2)(保持 gripper/target/goal 不动)是**唯一正交于全部全局对称(旋转 *和* 反射)**的操作:

- **非冗余**:C4 + seq_rot + 反射都把干扰物与全场景**刚性同转** → 永远无法产生「相同 gripper/target/goal 布局、不同干扰物方位」。这是全局群到不了的构型。
- **穿相机**:干扰物**天生离图心有偏移**(不在 pivot 上),动它就改图 → 不是 no-op,连接触帧也成立(而接触帧 target 恰恰被遮挡)。
- **打在 POMDP 痛点上**:两物体在占据图里同形,「哪个是 target」是隐状态;去相关直接正则化 belief-RNN 的目标/干扰判别——这是任何旋转/反射对称都碰不到的困难。对应 new_idea 的「mask out / 独立变换无关实体」。
- **代价**:需 per-entity pc + `Occup(DummyEnv)` 重渲 + 拒绝采样(保证不出界/不重叠);push 的「原地转 yaw」变体零 off-manifold 风险,最安全先试。
- **诚实上限**:它只能教「无视干扰物」,解不了 target 身份的不可观测性;drawer 因 reset 已随机化哪侧可动,in-distribution 可能不动指标,收益更偏 OOD 鲁棒性。

---

## 5. 诚实的天花板(必须先确认再跑)

1. **当前 baseline 已经在自由空间做连续 seq_rot**(整段一个 θ)。若论文 +25/40/45% 的 baseline **不含**自由空间旋转增强,那么我们能兑现的**增量只剩「接触帧扣旋转」这一薄片**,会**小于**论文标题数字。→ **动手前先核对 2508.11204 的 baseline 到底带不带自由空间增强**(决定我们的可实现 delta)。
2. **近精确对称限制头寸**:理论预测收益集中在对称破缺处 → **block_pick(真接触 + 遮挡)> block_pull / drawer(全程近精确对称,增益主要是去噪)**。
3. 一个被核验**证伪**的子声明:2508.11204 有处把增益归于「时空双维方差」的 10K-vs-12K 消融——核验发现那个消融隔离的是 **voxel 表示**而非增强,方向与该叙述相反。→ **引用论文时用「分相位/平凡群」主结果,不要用那条时空方差叙述。**

---

## 6. 建议实验(闭环当前方向)

- **主 A/B(最该做)**:**分相位旋转门控**(接触帧扣旋转)vs 现 baseline,3 seed,先在论文三任务(pull/pick/drawer)+ push。改动 = seq_rot 逐帧 θ mask,几乎零成本、可隔离。**这条直接测「论文的取胜机制在当前 pipeline 上还剩多少可兑现增量」**,同时补上 [[training-status]] 里缺的「同 pipeline baseline」缺口。
- **次选(廉价并行)**:(a) block_push/pull 的**修正标签全局反射**(旧负结果被 bug 污染,实为未测);(b) block_pick/push 的**干扰物去相关**(唯一正交于全部对称的杠杆)。
- **不做**:approach-gauge 绕 target 转夹爪(4 任务皆 no-op)。

---

## 参考(均已对抗核验,✓=supported)

- **2508.11204** ✓ Multi-Group Equivariant Augmentation for RL(**本项目论文**;分相位/平凡群,pull+25/pick+45/drawer+40)— https://arxiv.org/abs/2508.11204
- **2203.04439** ✓ SO(2)-Equivariant RL, ICLR 2022(SO(2) 增强对等变网冗余/有害)— https://arxiv.org/abs/2203.04439
- **2411.04225** ✓ Approximate Equivariance in RL(仅对称破缺时领先;误差界 (ε_R+γρε_P)/(1−γ))— https://arxiv.org/abs/2411.04225
- **2512.00915** ✓ Partially Equivariant RL(PE-SAC/PE-DQN,按相位在等变/非等变间门控)— https://arxiv.org/abs/2512.00915
- **2312.09016** ✓ Symmetry Breaking and Equivariant NN(严格等变无法在对称输入破缺对称)— https://arxiv.org/abs/2312.09016
- **2303.04745** ✓ General Theory of Correct/Incorrect/Extrinsic Equivariance(错配对称的误差下界)— https://arxiv.org/abs/2303.04745
- **2110.10211** ✓ Learning Partial Equivariances (Partial G-CNNs) — https://arxiv.org/abs/2110.10211
- **2201.11969** ✓ Approximately Equivariant Networks for Imperfectly Symmetric Dynamics — https://arxiv.org/abs/2201.11969
- **2112.01388** ✓ Residual Pathway Priors(软等变约束)— https://arxiv.org/abs/2112.01388

---

## 9. 作者更正(2026-07-27):取胜杠杆是「相对接近角 α」,不是刚性旋转

**来源**:用户 = arXiv:2508.11204 一作,当面更正。可视化 `bash/viz_approach_angle.py` →
`context/plan/approach_angle_<task>.png`(4 任务)。

### 更正内容

§2、§3(row3)、§6 把取胜配方写成「分相位**刚性**旋转:自由空间帧整场景转 θ、接触帧扣旋转」。
**这是弱增强,不是论文的机制。** 关键区分(egocentric 相对接近角 Δα = target 方位 − 夹爪朝向):

| | 自由空间帧做什么 | gripper↔target 相对位姿 | 等价于 | 强度 |
|---|---|---|---|---|
| **刚性旋转**(我之前 row3 / `candidate_aug_*.png`) | 整场景(**含夹爪朝向**)一起转 θ | **不变**(Δα≡0) | 相机在轨迹中途转一次(实际很少) | **弱**,只提升相机旋转鲁棒性 |
| **接近角增强**(论文 2508.11204) | **只转 target 相对夹爪的方位** α(t),夹爪朝向不动;α(t)→0 收敛到同一抓取 | **改变**(Δα=α(t)) | 同一目标、**全新接近轨迹** | **强**,非冗余真信息 |

`approach_angle_block_pick.png` 底部面板是判据:刚性(蓝)与真实(绿)Δα **完全重合在 0** →
证明刚性只是换视角;接近角(红)从 +55° 递减、接触帧收敛到 0 → 注入了新的相对接近角。

### 这纠正了 design_mea_v2 §8 的「no-op 搁置」

design_mea_v2 §8 把 conditional 的「approach 局部 gauge」当作 no-op 而搁置,转向 global+reflect。
**真正原因是实现错了**:`sym_v2.py` conditional 分支 `frame_tf(t<k)={"gripper":T_grip}` 转的是
**夹爪**(绕 target 锚点)——夹爪在图心,转它 → 夹爪离心 = **off-manifold**(真实观测夹爪永远居中),
所以看似 no-op。**正确做法是反过来**:夹爪(朝向)不动、把 **non-gripper 实体绕图心(=夹爪)转 α(t)**,
target 方位改变而夹爪居中 = **on-manifold**,egocentric 接近角真的变了。论文正是这样(分割 target+gripper、
深度重投影),并拿到 +25/45/40%。

### 落地(取代 §6 主实验)

- **APPROACH-phase 相对接近角增强**:相位 t<k(自由空间)对 **non-gripper 实体**绕 pc 原点(=夹爪)
  施 `se2_about(0, α(t))`,`α(t)=α₀·(k−t)/k` 递减到 0;夹爪点云不动;`Occup(DummyEnv)` 逐实体重渲。
  接触/终止帧恒等(§2 的扣接触帧那一半仍保留,作为安全项)。
- **动作标签**:approach 帧的 (dx,dy)/yaw 需按「target 相对夹爪转了 α(t)、夹爪不动」做**相对**共轭
  (不是 §2 的全局共轭)——这是最需要在真机轨迹上核对的一环(对应 2508.11204 的分割+重投影细节)。
- **候选优先级更新**:approach-angle 相对增强 = 首选(有作者论文正结果);分相位刚性旋转降级为「仅相机鲁棒性」;
  反射维持次选(修正标签复核);干扰物去相关维持(但逐实体 pc 遮挡噪声大,见 `viz_candidate_aug.py` 发现)。
