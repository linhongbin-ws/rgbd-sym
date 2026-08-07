# MEA v2(diffusion 路线)对抗证伪 + pivot

- **日期**:2026-07-28 · 分支 `meav2`
- **来源**:对抗核验 workflow(4 攻击线 × 每条坐实证伪 + 综合)。
- **一句话**:**方法按现状站不住;3 条臂里 2 条死(global 冗余、approach 冗余且被 EquiDiff 作者自己的表示法碾压);headline 的「方赢圆平」证伪测试本身是坏的。只有 1 个核心活着:离散物体点群(C4)ENGAGE 增强。必须大改并 pivot 到这唯一核心。**

---

## 1. CONFIRMED 威胁(按杀伤力排序)

**T1 —（headline-killer）「方赢圆平」控制实验不诊断。** ENGAGE 是**局部**旋转(绕物体/孔,不是整场景),方、圆**都**注入新 (obs,action) → 按 MEA 自己「非整场景旋转=有信息」的判据,圆也该**赢**、而不是平。雪上加霜:robosuite 圆螺母**带偏置手柄**(`handle_site` ~0.06m)→「圆=SO(2)」在物体/抓取层面**是假的**(只有孔对称);且**没有现成 single-Round MimicGen 数据**(Nut Assembly 是双销长程,另一个 regime)。→ **唯一用来隔离机制的实验,隔离不了机制。**

**T2 — approach 臂死:冗余 + 用错工具。** 绕固定物体转夹爪 ≈ 绕物体中心的全局旋转,落在 EquiDiff 的 C8 轨道内(唯一新内容是 C8 45° 之间的连续填充,微乎其微)。**MEA 自己的 RL 结果早就发现这个 washout**(design_mea_v2 §8.2:绕夹爪旋转≈网络内建等变=负结果机理)。更糟:**EquiDiff 作者自己发了 `arXiv:2505.13431`「Practical Guide for Symmetry in Diffusion Policy」——eye-in-hand + relative/delta 动作让这种「目标相对运动」按构造就不变**(精确、非近似、不烧容量)。用增强去让全局网重学它 = 严格被支配。

**T3 — global 臂只能当负控制。** 整场景旋转 = `2203.04439 §H.4` 证明会**拖累**已等变策略的那种冗余增强。repo 早标了 `global(redundant)`。→ 论文全部重量压到 ENGAGE。

**T4 — 自我 scoop:「分相位多群增强」就是 MEA v1(2508.11204,同作者)。** v2 的 delta 只有:(i) **离散物体点群**(v1 只用连续 SO(2),从无离散物体对称);(ii) 物体对称 keyed 的可证伪预测;(iii) BC/diffusion vs RL。不点明会被当增量。

### 扛住了(但论文里必须主动写明,否则被 raise)
- **C4 ⊂ C8 冗余 — REFUTED**:增强是**局部场景**旋转,在 C8 轨道**之外**。必须在文里明说「keyed 增强不是整场景群元」。
- **diffusion 本就多模态 — REFUTED**:MimicGen 保持相对位姿 → demo 在插入 yaw 上**单模态**,另外 3 个 C4 朝向**out-of-support**。**把它变成正面证据**(画 demo 插入 yaw 直方图=单峰;未增强 EquiDiff rollout 从不产生另 3 个朝向)。
- **手柄破坏 C4 — REFUTED**:合法性由**孔/销点群 + 位置成功判据**决定,不是螺母整体。但**别叫螺母「C4 对称」**(它是 C1);说「销接受 C4 等价插入,抓握刚性携带」。

### PLAUSIBLE 且能翻转结论符号的(出任何 headline 数字前必须清掉)
- **动作共轭 bug**:RL 版把 rot6d/quat 共轭**符号错过两次**;diffusion 版只在**合成 numpy** 上测过,没接过真 EquiDiff。→ **阻断门:在真 square_d0 上做逐帧闭环动作一致性测试**(增强 obs+action,从增强后 next-state 反推动作,断言=共轭标签)。这是 go/no-go。
- **制造够不到的位姿**:ENGAGE 原来绕孔轴转**整段搬运**(180/270°→横扫出桌/够不到),位置型 `on_peg` 判据看不出来。→ **已修**(见 §3):keyed 每 episode 采一次、锚在**物体自身轴**(原地转)。仍需 IK/工作空间过滤兜底。
- **高 demo 数下精度反降**:方销是精度受限非覆盖受限;多余模式分走 diffusion 概率质量,≥200 demo 可能**降**成功率。→ 扫 demo 数验证。

### 2026-07-31 实测:C4 在运动学上做不满(per-k 生成成功率,mimicgen Square_D0,各 30 attempts)
| k | 朝向 | 生成成功率 |
|---|---|---|
| 0 | 0° | 60%(≈baseline,转 0°=原样) |
| 1 | 90° | **43%**(可行,略难) |
| 2 | 180° | **13%**(基本不可行) |
| 3 | 270° | **10%**(基本不可行) |
**结论**:抓着手柄把螺母翻 180/270° 插入 → 腕关节超限/够不到,大量失败。**"4-way C4" 在 Panda 上做不满,可达子集实质是 {0°,90°}**(≈+90° 增强,不是干净的子群——C4 的真子群只有 C1/{0,180},而 180 不可行)。不是 patch bug(k=0≈baseline)。→ **贡献从 "C4" 降级为 "可达子集 keyed 增强(+90°)"**,pitch 明显变弱;是否仍涨点交由 within-Square ablation 决定(**走 A:限定 k∈{0,1} 生成 → 跑 4 格 ablation**)。datagen hook = `mimicgen_c4_hook.py::PATCH`(已在真 repo 落地并跑通)。

### 2026-08-04 实测:EquiDiff host 上 keyed 不涨反微降 → headline 证伪
数据集 base/keyed 各 120 demo(k=0 vs k∈{0,1}),EquiDiff `train_equi_diffusion_unet_abs`,Square_D0 eval,n_demo=100,batch64,schedule=500 epoch(`50000/n_demo`),eval 每 10 epoch(满档 50 次),报告口径=**末 10 档 `test/mean_score` 平均**。

| EquiDiff host, 1 seed | last10 | peak | final |
|---|---|---|---|
| base(aug-OFF) | **0.866** | 0.96 | 0.86 |
| keyed-C4(+90°) | 0.844 | 0.90 | 0.80 |
| **Δ = keyed − base** | **−0.022** | −0.06 | −0.06 |

**三口径全是 base ≥ keyed**;逐档对照(rollout_every=10,两 run 同 epoch 可比)也是 base 全程 ≥ keyed。→ **keyed 注入的 90° 插入朝向被 C8 等变性本就覆盖 = 冗余增强,对已等变网络最坏轻微拖累(印证 T2/T3 + `2203.04439 §H.4`)。核心赌注「keyed 补架构表达不了的对称」证伪。** 单 seed/−2.2pt 在噪声内,但符号一致 → 加 seed 翻正概率极低。
- ⚠️ 踩坑:首个 base run 在 epoch 65/500(13%)被中断 → 假 delta 0.84 vs 0.34,是**训练时长差**不是效果;重跑满 500 才得上表。**任何 arm 必须跑满 `max_epoch=499`(50 档 eval)再比。**
- **仅剩的救故事路径 = DP 行**(非等变宿主):DP×keyed > DP×base → 「增强⟷等变架构替代」故事(弱,部分被 `2203.04439` scoop);DP×keyed ≈ DP×base → keyed 全局惰性,硬负结果。**下一步:跑 DP×{base,keyed}。**

### 2026-08-06 实测:完整 2×2 → keyed-C4 两个宿主都无效(硬负结果,augmentation 路线证伪)
DP host = `train_diffusion_unet`(非等变 `DiffusionUnetHybridImagePolicy`,`task: mimicgen_abs`,与 EquiDiff 同 obs/action/schedule/seed0)。四格同 n_demo=100 / batch64 / 满 500-epoch / 末10 规则:

| host | variant | last10 | peak | final |
|---|---|---|---|---|
| DP(非等变) | base | 0.736 | 0.82 | 0.74 |
| DP(非等变) | keyed | 0.748 | 0.82 | 0.78 |
| EquiDiff(C8) | base | **0.866** | 0.96 | 0.86 |
| EquiDiff(C8) | keyed | 0.844 | 0.90 | 0.80 |

**Δ(keyed−base):DP +0.012,EquiDiff −0.022 —— 两个都在噪声内(n=50,SE≈0.02–0.03)→ keyed-C4 对两个宿主都实质无效。**
- 健全性:equi base 0.866 vs DP base 0.736 = **+13pt**,复现文献 EquiDiff≫DP → setup 正确、null 为真非 bug。
- **两个独立死因**:(1) 等变网络免费拿旋转 → 旋转增强对它**按构造冗余**,杀 headline 且换任何 eval 都成立(只要宿主是 EquiDiff);(2) eval 单模态(D0 固定插入朝向)→ 多加的 90° 朝向是"eval 从不考的技能",连非等变 DP 也几乎不吃。符号方向(增强微助非等变/微损等变)与理论一致但幅度=噪声,不成 contribution。
- **结论:augmentation-beats-baseline 赌注在 Square 上证伪,死因(1)构造性、无法靠 eval/seed 翻盘。** 要"击败 baseline"须换机制——回 ICLR pivot 的 **context 进网络(B2)**(与旋转-冗余无关),或换一个"eval 真需多朝向 且 架构不自带该对称"的任务(但那就不能用 EquiDiff 当宿主)。**diffusion/keyed 这条线到此为止,作为 negative-analysis 材料保留。**

---

## 2. 血淋淋的结论

**按现状不可辩护,而且它设计的证伪测试是坏的。但没死——有且只有一个非冗余内核。**
- 3 臂里 2 个走了(global=冗余控制;approach=冗余且被作者自己的表示法碾压)——**都别护**。
- 「方赢圆平」headline 预测**确认不诊断**——**这条死了**。
- 扛住所有 CONFIRMED 攻击的**只有一件事**:**离散物体点群(C4)ENGAGE 增强**注入 out-of-support 的插入朝向,是 C8 架构(局部≠整场景)、diffusion 原生多模态(demo 单模态)、canonicalization(对 C4 物体 4 重歧义)、relative-frame 表示(单帧注入不了 solution-set 对称)**都注入不了**的——**这是真实、可辩护的小生态位**。
- 但这个内核仍是**未验证的经验赌注**,被两个能翻符号的隐患守着(动作共轭 bug、够不到位姿),出数字前必须清掉;并要正面处理 v1 自我 scoop。

---

## 3. 修订计划(pivot)

**一句话新贡献**:*一个分相位的 **离散物体点群(C4)ENGAGE 增强**,向全局-C8 等变 diffusion BC 策略注入 out-of-support 的 C4 等价插入朝向——架构、canonicalization、相对帧表示都表达不了的 solution-set 对称。* **丢掉 global(仅控制)与 approach(让给相对帧)。**

**修增强本身**:
1. keyed C4 仅施于**对齐插入帧**(或整段但锚物体自身轴——**已改成后者**);
2. **IK/工作空间过滤**每个增强 EEF 位姿,丢不可行的,报 %通过 IK;
3. **别信 `on_peg` 位置判据**兜底可行性。

**阻断正确性门(出任何结果前)**:真 square_d0 上的逐帧闭环动作一致性测试(非合成)。

**换实验设计**:
- **Headline = Square 内部 ablation**(matched budget + matched 增强集大小):`aug-OFF` vs `keyed-C4-ON` vs `global-rotation-only`。**(keyed − global) 这个差 = 论文。**
- **加非等变 DP 宿主臂**:若 keyed 帮到 DP 却拖累/持平 EquiDiff → destabilization(2303.13458)坐实、前提垮 → **这是最干净的证伪测试,早跑**。
- **扫 demo 数**(50/100/200):证明是低数据覆盖效应、≥200 不反转。
- **正面证据**:demo 插入 yaw 直方图(单峰)+ 未增强 EquiDiff 从不产生另 3 朝向。
- **圆控制(若留)**:改成预测并测 `square_gain > round_gain`,用**同一 pipeline** 生成真正 SO(2) 对称抓取(环形销+居中抓取无手柄)的 single-Round 集,并 pre-register。**别发「tie on round」。**

**定位/related work(先发制人堵 scoop)**:显著引 v1(2508.11204)+ 三个 v2-delta;引 Practical Guide(2505.13431)、Canonical Policy(2505.18474,对 C4 4 重歧义→正指向你的位)、Fourier Transporter(2401.12046,已架构式做 object×goal 点群但仅开环俯视)——claim「只在闭环接触 6-DoF diffusion BC + C4(canonicalization 可证歧义)这个 regime 里,用增强交付离散 bi-equivariant 收益」。

**最强版(stretch)**:把 C4-engage 增强跑在**相对帧/eye-in-hand EquiDiff** 之上(而非裸全局网),证明它找回一部分 bi-equivariant 模型的样本效率 → 把方法定位成**对架构互补**(唯一持久的位置)。

> ⚠️ 很新、引用前核实:2505.13431 · 2505.18474 · 2410.23179 · 2303.13458 · 2206.09450
