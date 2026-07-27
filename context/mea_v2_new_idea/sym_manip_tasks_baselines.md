# 对称机器人操作:任务菜单 + baseline 对比阵容(MEA v2 选型)

- **日期**:2026-07-27 · 分支 `meav2` · 承接 [[context_to_network_designs]] §1b、[[iclr_strategy]]
- **产出**:5-agent workflow 调研(EquiRL / EquiIL / 接触装配 / benchmark suites + 综合),全部 WebFetch 原文
- **一句话**:contextual-equivariance 只在**合法群随相位真变**的任务上有立足点;那不是"桌面 grasp",而是 **keyed 插入/装配**(方销/齿轮/极化)。给定你 **CPU-only**,首选把 MEA 改成 **MimicGen Square+Threading 上的 diffusion 增强 vs EquiDiff**;或退而在自有 BulletArm/equi-RSAC 栈里造一个 **方销 vs 圆销**插入任务。

---

## 0. 框架纠正(重要,呼应前几轮):不是"接触破缺",是"keyed 点群 stabilizer reduction"

**GIC(arXiv:2308.14984)证明 SE(3) 等变在 body/相对帧里穿过接触保持不变** → "contact breaks symmetry" 是错的(桌面任务如此,插孔也如此)。真正使全局旋转对称收窄的是**被啮合几何的点群**:

| 几何 | 啮合时的有效轴向旋转对称 | 对 thesis 的价值 |
|---|---|---|
| 圆销 / 轴对称 | **SO(2)(不收窄)** | ❌ decoy——MEA v2 应≈平齐 baseline |
| **方/矩形销** | **C4 / C2** | ✅ 连续→离散,真收窄 |
| 齿轮 | C_n(齿数) | ✅ |
| 极化连接器(NEMA)| **C1** | ✅ 最强(几乎全破) |

→ **thesis 必须写成"全局/物体点群 stabilizer 的相位收窄(自由接近的连续朝向 → 啮合时的离散 keyed 子群)",绝不写成"接触破缺对称"**(否则被 GIC 一句话反驳)。**方销 vs 圆销(几何全同、只差截面)= 最干净的受控实验**:圆销无收窄(≈baseline)、方销收窄到 C4(MEA v2 该赢)。

---

## 1. 任务菜单

### [A] SE(2)-throughout(**避免**当核心 testbed)
物体全程在桌;yaw+平移处处合法、pitch/roll 处处破;**合法群每相位都是 SE(2),无相位变化**——contextual 无立足点(本会话 null 的根因)。
- **BulletArm**(你的血脉/在仓):Block Pick/Pull/Push、Drawer、Stacking、House Building、Bin Packing、Palletizing、Block-in-Bowl。(ramp/bump 6-DoF 变体只是斜面,仍 SE(2)。)
- robosuite:Lift/Can/PickPlace/Stack/Door/Wipe;MimicGen:Stack/PickPlace/Kitchen;Ravens:hanoi/stacking/sweeping;Meta-World/CALVIN/Kitchen:reach/push + 固定轴 articulated。
- **直接前作 Equi-POMDP(2408.14336)就是在 Block-Pick/Pull/Push+Drawer 上跑 C4/SE(2),物体从不离桌——这是审稿人已接受的前提。**

### [B] 相位收窄(**用这些**)
- **MimicGen(robosuite)** ← 社区对称-IL 基准、EquiDiff 的同一套:**Square**(连续→C4)、**Threading / Coffee**(→~C1)、Three-Piece-Assembly、Nut-Assembly。
- **robosuite core**:NutAssembly(square)、**ToolHang**(两段插入)、TwoArmPegInHole。
- **ManiSkill2(SAPIEN,GPU)**:PegInsertionSide(3mm)、**PlugCharger**(0.5mm 双销→C1)、AssemblingKits。
- **RLBench**:insert_onto_square_peg、stack_wine(keyframe/IL 风)。
- **Factory/IndustReal(Isaac Gym,GPU)**:圆 vs **矩形**销、齿轮、NEMA——**最干净的受控 SO(2)→C2→C_n→C1 隔离**(但 GPU)。
- **FMB(真机)**:grasp→reorient→insert——只作**动机引用**(真机-only、IL、无对称 baseline)。

**两个坑**:
- **Meta-World 的 peg-insert-side / assembly 是 decoy**:4-DoF、夹爪朝向固定朝下,朝向既不控制也不观测 → 展不出"旋转群变化"。**别用。**
- **圆销-only 是 decoy**:轴对称保 SO(2) → 无收窄 → 无 thesis。**必须用 keyed(方)几何,或圆 vs 方配对。**

---

## 2. baseline 对比阵容(审稿人会要的四层 + rebuttal 锚点)

- **非等变地板**:CNN-SAC/CNN-DQN(RL);或 IL/diffusion 路线的 DP-C/DP-T、DP3/3D-DP、ACT、BC-RNN。
- **增强 baseline(挡"你就是做增强"这刀)**:RAD(crop/rotation)、DrQ/DrQ-Shift、FERM、CURL;IL 侧 DP+Aug/DP3+Aug。
- **等变 SOTA(要打败的)**:
  - RL/POMDP 血脉 → **Equi-RSAC** + **Equivariant RL under Partial Observability(2408.14336)**——**你最近的前作、首要要打败的对象**(正因为它假设**常量群**)。
  - IL/diffusion 插入套件 → **Equivariant Diffusion Policy(EquiDiff, 2407.01812,CoRL'24)**——在 Square/Threading 上仍强加**常量全局 SO(2)**;可选 **Spherical Diffusion Policy(SE(3))**、EquiBot/EquivAct(SIM(3))、ET-SEED、Fourier Transporter。
- **部分/近似等变(你真正的 ICLR-2026 对手,别漏)**:**PE-SAC(2512.00915)**——按 (s,a) **空间**门控等变 vs 无约束;RPP(Finzi 2021)、Approx-Equi-RL(Park 2024)、EMLP。**必须论证:PE-SAC 的局部/空间门控 ≠ 你的相位索引群变化**;且 PE-SAC 是 state-based、sim-only,你是图像 POMDP + 真机。
- **POMDP-recurrent 对照(你血脉里的)**:RSAC、RAD-Crop-RSAC、DrQ-Shift-RSAC、SLAC、DreamerV2/V3、RA2C、DPFRL。
- **必须处理的 rebuttal 锚点(不打败、但要正面回应)**:
  - **GIC(2308.14984)**:SE(3) 等变穿接触不变 → 把 thesis 定位成 **stabilizer reduction**,不是 contact。
  - **EquiContact(≈2507.x,⚠️核实号)**:分了 free-space(全局 SE(3))/contact(局部帧),但**重锚同一个群**;MEA v2 的区别是**群本身**随相位变,不只是换帧。

---

## 3. 决定性推荐(考虑到 CPU-only 这个硬约束)

**硬约束(来自 env 记忆):你是 CPU-only。** → MuJoCo/robosuite 在线 RL 慢;ManiSkill2、IndustReal/Factory 假定 **GPU**(并行 sim/Isaac Gym)→ **基本出局**(尽管对称故事最干净)。这主导选择。

### 首选:**MimicGen Square + Threading**,用 **IL/diffusion 增强**框架,对打 **EquiDiff**
- **为什么这两个**:Square = 干净的**连续→C4**物体 stabilizer 收窄;Threading ≈ **C1** 近全破——覆盖"keyed 离散"与"完全极化"两个诚实实例。二者是 EquiDiff 的同一套 → 直接**和不建模相位变化的等变 SOTA 正面对撞**。
- **baseline**:EquiDiff(常量全局 SO(2))= 要打败的 SOTA;DP-C/DP-T/DP+Aug = 地板/增强;SDP 给 SE(3) 对照;ACT/BC-RNN 做 IL sanity;加 PE-SAC 讨论(+ 自模型的部分等变 ablation)以区分相位索引 vs 空间门控。
- **代价(诚实)**:MimicGen demo 预生成(**无在线 RL 采样成本 = CPU 大利好**),DP 训练 CPU/中端 GPU 可行。**真正代价 = MEA v2 从"model-free RL 增强"变成"diffusion/IL 策略的增强"——宿主算法换了,是真重写**,但这是唯一同时 (a) 合你算力、(b) 落在审稿人期待的 EquiDiff 对比上的路。

### 退路(若坚持在现有 model-free RL / BulletArm / equi-POMDP 栈)
- **在 BulletArm 自造一个 close-loop 插入任务:方(keyed)销 + 圆销对照**。你已拥有 `ext/equi-rl-for-pomdps` 的 BulletArm+equi-RSAC+equi-POMDP 代码,基建迁移小、CPU 上 RL 可负担。
- **圆 vs 方配对本身就是实验**:几何全同、只差截面 → 圆保 SO(2)(应≈常量群 baseline 平齐)、方收窄到 C4(MEA v2 应赢)。**这个受控隔离是 thesis 最有力的证明,也直接回应 GIC。**
- **baseline**:Equi-POMDP(2408.14336)+ Equi-RSAC = 常量群 SOTA;RSAC/RAD-Crop-RSAC/DrQ-Shift-RSAC/SLAC/non-Equi-RSAC = recurrent/aug 地板;PE-SAC = 部分等变对照。
- **代价(诚实)**:BulletArm 无原生插入,要**自建接触动力学并校验** —— 工程量不小,且结果落在自造 env(对 ICLR **重投**而言,弱于"在 MimicGen 上打败 EquiDiff")。

### 不要选
Meta-World 插入(4-DoF、无朝向控制)· ManiSkill2/IndustReal/Factory 当主场(GPU)· 圆销-only · FMB 当主场(真机-only/IL/无对称 baseline)。

**底线**:thesis 只在 **keyed 插入**(方/矩/齿轮/极化)上有立足点,定位为**全局/点群 stabilizer 收窄**(非"接触")。CPU-only 下:能接受换宿主算法 → **MimicGen Square+Threading + diffusion 增强 vs EquiDiff**;否则 → 自造 **方 vs 圆 BulletArm 插入 vs Equi-POMDP/Equi-RSAC**。第三个任务只在便宜时加(MimicGen Coffee 或圆销对照),别镀金。

---

## 参考(agent 已读;很新的号引用前核实)
- Equi-POMDP 2408.14336 · SO(2)-Equi-RL 2203.04439 · Equi-Q spatial 2110.15443 · On-Robot Equi 2203.04923 · PE-SAC 2512.00915(ICLR'26)
- Equivariant Transporter 2202.09400 / 2308.07948 · Fourier Transporter 2401.12046 · SEIL 2211.00194 · TAX-Pose 2211.09325 · Diffusion-EDF 2309.02685 · **EquiDiff 2407.01812** · ET-SEED 2411.03990
- IndustReal(RSS'23)· ManiSkill2(ICLR'23, Gu et al.)· RLBench(RA-L'20)· Meta-World · MimicGen 2310.17596 · FMB(IJRR'25)
- ⚠️ 核实号:**GIC 2308.14984** · **EquiContact ~2507.x** · Spherical Diffusion Policy
