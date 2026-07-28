# MEA v2 → ICLR:diffusion 增强路线计划

- **日期**:2026-07-28 · 分支 `meav2` · 决策:**MEA 从 model-free RL 增强 → 等变 diffusion 策略的增强**
- **主场**:MimicGen(robosuite)keyed 插入 · **要打败的 SOTA = Equivariant Diffusion Policy(EquiDiff, 2407.01812)**
- **依据**:[[sym_manip_tasks_baselines]](任务/baseline)、[[context_to_network_designs]] §1b、[[iclr_strategy]]
- **实现细节**:后台核实 workflow `wcthuiypi`(EquiDiff 代码 / MimicGen 数据 / 增强插入点)——回来补进 §5。

---

## 0. 先把贡献定位精确(否则打不过 EquiDiff)

EquiDiff 已经强加**全局 SO(2) 等变**(整场景一起转 → 等价)。所以 MEA 要涨,必须注入**全局等变拿不到**的对称/覆盖。**GIC 已证全局旋转穿接触不变**,所以不能靠"接触破缺"。真正非冗余的三块:

| MEA 注入的 | 为什么全局 SO(2) 等变拿不到 | 相位 |
|---|---|---|
| **物体点群自等价**(方销 C4:只转**销自身** 90° 而 hole/场景不动 → 同任务)| 全局等变只知"整场景一起转",**不知道"只转销 90°"也等价**(这是 object-local 自对称,不是 whole-scene 旋转)| 抓取 + 啮合 |
| **相对接近位姿多样性**(自由空间绕物体变接近角 α)| 全局等变只把见过的构型旋转,**给不出新的 gripper↔物体相对接近轨迹**(few-demo 下覆盖不足)| 自由接近 |
| (可选)**反射**(若 EquiDiff 是 SO(2) 非 O(2))| C-群缺反射陪集 | 全程 |

**相位索引(contextual)在这里的含义**:增强用的群**随相位变**——自由接近用**连续 SO(2) 接近角**,啮合用**离散 keyed 点群**(方=C4)。这是 SEIL(高斯、无 keyed、无相位离散群、非等变宿主)和 PE-SAC(空间门控、非相位、state-based)都没有的交集。

> **⚠️ 诚实风险(必须正视):我无法从第一性原理确信 MEA 一定打得过 EquiDiff**——全局等变已经很强,keyed 自等价的增益是**经验问题**。所以实验要设计成**即使 null 也有信息量**(见 §3 的方 vs 圆控制)。

---

## 1. 方法(MEA-v2-as-diffusion-aug)

- **宿主**:EquiDiff(等变 diffusion policy)。对照另设**非等变 Diffusion Policy(DP-C/DP-T)**。
- **增强**:对 MimicGen 的示教数据集做**离线**增强(diffusion 是 BC/去噪,增强作用在 demo 数据上,不需在线 rollout → **CPU 友好**):
  1. **相位切分**:每条 demo 切 自由接近 / 啮合(用夹爪↔物体距离、接触标志、或 robosuite 阶段);
  2. **自由接近段**:绕物体施**连续 SO(2)**(可 + 反射)于 gripper↔物体相对位姿(接近角 α,衰减收敛到抓取,见 [[mea-paper-and-winning-mechanism]] §9),动作按相对共轭;
  3. **啮合段**:施**物体点群**(方销 C4:把"销+夹爪复合体"绕孔轴转 0/90/180/270°,hole 固定;圆销:连续 SO(2));
  4. **重渲/变换**:MimicGen 是**状态+图像**;若能拿到 per-object 位姿则在**状态空间**变换物体+动作位姿最干净(避免重渲图像),否则需点云/图像重渲(见 §5 风险)。
- **(可选)网络侧**:轻量把相位/keyed 阶数喂进 EquiDiff 的条件(不是必须;首版走**纯数据增强**,把架构改动留作 ablation)。

---

## 2. 任务(全在 MimicGen / robosuite,CPU 可训)

- **Square**(NutAssemblySquare):啮合 = **连续→C4**,主力。
- **Threading**:≈ **C1** 近全极化,第二个。
- **Round**(NutAssemblyRound):啮合 = **SO(2) 不收窄** —— **控制组**(见 §3)。
- (便宜再加)Coffee 或 Three-Piece。**不镀金。**

## 3. 实验矩阵 + 决定性的受控隔离

**判据**:成功率 **vs 示教条数**的曲线(data-efficiency,MEA 的主场);报 AUC / 达标 demo 数,多 seed。

| 臂 | 目的 |
|---|---|
| DP-C / DP-T | 非等变地板 |
| DP + 常规增强(RAD/DrQ 式) | 挡"你就是做增强" |
| **EquiDiff** | **要打败的等变 SOTA(常量全局 SO(2))** |
| **EquiDiff + MEA-v2-aug**(本法) | 完整方法 |
| DP + MEA-v2-aug | 看增强在非等变宿主上是否也涨(隔离"增强 vs 等变") |

**★ 决定性控制:方(Square)vs 圆(Round)销**——几何/流程全同,只差截面:
- **圆销**:啮合 SO(2) 不收窄 → keyed 增强无新信息 → **MEA 应 ≈ EquiDiff 平齐**;
- **方销**:啮合收窄到 C4 → keyed 自等价是真新信息 → **MEA 应赢**。
- → **"方赢圆平"= thesis 干净证实,并直接回应 GIC**(是 keyed 点群 stabilizer,不是 contact);**"方也不赢"= 诚实负结果,省下继续投入**。这就是"即使 null 也有信息量"。

**归因 ablation**:关掉相位索引(全程同群)、只留接近角、只留 keyed —— 隔离每块贡献。

---

## 4. baseline / 定位(详见 [[sym_manip_tasks_baselines]] §2)

打败:EquiDiff;地板:DP-C/DP-T/ACT/BC-RNN;增强:RAD/DrQ/DP+Aug;对手论证:**PE-SAC(空间门控 ≠ 你的相位-keyed 群)**;rebuttal 锚点:**GIC(2308.14984)**(定位成 stabilizer reduction)、**EquiContact**(它重锚同一个群,你是群本身变)、**SEIL**(高斯 vs 结构化 keyed、BC vs diffusion)。

## 5. 实现路径(待 `wcthuiypi` 核实后补实锚点)

- EquiDiff 官方 repo → demo 数据加载处插入 dataset-transform(离线增强);
- MimicGen 预生成 demo(robomimic hdf5)下载;确认 Round vs Square 都有 demo(或 Round 需在 robosuite 自生成);
- 增强模态:优先**状态空间**(robosuite 暴露 per-object 位姿)变换物体+动作,规避图像重渲;
- **CPU-only 最大风险**:diffusion 训练本身的算力(即便 demo 预生成)——需评估单任务单 seed 的可行 wall-clock;若过重,先用最小 demo 预算 + 单任务(Square)跑通 pipeline 再扩。

## 6. 立刻可做的第一步(建议)

**Pipeline 跑通优先于铺满**:EquiDiff + MimicGen **Square**,先复现 EquiDiff 基线一条 seed(确认 CPU 可训、拿到成功率曲线),再接 MEA-v2-aug 一条臂。**通了再上 Round 控制 + Threading + 多 seed。** 这样最快撞到"CPU 到底训不训得动 diffusion"这个真风险。
