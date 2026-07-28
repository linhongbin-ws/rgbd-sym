# MEA v2 → ICLR:diffusion 增强路线计划

- **日期**:2026-07-28 · 分支 `meav2` · 决策:**MEA 从 model-free RL 增强 → 等变 diffusion 策略的增强**
- **主场**:MimicGen(robosuite)keyed 插入 · **要打败的 SOTA = Equivariant Diffusion Policy(EquiDiff, 2407.01812)**
- **依据**:[[sym_manip_tasks_baselines]](任务/baseline)、[[context_to_network_designs]] §1b、[[iclr_strategy]]
- **实现细节**:后台核实 workflow `wcthuiypi`(EquiDiff 代码 / MimicGen 数据 / 增强插入点)——回来补进 §5。

---

> **⚠️ 2026-07-28 对抗证伪后大改,以 [[falsification_and_pivot]] 为准。** 本文 §0/§1/§3 的"三臂(global/approach/engage)+ 方赢圆平"框架**已作废**:global 冗余(仅控制)、approach 冗余且被 EquiDiff 作者自己的 eye-in-hand+相对动作配方(2505.13431)碾压、「方赢圆平」控制**不诊断**。**唯一贡献 = 离散物体点群(C4)ENGAGE 增强。** 修订后的方法/实验矩阵见下方 §7bis + [[falsification_and_pivot]] §3。

## 7bis. 修订实验矩阵(2026-07-28 pivot,取代 §3)

- **贡献(唯一)**:向全局-C8 diffusion BC(EquiDiff)注入 out-of-support 的 **C4 等价插入朝向**——架构/canonicalization/相对帧都表达不了的 solution-set 对称。
- **阻断门(出任何数字前)**:`mea_diff/test_action_consistency.py --hdf5 square_d0_abs.hdf5`——真数据上验动作共轭(合成版已过)。**FAIL 就别训。**
- **Headline = Square 内部 ablation**(matched demo 预算 + matched 增强集大小):`aug-OFF` / `keyed-C4-ON` / `global-only`。**报 (keyed − global) 差 = 论文。**
- **最干净的证伪臂**:同样三条跑在**非等变 DP** 宿主上。若 keyed 帮 DP 却拖累/持平 EquiDiff → destabilization 坐实、前提垮。**早跑。**
- **扫 demo 数** 50/100/200(防高数据精度反降);**正面证据**:demo 插入 yaw 直方图(单峰)+ 未增强 EquiDiff 从不产生另 3 朝向。
- **圆控制(可选、别当 headline)**:预测并测 `square_gain > round_gain`,自生成真正 SO(2) 抓取(环形销无手柄)的 Round 集,pre-register。
- **stretch**:C4-engage 跑在 relative-frame/eye-in-hand EquiDiff 之上 → 定位成对架构互补。
- **代码状态**:`mea_diff/phase_aug.py` 默认 = C4-engage-only(approach 降级为 `approach_aug=True` opt-in;engage 锚物体自身轴 + 可选 `workspace_radius` 过滤);`test_phase_aug.py` + `test_action_consistency.py` 离线全过。

---

## 0. ~~先把贡献定位精确~~(见上方 pivot;本节 approach/相对接近角部分已降级)

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

## 5. 实现路径(已核实 `wcthuiypi`)

**好消息:集成点很干净。**
- **Host**:`github.com/pointW/equidiff`(包名 `equi_diffpo`),基于 Diffusion Policy + robomimic 格式;等变是**架构式**(escnn steerable,SO(2)/离散 Cn 如 C8),**pipeline 里没有任何数据增强**("sym" dataset 只是对称**归一化**不是几何增强)→ MEA 有干净的落点。
- **增强钩子**:`equi_diffpo/dataset/robomimic_replay_image_dataset.py:197` 的 `__getitem__`(on-the-fly per-sample),或离线 `_convert_robomimic_to_replay`。
- **最省的做法(CPU 友好、首选)**:**不重渲**——在 dataloader 里读 low-dim 物体位姿 + point-cloud/voxel 点,施相位对应的 SE(2)/SO(2)/C4(绕物体锚),同步变换**物体位姿 + 点 + 动作**;`_abs` 配置转绝对位姿动作、`_rel` 转 delta。纯解析、无 MuJoCo。
- **相位索引 = 白送的**:**MimicGen 原生就把每条 demo 切成 object-centric subtask,每段带物体锚帧 + 相对 EE↔物体 SE(3)**。直接复用:pre-grasp subtask = 自由接近段(增强接近位姿),insertion subtask = 啮合段(增强 keyed 点群)。**不用自己切相位。**
- **动作/模态细节**:MimicGen 动作 = 7-D OSC delta(dx,dy,dz + axis-angle + gripper);EquiDiff 转 10-D abs(pos3 + rot6d + gripper)。图像是**预渲染 RGB**不能像素重渲;几何增强要走 **voxel/point-cloud**(EquiDiff 支持)或 sim 重渲。
- **命令**:下载 `square_d0`(1.62GB)→ `dataset_states_to_obs.py` + `robomimic_dataset_conversion.py` → `python train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d0 n_demo=100`;基线宿主 DP 用 `--config-name=train_diffusion_unet`。把 MEA-aug 做成 config 开关,keyed-vs-unkeyed / MEA-vs-EquiDiff 都是一行 ablation。

## 6. ⚠️ 两个硬障碍(核实后浮现,必须正视)

**障碍 A —— 算力:✅ 已解决(2026-07-28)。** 用户有 **RTX 3090(24GB,训练)+ RTX 3070(8GB,dev/验证)**(之前记忆里"CPU-only"指的是 Claude 的命令沙箱,不是用户硬件)。EquiDiff ~22GB@batch128 **能装进 3090**(24GB,必要时略降 batch);**3070 做缩规模 dev/单 seed 冒烟**(小 `policy.enc_n_hidden`、`dataloader.batch_size=32/64`、`n_demo=100`)。demo 静态可复用,只有训练那步用 GPU,用户在自己终端跑(同 pip 的模式)。→ **blocker 解除,可推进。**

**障碍 B —— Round 控制组不存在现成数据。** MimicGen 只发 `square`(单方销,C4)和 `nut_assembly_d0`(双销,非受控)。**没有 round-only 预生成数据集**。robosuite 有 `NutAssemblyRound` env,但要**自己写 MimicGen datagen wrapper(镜像 Square 的 subtask 定义)+ ~10 条源 demo + 跑生成**(用 MuJoCo,渲染耗时)。→ 干净的方 vs 圆控制**需额外工程**;退路 `nut_assembly_d0`(双销)是不受控的弱代理。

## 7. 立刻的第一步(算力已解决 → 可执行)

**Pipeline 跑通优先于铺满。里程碑:**
1. **[3090/3070] 复现 EquiDiff 基线**:clone `pointW/equidiff` → 下 `square_d0` → `dataset_states_to_obs.py` + `robomimic_dataset_conversion.py` → `python train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d0 n_demo=100`。先在 3070 用缩规模冒烟跑通,再上 3090 正式。拿到 EquiDiff 与宿主 DP(`train_diffusion_unet`)的成功率曲线。
2. **接 MEA-v2-aug 一条臂**:在 dataset `__getitem__` 里加相位索引增强(low-dim/point-cloud 解析变换,复用 MimicGen subtask 相位;动作按 `_abs`/`_rel` 共轭),做成 config 开关。
3. **上受控实验**:方(现成)vs 圆(需自生成 datagen,障碍 B)+ Threading + 多 seed + 归因 ablation。

**分工**:训练/渲染在用户的 3090/3070(自己终端);Claude 沙箱只能写代码 + 解析增强的离线单元测试(无 GPU)。
