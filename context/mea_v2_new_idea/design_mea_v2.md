# mea_v2 设计:Context-Conditioned(条件等变)数据增强

- **日期**:2026-07-08
- **分支**:`meav2`
- **依据**:理论 [[new_idea]](`context/mea_v2_new_idea/new_idea.md`)+ 证明 `proof.tex`;负结果与根因见 [[mea_screening_results]]、[[mea_augmentation_issues]]。
- **目标**:把 new_idea 的「Conditionally Equivariant POMDP」落成一套**任务正确、物理合法、非冗余**的增强,替代 v1(`generate_sym3`)。**加法式、开关门控,不动现有 v1 实验管线。**

---

## 0. 先纠正一个关键错配(否则会做错)

new_idea 的样例是 **pick-and-place**(approach → grasp → place-to-**goal**),其 pull/place 阶段「只转 夹爪+物体 复合体、绕 goal 锚点」。**但当前任务 `block_pull` 不是 place-to-goal,而是关系型任务:**

- 成功判据(`close_loop_pomdp_block_pulling.py:47`):`objects[0].isTouching(objects[1])` —— 把 movable 块 0 拉到**与块 1 接触**。
- 奖励(`:50`):`-‖block0_xy − block1_xy‖ + 0.1` —— **只依赖两块相对位置**。

推论(决定整套设计):

| | 结论 |
|---|---|
| **唯一 reward-invariant 的对称** | 整场景**全局 SE(2)**(块0+块1+夹爪一起刚性旋转/平移);块间距不变 → 接触判据不变 |
| **为什么 v1 ≈ baseline** | 这个全局对称 C4 等变网络**已内建**(离散 90°+conv 平移)→ 全局对称增强对它冗余(根因 E),实测证实 |
| **new_idea 的 pull 复合体变换在这里非法** | 只转「夹爪+块0」而块 1 不动 → 改变块间距 → **破坏 reward/成功**,不能照搬 |

所以 mea_v2 在 block_pull 上的**真正价值点**,必须是全局对称(离散或连续)都**给不了**的东西 —— 见 §2。

---

## 1. 代码事实(已核实,增强要落在这些对象上)

- **动作** `a = [gripper∈[-1,1], dx, dy, dz, dyaw] ∈ [-1,1]^5`;物理尺度 `dpos=0.05, drot=π/8`。
- **`obs['pc']` 是按实体分开的点云 dict**:`{gripper, object1, object2, object3(, goal)}`,各 `(N,3)`(`pomdp/env.py:_obs_proc`)。→ **可对不同实体施加不同刚体变换**,这是条件增强的基础。
- **重渲染现成**:`dummy_env = Occup(DummyEnv)`。`DummyEnv.step(action)` 按实体变换点云,外层 `Occup` 直接产出 `occup_image`(`api.py:14-15`)。
- **网络输入 = 2×84×84**(`gym_regularizer.py`):`layer0`=occupancy,`layer1`=**常数标量平面**,当前取 `obs['image'][1,0,0]` = **gripper-state**(`getGripperImg` 广播)。→ **layer1 是 context 变量 c 的天然注入槽**(常数平面,C4 下不变 = 天然 gauge 通道)。
- **上下文信号**(每步 obs 都有,增强时可用):`gripper_pos[2]`(高度)、`z_distance`(夹爪→物体垂距)、`gripper_close`(0/1)。→ 足以切分阶段。
- **增强触发**:`Sym._on_end_gt_eps`,每条真实 episode 结束生成 `mea_expert_eps` 条,排队重放入 buffer(`env/wrapper/sym.py`)。
- **两块布局**:块0=blue=movable(target),块1=other;同 x、不同 y,随机 yaw(`reset`)。

---

## 2. mea_v2 的对称结构(按 context 条件化)

核心思想:**不同任务阶段,合法且非冗余的对称群不同**。对 block_pull:

### 阶段 c1 — APPROACH(抓取前,夹爪自由空间接近块0)
- **锚点**:块 0(target)质心,竖直轴。
- **群**:连续 `SO(2)`(θ₁∈[0,2π)),**只作用于 gripper 点云**;块 0、块 1 **不动**。
- **为什么合法**:夹爪在自由空间、两块静止;绕块0 锚点旋转夹爪 → 当夹爪到达块0(=锚点)时旋转对其自身为恒等 → **抓取位姿被保住**;远离块0 时接近角被旋转 → 得到「同一块布局、不同接近角」的样本。块不动 → reward 不变。
- **为什么非冗余(关键)**:全局对称(离散或连续)永远是**两块一起转**,网络**从未见过**「块布局固定、只有接近角变化」。这是 C4 网络拿不到的新信息。← **mea_v2 的主要增益来源**,也直接修复根因 B(v1 只照抄、几乎无新信息)。
- **context**:`c=0`(unloaded / 局部 gauge,anchor=block0)。
- 对应 new_idea 的 Phase 1「strictly transform relative pose between robot and target object, mask out irrelevant goal」——这里「mask out」= 块 1 不参与变换。

### 阶段 c2 — GRASP / CONTACT(抓取过渡帧)
- **群退化为恒等**(或极小 yaw);**禁 dz、禁 pitch/roll**。
- **为什么**:接触时刻任何 z/倾斜扰动都会产生穿桌/脱手的非法帧(new_idea 的 "physics violation")。窗口很短(1–2 帧,`gripper_close` 翻转处)。
- **context**:`c=0→1` 的边界。

### 阶段 c3 — PULL(抓取后,负载,把块0拉向块1)
- **锚点**:场景锚(块1 或 grid 中心)。
- **群**:连续 `SO(2)`(θ₃∈[0,2π)),作用于**整场景**(gripper+块0+块1 一起刚性旋转);夹爪对块0 的刚性抓握天然保持。
- **为什么合法**:整体刚性旋转 → 块间距、接触判据不变 → reward 严格不变。
- **为什么非冗余**:**连续**角填补 C4 只有的 4 个离散角之间的空隙(边际,弱于 c1)。
- **context**:`c=1`(loaded / 全局 gauge)。
- 注:这里**不能**像 new_idea 那样只转复合体不管块1 —— 见 §0。

> **条件等变的意义(对应 proof.tex 两条定理)**:每个 context 内部,增强都是该 context 合法群的变换 → Assumption 1/2(reward 不变、transition 等变)在**该 context 内**成立 → Theorem 1/2(Q/V 局部不变、策略条件等变)成立。**把 `c` 喂进网络**(§3)就是让网络知道当前处于哪个 gauge、避免把 c1 与 c3 两套不同变换语义混为一谈(new_idea §5 的「high variance / fail to converge」正是没有 c 时的后果)。

---

## 3. context 变量 c 如何进网络(可选、加法式)

- 现成槽位:`gym_regularizer.py` 的 **layer1 常数平面**(现放 gripper-state)。
- **v2-aug-only(先做)**:不改网络,c 只用于**生成**增强(决定各阶段用哪套变换)。layer1 保持 gripper-state 不变。→ 零网络改动,先验证增强本身有没有用。
- **v2-context(后做,对应 new_idea 的 MoEE / gauge estimator)**:把 c(如 `{0:unloaded, 1:loaded}` 或 phase one-hot)编码进 layer1(或新增第 3 通道,需改 `observation_space` 与 encoder 首层)。这是 new_idea 的完整版,属**下一步**,不在首版范围。

---

## 4. 算法(生成一条增强 episode)

输入:真实 episode 的 `obs[0..T]`、`actions[0..T-1]`、采样角 θ₁, θ₃。

```
1. 切阶段:
   grasp_step k = 第一个 gripper_close 翻转(open→closed)的帧 (fallback: 第一个 gripper_pos[2] < z_thres)
   c1 = [0, k)   ; c2 = {k}      ; c3 = [k+1, T]
2. 定锚:
   p0 = 块0 质心_xy @ obs[k]（抓取时target位置）; q = 块1 质心_xy 或 grid 中心
   grasp_obj = 离 gripper 质心最近的 object* @ obs[k]   # 认定 target=块0
3. 逐帧重渲染(用 dummy_env.set_current_points + apply_transform):
   for t in c1:  只把 gripper 点云绕 p0 旋 R_z(θ1);块不动
   for t in c2:  恒等(或极小 yaw)
   for t in c3:  把所有实体绕 q 旋 R_z(θ3)
   → occup_image（dummy_env 的 Occup 现成产出)
4. 变换动作(刚体旋转下 delta 的变换):
   c1: a'[1:3] = R(θ1)·a[1:3];  a'[3]=a[3];  a'[4]=a[4];  a'[0]=a[0]   # 平移delta旋转, dz/dyaw/gripper 不变
   c2: a' = a
   c3: a'[1:3] = R(θ3)·a[1:3];  其余同上
   （yaw 绝对朝向的 θ 偏移:approach 段夹爪朝向也应随 θ1 转;首版把该偏移并入起始帧,见代码 TODO,需在真机轨迹上校验）
5. reward/done 原样复制（各变换在其 context 内 reward-invariant）。
6. context 标签:c1→0, c3→1，写入 obs['image'][1]（v2-context 时启用）。
```

**复杂度**:c1/c3 每帧一次刚体变换 + 一次 occupancy 渲染 = **O(T)**,比 v1 的 O(T²) 反复前向重放**更省**(顺带缓解根因 F 的渲染开销)。

---

## 5. 落地(加法、门控,不碰 v1)

1. **新文件** `rgbd_sym/tool/sym_v2.py`:`generate_sym_v2(obs, actions, dummy_env, mode, params)`。
   - `mode='global'`:整轨全局连续 SE(2)(§0 里唯一严格 reward-invariant 的安全基线;非冗余仅靠连续角)。**先实现、必对**。
   - `mode='conditional'`:§2 的分阶段版(c1 局部 + c3 全局)——**研究主体**,标 experimental。
   - 变换原语:`se2_about(anchor_xy, theta)`、`transform_pc(pc, T)`、`transform_action_se2(a, theta)`、`segment_phases(obs)`。
2. **加法方法** `DummyEnv.apply_transform(transform_dict)`(`env/embodied/dummy/env.py`):按实体施加任意 4×4 刚体变换(现有 `step` 只支持 action 驱动的绕原点 z 转+平移,不够)。纯新增,不改 `step`。
3. **门控 hook**(`env/wrapper/sym.py`):`Sym.__init__` 加 `mea_version='v1'`(默认);`_on_end_gt_eps` 里 `if self._mea_version=='v2': generate_sym_v2(...) else: generate_sym3(...)`。**默认 v1,现有实验零变化。**
4. **config**:新增 `configs/block_pull/mea_v2-rnn-equi-all.yml`(复制 mea 版 + `mea_version: v2` + `mea_v2_mode`),`main.py` 透传 flag。
5. **api.py**:已 `**kwargs` 透传,无需改。

---

## 6. 验证实验(回答「mea_v2 到底有没有用」)

同 data-scarce 协议(demo=15、3 seed、equi 网络),四臂对照:

| 臂 | 说明 | 检验什么 |
|---|---|---|
| baseline(mea=0) | 参照 | — |
| v1(`generate_sym3`) | 旧增强 | 复现负结果 |
| **v2-global** | 整轨连续全局 SE(2) | 连续角对 C4 网络有无边际增益 |
| **v2-conditional** | 分阶段(c1 局部 + c3 全局) | **接近角局部增强**(非冗余主张)是否带来真实增益 |

- 判据用 **AUC / 达标步数**(抗 10-episode eval 噪声,见 [[mea_screening_results]]),不看抖动终点。
- 预期:若 conditional > global ≈ baseline,则「局部接近角增强是唯一非冗余增益」被证实,给论文一个干净的因果叙事;并可再上 **v2-context**(把 c 喂网络)看能否进一步放大。
- **对照实验**(呼应上一轮建议):把 v2 接到 **normal(非等变)网络**,若 global 那臂在 normal 上明显领先、equi 上不领先 → 冗余假设(根因 E)彻底坐实。

---

## 7. 风险 / 待校验

- **接近角局部变换的动力学一致性**:c1 只转 gripper、块不动,`(obs_t, a_t)→obs_{t+1}` 需在真机轨迹上核对(尤其 yaw 绝对朝向偏移、抓取衔接帧)。首版给 `mode='global'` 作为**必对基线**兜底。
- **出界**:全局/局部旋转可能把点云转出 occupancy 窗口(`pc_range=0.4`);需夹取角度或按场景质心定锚,渲染后做 in-bounds 断言。
- **grasp_obj 认定**:靠「离夹爪最近的 object」定 target=块0,随机 yaw 下需校验;可改用 env 已知的 `objects[0]`。
- **代码尚未在真机轨迹上跑过**(本环境无法起 pybullet rollout 验证)——§5 的实现按 scaffold 交付,需你在 env 里用一条真实 episode 做 `generate_sym_v2` 单元测试(存一条 rollout 的 `obs/actions`,离线跑增强,可视化 occupancy + 断言动作变换)。

---

## 8. 验证发现(2026-07-08,真机一条 rollout,`bash/test_sym_v2.py`)

抓了一条真实 `block_pull` expert episode(10 帧、成功拉动)离线跑 `generate_sym_v2`,三点关键发现,**修正了上面部分设计**:

1. ~~**动作约定其实很干净(已验证)**:`action[1]→世界x、action[2]→世界y` 是**恒等映射**(φ=I,det=+1,无轴交换/翻转),`dG≈action[1:3]`。→ **`mea_v2_action_sign = +1` 确认正确**;旋转/反射对动作 (a1,a2) 直接施加即可。~~
   **⚠️ 2026-07-17 作废:这条「验证」有漏洞,结论错误。** 它只验证了 action↔**世界**系(gripper_pos),但增强旋转的是 **pc/图像**系;实测(`bash/check_pc_action_frame.py`,静止物体在夹爪居中相机里的反向运动)pc 系 = 世界系 **x/y 对调**(M=[[0,1],[1,0]],det=−1,镜像副本;`DummyEnv.step` 的 `(−a2,−a1)` 早已编码此约定)。共轭后:pc 旋转 +θ ≡ 世界旋转 −θ → **`mea_v2_action_sign = −1` 才正确**;镜像应翻 **a[2]**(不是 a[1]);旋转锚点应为 **pc 原点(=夹爪=图心)**。旧代码让每条增强 demo 的动作标签错 2θ / 镜错轴 → **§8 之后所有 v2 实验臂作废**,详见 `mea_screening_results.md` §7.b。已修复并通过一致性断言。

2. **相机是 `camera_center_xyz`——夹爪恒在图像正中(gripper centroid≈(0,0))**。后果:
   - **`global` 模式渲染正确**(整场景绕图心刚性旋转,肉眼确认)。
   - **`conditional` 的「approach 绕 target 转夹爪」几乎是 no-op**:夹爪在中心,转它对占据图基本无改变;且被抓的 movable 块在抓取帧被夹爪**遮挡(0 点)**,`nearest_object_key` 会误选可见的另一块。→ **§2 的 c1 局部 gauge 在这个相机设定下失效**(它假设接近角写在图里,但夹爪居中相机把接近角吸收掉了)。
   - 更关键:reward-invariant 的对称(绕夹爪旋转)**≈ C4 网络内建等变** → 强冗余 → **这是负结果的机理**。

3. **网络是 C4(`flip_symmetry=false`),不是 D4 → 反射是网络结构上没有、而任务合法(关系型 reward 镜像不变)的对称** → **这才是真正非冗余的增强杠杆**(已实现,`mea_v2_reflect_prob`;渲染+动作变换均验证:镜像整场景 + `a[1]→-a[1], a[4]→-a[4]`)。

### 修正后的优先级
- **主推:`global` + `reflect`(连续旋转 + 50% 镜像)**。旋转填补 C4 的离散角空隙(弱),**反射补上 C4 缺的整个 O(2)\\SO(2) 陪集(强、非冗余)**。这是最可能把负结果翻正的一招,且严格 reward-invariant。
- `conditional`(approach 局部 gauge)在 `camera_center_xyz` 下**降级**;若要救,需换成**扰动两块的相对构型**(改变 target 相对 gripper 的角向位置而另一块不同步),但这与关系型 reward 耦合、valid 性更难保证,列为后续研究,不作首选。
- v2-context(c 进网络 / MoEE):视 `global+reflect` 结果再定。

### 实验建议(更新)
四臂改为:baseline / v1 / **v2 global+reflect** / v2 global(仅连续旋转,无反射)。第 3 vs 第 4 臂直接**隔离出反射的贡献**(核心假设);AUC 判定。

### 状态
- 设计 + 验证:**本文件**(§8 为真机验证结论)。
- `sym_v2.py`(rotation + **reflection**)+ `DummyEnv.apply_transform` + 门控 hook + `bash/test_sym_v2.py`:已实现。~~`action_sign=+1` 已定~~ → **2026-07-17 更正:`action_sign=−1`、mirror 翻 a[2]、anchor=origin**(见 §8 第 1 条的作废说明与 `check_pc_action_frame.py`)。
- 待你决定:先跑 **v2 global+reflect vs global** 对照(隔离反射贡献),还是先补 c1 的相对构型版本。
