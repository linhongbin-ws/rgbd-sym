# B2(context 进网络)落地计划 —— 先证 floor 存在,再建架构

- **日期**:2026-08-07 · 分支 `meav2` · 承接 [[iclr_strategy]] B2、[[context_to_network_designs]]、[[falsification_and_pivot]](2×2 证伪)
- **一句话**:augmentation 侧已在 Square 上证伪(2×2);方向转 B2(网络侧)。但 **B2 只在"extrinsic-equivariance floor 真实存在"的任务上才可能涨**,而我们刚测的 2×2 恰恰说明 **Square 没有 floor**(C8≫C1,加等变单调变好)→ **B2 在 Square 上大概率同样净损**。所以 B2 的第一步不是建网络,是**用便宜实验证明 floor 在哪个任务上真存在**;没有 floor,任何 B2 架构都无效。

---

## 0. 两条硬约束(来自我们自己的文档,违反就白做)

**约束①(Curie no-op,`context_to_network_designs.md §0`)**:C4-不变标量喂 C4 等变网 → 映射仍全局 C4 等变 → **结构上破不了对称、涨不过 baseline**。
→ **`iclr_strategy §5.1` 的"最小 B2 = 相位信号写 layer1 常数平面、网络不变"是被证伪的死路,只配当反面对照(B2a)。** 真 escape floor 只有三条:(i) **群退化**(C4→C1/子群,在被约束相位);(ii) **加无约束自由通路**(λ 门控);(iii) 用等变**方向**(接触法向)当规范帧、在该帧放开。

**约束②(floor 必须先存在)**:B2 靠"放松等变逃 floor"取胜。**没有 floor,放松只丢样本效率 → 净损。** extrinsic floor(2303.04745)= 强加错对称把等变模型渐近上限压到无约束模型之下。GIC(2308.14984):全局 z 旋转在桌面/插孔**全程合法**(body/相对帧不变)→ 在全观测 sim 里全局等变**几乎总是 intrinsic** → floor 极难制造。这正是历轮 null 的根因,也是必须先解决的前提。

---

## 1. 关键判断:2×2 已是 floor 探测器的初步读数 → Square 无 floor

| host | last10 | 群阶 |
|---|---|---|
| EquiDiff | **0.866** | C8(大)|
| plain DP | 0.736 | C1(无)|

**加等变单调变好(+13pt)→ 强加对称没有把上限压低 → Square 上等变是净有益、无 floor。** 推论:B2(放松/退化群)在 Square 上没有可逃的 floor,预期与 keyed-aug 一样净损(实测 −2.2pt)。**Square 不是 B2 的战场。**

---

## 2. 便宜且决定性的第一步 —— group-size ladder floor 探测器

把上面的两点读数补全成一条**方法论工件**(论文可直接用的"floor detector"):在候选任务上跑 **C8 / C4 / C2 / C1** 四档等变(EquiDiff 的 `rot2dOnR2(N)`,理想是 1 行 config 改 N;plain DP = C1 已有)。

- **成功率随群阶单调 ↑(C8≥C4≥C2≥C1)= 无 floor**(该任务对 B2 无望,如预期的 Square)。
- **在某中间/更小群处出现峰值(如 C4>C8,或 C1>C8)= floor 存在** → **B2 在这里有靶子**,且峰值群阶直接告诉你该退化到哪。

代价:每档一次 500-epoch 训练(和已跑的一样)。Square 上先补 C4、C2 两档(C8/C1 已有)→ **把"Square 无 floor"从推测坐实为测量**,同时验证探测器方法;然后把同一探测器搬到真正的候选 floor 任务(§3)。
**待确认**:EquiDiff 配置里群阶 N 是否是可改参数(`grep -rn "rot2dOnR2\|N=\|group.*order\|policy.N" equi_diffpo/config equi_diffpo/policy`)。若硬编码则是小改代码。

---

## 3. floor 从哪来 —— 任务抉择(pivotal,决定下游一切)

全观测 z-旋转 sim 里 floor 的**唯一真实来源**是"全局旋转把合法态映射到非法/分布外态"。三条候选:

| 选项 | floor 来源 | 便宜? | 强度 | 风险 |
|---|---|---|---|---|
| **A. 宽朝向 Square_D2 eval** | **臂运动学**:定基座臂的可达性随朝向剧变(per-k 实测 0°:60%→270°:10%)→ 宽朝向下全局旋转 extrinsic | ✅ 全复用现有 infra,只换 D0→D2 数据/eval | 中(D0 上被样本效率盖过;D2 放大)| 低 |
| **B. place-to-固定世界目标**(new_idea Phase-3)| **未随场景旋转的世界锚**:goal 固定在世界系 → 旋转场景移动 goal 但真 goal 不动 → extrinsic | ❌ 需在 robosuite/MimicGen 造/改任务 | 强(教科书式 floor)| 中(工程 + 若 goal 被观测则又变 intrinsic,须让 goal 部分不可观测)|
| **C. 方 vs 圆插入受控对**(sym_manip §0)| stabilizer 收窄:方→C4、圆→SO(2) | ❌ 需造 round 数据集 | 最干净的 thesis 隔离 | 中(GIC 下全局等变对两者都 intrinsic,floor 可能仍不现)|

**推荐:先 A(最便宜、全复用),把它当 floor 探测器的第一个真实靶子。** A 若现出 floor(D2 上 EquiDiff 对 DP 的优势收窄/反转)→ 直接进 B2 架构。A 若也无 floor → 说明全观测 sim 根本压不出 floor(GIC 主导)→ **paper 的证据重心必须移到 partial-obs / 真机**(你文档 `iclr_strategy §4.4` 已预判 context 的价值在真机最大),或走 B 构造未观测锚。

---

## 4. B2 架构(floor 任务确定后才建)—— 群退化,不是喂标量

最小可 escape 版(对应 `context_to_network_designs §3` 的 ①):
- 主干:现有等变 stack(EquiDiff 的 C8 / Equi-RSAC 的 C4)。
- **相位门控的群退化**:在被约束相位(啮合/插入),用 `escnn.RestrictionModule` 把 C_N-regular 降到 C1(或峰值群阶),或并联一条 `trivialOnR2()` 自由通路,融合 `(1−λ)·等变_restrict + λ·自由`,λ = 相位后验的单调 schedule(自由 λ=0 → 啮合 λ→1)。
- 相位后验:可观测相位信号(接触/gripper_close/z 或 belief-MLP)→ 相位本身旋转不变,路由合法;escape 来自**被选中子网群更小**,与路由是否不变无关。
- 稳定性:λ/expert 上 load-balance/entropy 正则防坍缩;自由通路只在啮合少数帧开,保住自由相位的样本效率。

---

## 5. Ablation ladder(一次证明 novelty=涨点)

| 臂 | 说明 | 预期 |
|---|---|---|
| B0 | 全局等变 baseline(EquiDiff / Equi-RSAC)| 参照 |
| **B2a** | 不变标量喂等变网(Curie no-op)| **≈B0** → 反衬"喂标量没用"(约束①的实证)|
| **B2b** | 相位门控群退化(§4)| **>B0**(escape floor)→ 证明群退化才是关键 |
| B3 | B2b + 匹配数据增强 | 完整方法(增强侧已知在 Square 惰性,须在 floor 任务上重估)|

因果叙事:**B2a≈B0 ≪ B2b** → "把对称**条件化退化**(而非喂不变标量)才 escape floor"。直接回击"就是喂个 context"的质疑,并用"群阶梯+相位门控"与 PE-SAC(2512.00915,空间门控、全观测)区分。

---

## 6. 宿主抉择:diffusion(EquiDiff)vs RL(Equi-RSAC)

- **diffusion(EquiDiff/MimicGen)**:infra 已就绪、有 GPU、Square 是 keyed;但"context 进等变 diffusion 网"改动大,且 Square 无 floor(须先解决 §3 任务)。
- **RL(equi-rl-for-pomdps/BulletArm/Equi-RSAC)**:new_idea/design_mea_v2 的原生栈(layer1 gauge 通道、MoEE、belief 相位),CPU 可负担;但 BulletArm 无原生插入,round-vs-square 要自建(工程量)。
- **倾向**:floor 探测器(§2)先在 diffusion 栈跑(复用 infra、快出结论);B2 架构在哪建取决于 §3 选定的任务落在哪个栈。

---

## 参考(引用前核实很新的号)
Curie/对称破缺 2312.09016 · extrinsic-equi 下界 2303.04745 · GIC 2308.14984 · PE-SAC 2512.00915 · Equi-POMDP 2408.14336 · EquiDiff 2407.01812 · SO(2)-Equi-RL 冗余 2203.04439
