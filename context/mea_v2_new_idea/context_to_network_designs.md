# context → 网络:哪些 context 对应哪些(不同等变)子网络

- **日期**:2026-07-27  ·  分支 `meav2`  ·  承接 [[iclr_strategy]] 的 B2(网络侧)
- **产出**:workflow(3 angle:MoEE 群阶梯 / 部分等变门控 / gauge-frame)+ 几何分析(上一轮)
- **判据**:每个设计**是否 escape extrinsic-equivariance floor**(= 能否在接触相位真破缺对称 = 能否真涨过全局等变 baseline)。理论依据 Kaba & Ravanbakhsh(2312.09016,Prop 2.1:等变映射只能**增**对称 G_{φ(x)} ⊇ G_x)+ Wang/Walters/Yu(2303.04745:错配/extrinsic 等变有不可约误差下界 = orbit 内方差)。

---

## 0. 一个必须先钉死的事实:context 标量喂等变网 = Curie no-op

把 **C4-不变标量**(spare layer1 平面)喂进 **C4 等变网** → 整个映射**仍全局 C4 等变**(不变量是每个旋转的不动点)。→ **结构上破缺不了对称、涨不过 baseline。** 这个"最小 B2"**只配当反面 ablation**,用来展示 floor 的存在。**真正 escape floor 只有三条路**:(i) 降群(C4→C1/子群);(ii) 加无约束自由通路;(iii) 用等变**方向**(如接触法向)当规范帧,在该帧里放开。

---

## 1. 「不同 context → 不同网络」四种,按 escape-floor 分级

| 设计 | 每个 context 换的是 | escape floor? | 说明 |
|---|---|---|---|
| **① 群阶梯 C4→C2→C1** | **对称群本身**(接触相位退化到平凡/子群) | ✅ **真 escape** | C1 专家无权重共享 → 能发出旋转依赖输出,破缺接触几何的对称 |
| **② 等变+自由 双通路门控** | 等变 vs 无约束 pathway | ✅ 干净 escape | 自由通路 λ>0 处非等变 → 教科书式 escape |
| **③ 锚点切换 object→goal** | 参考系/锚点(同群) | ❌ **不 escape** | 等变式规范化**保持**全局等变,只换 bias/中心;与几何结论一致(锚点对全局旋转不可见) |
| **④ loaded/unloaded 动力学 head** | 价值/动力学头 | ❌ 不破缺 | 只是别让 critic 平均两套动力学(value-fitting 收益),非对称收益 |

**含义**:能带来"超过 baseline"头寸的只有 ① 和 ②;③④ 是配套的 bias/动力学改进,单独不产生 escape。**锚点切换(③)不 escape** 与上一轮几何结论一致——夹爪居中观测下锚点对全局旋转不可见,重锚只挪等变中心。

---

## 2. context → 子网络 的具体映射(推荐配置)

| context(相位) | 有效对称 | 子网络 / expert | escape? |
|---|---|---|---|
| **c1 approach**(自由/空载)| SE(2)/C4,锚 **object** | C4 等变 expert,object-gauge | 保持等变(对) |
| **c2 contact/grasp** | 旋转被接触几何破缺 | **C1(无约束)通路 / C4→C1 restriction**,gate 由 belief 相位打开 | ✅ **这里 escape floor** |
| **c3 manipulate**(负载/拉推放)| SE(2),锚 **goal** | C4 等变 expert,goal-gauge + **loaded head** | 重锚(bias)+ loaded 头(动力学) |

**路由**:phase 后验 p(c\|h_t) 由 RNN belief 上一个小 MLP 给出(相位本身旋转不变 → 用不变 belief 路由**合法**;escape 来自**被选中的子网群更小**,与路由是否不变无关)。硬 argmax(单 expert)或软 softmax。

---

## 3. escnn/e2cnn 实现要点(最小版 = ① 的 C4 + 接触 C1 通路)

- 主干:现有 `gspaces.rot2dOnR2(N=4)` steerable stack(= 现 Equi-RSAC)。
- 接触通路:一条并行 `trivialOnR2()` 普通 CNN;或对主干输出用 `escnn.nn.RestrictionModule` / `FieldType.restrict(...)` 从 C4-regular 降到 C1(4 个 trivial 通道)。
- 融合:在公共 field(C1)上做 `(1−λ)·C4分支_restrict + λ·C1分支`,λ = belief 相位 gate(单调 schedule:自由 λ=0、接触 λ→1);再进共享不变 Q-head / steerable 高斯 actor-head。
- (可选)③重锚:接触后把观测**以 goal 重居中/规范化**(相对帧);④ loaded head:c3 用单独价值头。
- 稳定性:对 λ 或 expert 用 load-balance/entropy 正则,防坍缩;free 通路只在接触少数帧开,自由/manipulate 保住 C4 的样本效率。

---

## 4. novelty 地图(诚实)

**已被占,不能当贡献(只能作对比 baseline)**:
- **② 等变 vs 自由 门控 = PE-SAC「Partially Equivariant RL」(arXiv:2512.00915, ICLR 2026)** ——按状态门控等变/无约束 critic+actor,已发表且就是 ICLR。**这是最大威胁**:"把 context 喂网络放松等变"的裸想法已是 prior work。
- 全局 equi+free = **Residual Pathway Priors**(2112.01388);全局 relaxed group conv = **Approximate Equivariance in RL**(2411.04225)。
- ③锚点/frame 切换有**并发风险**:Mixture-of-Frames(⚠️`2607.11884`,2026-07,**未核实**)+ bi-equivariant Transporter(2401.12046 / 2202.09400)。

**未被占、可辩护的交集(= 架构侧 novelty)= 下面的合取**:
- (a) **K>2 群阶梯 C4/C2/C1**(非 PE-SAC 的 G-vs-trivial 两专家);
- (b) 路由来自 **POMDP 的 RNN belief 推断的接触相位**(PE-SAC 全可观、dynamics-disagreement、无记忆);
- (c) 绑**物理接触相变**的单调 schedule(自由严格等变 → 接触退化 → 抓取后重锚+loaded);
- (d) **锚点重选 object→goal + loaded/unloaded 头**;
- (e) 与**数据侧增强(MEA)配对**——architecture 侧与 data 侧是同一 contextual-equivariance 的两面。

**一句话定位**:*"belief-相位路由的 群退化 + 重锚,在 POMDP 等变 RL 里,统一数据增强侧与网络侧"——即 PE-SAC 两专家全可观门控的 多群/belief/POMDP 推广。*

---

## 5. 证明它的 ablation(在 [[iclr_strategy]] 阶梯上加一臂)

| 臂 | 说明 | 预期 / 作用 |
|---|---|---|
| B0 baseline | 全局 C4 | 参照 |
| **B2a 不变标量喂 C4 网** | context 当不变通道 | **Curie no-op,应≈B0** → 反衬"标量没用" |
| **B2b 群退化(①)** | 接触相位 C4→C1 gate | **应 > B0**(escape floor)→ 证明群退化才是关键 |
| B2c +重锚+loaded(③④)| 加 c1↔c3 gauge / 动力学头 | bias/动力学增益 |
| B3 full | + 数据增强(MEA) | 完整方法 |

**因果叙事**:B2a≈B0 ≪ B2b → "把对称**条件化退化**(而非喂不变标量)才 escape floor";再叠 MEA 数据侧 → B3 最高。直接回击"就是换个增强 / 就是喂个 context"的质疑,并和 PE-SAC 用"群阶梯+belief+POMDP+数据侧"区分。

---

## 参考

**已核实**:PE-SAC/Partially-Equi-RL 2512.00915(ICLR 2026)· RPP 2112.01388(NeurIPS 2021)· Approx-Equi-RL 2411.04225 · Symmetry-Breaking-ENN 2312.09016 · Extrinsic-equi 2303.04745 · Canonicalization 2211.06489(ICML 2023)· Frame Averaging 2110.03336 · e2cnn 1911.08251 · Equivariant Transporter 2202.09400 · Fourier Transporter 2401.12046 · TAX-Pose 2211.09325 · Equi-RL under POMDP 2408.14336
**⚠️ 引用前核实(很新/并发)**:Mixture-of-Frames 2607.11884 · STAR 2510.27222 · G-biases RREConv 2408.12454 · 各 2026 subgroup-breaking 条目
