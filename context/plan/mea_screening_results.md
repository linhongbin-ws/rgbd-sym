# MEA vs baseline 筛选实验结果(数据稀缺 regime)

- **日期**:2026-07-08(6 个 run 于 07-05 ~ 07-07 跑完)
- **分支**:`meav2`
- **目的**:代码 bug(A 动作回填 / F delta_rot)修复**之后**,在专为增强设计的「数据稀缺」regime 里,检验 MEA 对称增强到底有没有收益。设计与背景见 [[mea_augmentation_issues]]、脚本见 `bash/bench_train.sh`。
- **一句话结论**:**负结果**——即便在 MEA 本该最占便宜的稀缺 regime,`mea=12` 也**打不过 baseline**;两条曲线完全重叠,中段 baseline 反而略高。原因指向根因 B/E(与等变网络的**自我冗余**),属方法 limitation。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 任务 | `block_pull`,C4 等变 actor/critic(`equi/equi`,r4) |
| 对照 | MEA(`--mea_expert 12 --mea_normal 0`) vs baseline(`--mea_expert 0`) |
| 数据稀缺 | `num_expert_episodes=15`(完整 run 用 80) |
| 缩短 | `num_iters=500`(完整 run 用 800),各 ~26k env steps |
| 配对 | 3 个相同 seed(0/1/2)分别跑两臂,降方差 |
| eval | `num_eval_tasks=10`(每次 eval 只 10 个 episode) |
| 规模 | 3 seed × 2 臂 = 6 run,串行(每个 buffer ~10GB,不能并行) |

wandb 项目 `linhongbin/Symmetry_block_pull_e15`,run 前缀 `scr_mea_d15_s*` / `scr_base_d15_s*`。分析脚本 `bash/analyze_screening.py`(用 `wandb.Api` 拉 `metrics/success_rate_eval` vs `env_steps`,按 seed 平均,出里程碑表 + `screening_mea_vs_base.png`)。

## 2. 最终 eval 成功率(每 seed,单点,10 episode)

| seed | MEA (mea=12) | BASE (mea=0) |
|---|---|---|
| 0 | **1.00** | 0.40 |
| 1 | 0.40 | 0.50 |
| 2 | 0.40 | **1.00** |
| **均值** | **0.60** | **0.63** |

终点上 MEA≈baseline,且 seed 间方差巨大(0.4↔1.0 来回翻)。**单次 10-episode eval 的 std≈0.15,终点被噪声完全主导,不能单看终点。**

## 3. 三 seed 平均曲线(权威值,来自 `wandb.Api`,各 100 点全 finished)

| env_steps | 5k | 10k | 15k | 20k | 25k |
|---|---|---|---|---|---|
| MEA (n=3) | 0.00 | 0.33 | 0.17 | 0.53 | 0.67 |
| BASE (n=3) | 0.00 | 0.27 | 0.37 | 0.53 | 0.77 |
| **MEA − BASE** | +0.00 | +0.07 | **−0.20** | +0.00 | **−0.10** |

曲线图 `screening_mea_vs_base.png`:蓝(MEA)红(BASE)**几乎完全重叠**,±std 阴影彼此淹没;中段(10k–18k)baseline 略高,末段收敛到 0.6–0.7 不分伯仲。**MEA 无优势,若有则略负。**

> ⚠️ **数据来源修正**:早先我从本地 `.wandb` 文件离线解析得到「MEA 中段领先」的结论是**错误**的——wandb datastore 离线读取会在中途抛 checksum/IndexError **提前截断**,BASE s1/s2 曲线被砍短才显得低。上表用的是 `wandb.Api` 的**完整**曲线,以此为准。

## 4. 结论

- 代码 bug 已修(A、F),所以这是对**方法本身**的干净读数。
- **两个 regime 都零收益**:
  - 本次 `mea=12` / demo=15(稀缺):MEA ≈ baseline,略负。
  - 早先 `mea=4` / demo=80(充裕):两臂都饱和到 100%,无差别。
- 即 **MEA 对称增强在当前设置下没有价值**。

### 为什么(根因 B / 冗余 E)

> **actor/critic 本身就是 C4 等变的。** 网络对 C4 旋转的泛化是**结构内建**的;再用 C4 对称变换增强数据,等于喂它「已经会的东西」,信息增量≈0。**在等变网络上做对称增强是自我冗余**——不是 bug,是方法与网络架构撞车。

这正是 [[mea_augmentation_issues]] 里搁置给 mea_v2 的根因 B(增强只覆盖抓取前接近段)与 E(与等变网络冗余)。

### 有效性边界(注意)

- n=3、eval 仅 10 episode、方差极大 → 「打平」本身也是软结论,但**至少可以排除 MEA 有强正收益**。

## 5. 下一步:决定性验证实验(便宜、干净)

**把 MEA 接到「非等变」网络上跑**(`--actor_type normal --critic_type normal`),对比 MEA vs baseline,同样 data-scarce 3 seed:

- 若 normal 网络上 MEA **明显领先** → **冗余假设坐实**:MEA 的价值只在网络**没有**内建对称时才有;等变网络上注定 ≈ baseline。→ 论文叙事变成「有条件的正结果」。
- 若 normal 网络上 MEA **依然** ≈ baseline → 问题更深(增强只覆盖抓取前接近段 = 根因 B),需走 **mea_v2** 重设计增强范围。

> 这条实验比继续加 seed / 提 `num_eval_tasks` 更有信息量——后者只是把「打平」收紧成统计显著,不会凭空造出 MEA 的优势。

### 状态
- 筛选实验(equi 网络):**完成,负结果**(本文件)。
- normal 网络对照:**TODO**(建议下一个跑,`bash/bench_train.sh` 加 `--actor_type normal --critic_type normal` 变体)。
- mea_v2 增强范围重设计:研究方向,视上面实验结果决定是否启动。

---

## 6. mea_v2 反射 A/B 实验(2026-07-11 完成)

设计见 [[design_mea_v2]] §8。同 data-scarce 协议(demo=15、500 iters、equi 网络、3 seed)。**baseline/V1 复用 §1-4 的 scr_ 数据**,只新跑 V2 两臂:

- **V2+REFL**:mea_v2 `global` 连续旋转 + 50% 镜像(`reflect_prob=0.5`)。
- **V2ROT**:mea_v2 `global` **仅**连续旋转(`reflect_prob=0`,消融)。

### 结果(AUC = 整段训练平均成功率;wandb.Api 权威完整曲线)

| arm | s0 | s1 | s2 | **mean** | std |
|---|---|---|---|---|---|
| BASE | 0.323 | 0.353 | 0.487 | **0.388** | 0.071 |
| V2+REFL | 0.411 | 0.376 | 0.223 | **0.337** | 0.082 |
| V1 | 0.387 | 0.188 | 0.384 | **0.320** | 0.093 |
| V2ROT | 0.121 | 0.312 | 0.237 | **0.223** | 0.079 |

**排名:BASE > V2+REFL ≈ V1 > V2ROT。** 配对差(同 seed):

| 对比 | per-seed | mean | wins |
|---|---|---|---|
| **V2ROT − BASE** | −0.20 / −0.04 / −0.25 | **−0.164** | **0/3** |
| V2+REFL − V2ROT | +0.29 / +0.06 / −0.01 | +0.113 | 2/3 |
| V2+REFL − BASE | +0.09 / +0.02 / −0.26 | −0.051 | 2/3 |
| V1 − BASE | +0.06 / −0.17 / −0.10 | −0.068 | 1/3 |

### 结论(注意别被 `V2+REFL−V2ROT=+0.47@25k` 的标题误导)

1. **唯一稳健的信号:纯旋转增强(V2ROT)伤害性能**——3/3 seed 低于 baseline(mean −0.164)。不是「冗余=中性」,而是**净负**。最可能机理:连续任意角旋转要插值重渲染 occupancy,注入**伪影**,而旋转本身又被 C4 等变吸收(冗余)→ 脏 + 无新信息 → 拉低。
2. **反射修复了旋转的伤害,但没超过 baseline**:V2+REFL 把 V2ROT 从 0.223 拉回 0.337 ≈ BASE 0.388(仍略低)。反射的 +0.113 是**相对被污染的 V2ROT**,不是相对 baseline。含 reflection 的两臂(V2+REFL、V1)都 ≈ baseline;纯旋转(V2ROT)才掉下去。
3. **没有任何增强臂打赢 baseline**。反射机理上是对的(补 C4 缺的镜像),但在这个任务+网络+相机组合下,增强的收益填不平其成本(伪影 + 稀释根因 D)。
4. n=3、eval 10-episode,除「V2ROT 伤害」(3/3)外其余都在噪声内。

### 研究含义 & 下一步

这是一条**干净的负结果证据链**:在 block_pull(关系型 reward)+ C4 等变网 + 夹爪居中相机下,合法对称增强要么冗余(旋转)、要么低多样性(反射只有 ×2),**都无法超过等变网络的内建对称**。

**决定性下一步:normal(非等变)网络对照**(见 §5,现已具备工具)。若 V2+REFL 在 normal 网上明显 > baseline,而 equi 网上打平 → 论文叙事:**「数据增强 = 用样本换架构对称;架构已有对称时增强无用甚至有害」**。这比在 equi 网上继续加 seed 更有信息量。

- 反射 A/B(equi 网):**完成,本节**。
- normal 网络对照(baseline vs V2+REFL,`--actor_type normal --critic_type normal`):**下一个跑**,见 `bash/bench_normal.sh`。

---

## 7. normal 网络对照结果 + 机理诊断(2026-07-13)

> ⚠️ **2026-07-17:本节(及 §6)所有 v2 增强臂的解读已被 §7.b 作废**——增强 demo 的动作标签坐标系错位(错 2θ / 镜错轴)。BASE 与 V1 曲线仍有效。

6 个 run(3 seed × {BASE-normal, V2+REFL-normal})已全部跑完。**待权威 AUC**(`python bash/analyze_screening.py --tag nrm_d15_s`),但本地 checkpoint(`agent_<iter>_perf<X>.pt`,即 10-episode eval 成功率)后段(iter 390–520)预览已很清楚:

| seed | BASE-normal(后段均值) | V2+REFL-normal(后段均值) |
|---|---|---|
| s0 | ≈0.35 | ≈0.05 |
| s1 | ≈0.55 | ≈0.29 |
| s2 | ≈0.42 | ≈0.23 |
| **均值** | **≈0.44** | **≈0.19** |

**3/3 seed,BASE 全胜(≈ −0.25),比 equi 网上输得更惨(equi −0.05)。** → **「增强替代架构对称」假设被证伪**:即便网络没有内建对称,V2+REFL 依然大幅落后 baseline。且注意:合成 episode **不计入 env_steps**(`learner.py:695` 只对 `sym_state==0` 累加),所以同一横轴点上增强臂有 13× 数据(195 vs 15 条)还是输——记账没占便宜,负结果更硬。

### 机理诊断(实证,`bash/diagnose_mea_v2.py`,CPU,抓 1 条真实 pull episode)

| 测试 | 结果 | 含义 |
|---|---|---|
| **T1 恒等回环** θ=0 | mean\|Δ\|=**0**,diff 图纯黑 | 重渲染管线**无损/幂等**——**排除**「伪影注入」假说 |
| T3 窗口裁剪 | frac_clipped ≈ 0 | 内容没转出 [-0.2,0.2]² 窗口 |
| T4 占用面积 | ratio ≈ 1.00 | 刚体旋转面积守恒,无插值空洞 |
| **T2 夹爪脱离中心** | 见下方**更正后**量级 | ⚠️ **病因,但量级此前记错** |

**根因 = 旋转中心选错(不是渲染,不是裁剪)。** 相机 `camera_center_xyz` **夹爪居中**:实测夹爪世界坐标**每帧恒为 ≈(0,0) = 图像正中**(100% 真实/eval 观测都如此)。但 `generate_sym_v2` 绕场景质心旋转,把夹爪推离图心——这种「夹爪不在中心」的观测在真实 rollout / eval 里**从不出现** → 增强数据是**离流形(off-manifold)**的。

#### ⚠️ 量级更正(2026-07-13,`viz_mea_v2_trajs.py` 用真实训练代码路径实测)

初版诊断(T2)错在:它绕**逐帧**场景质心旋转,得到「4→44 px 随 episode 增长」。**实际代码**(`sym_v2.py` global 分支)用的是 **obs[0] 的场景质心 q0,整条 episode 同一个变换** → 夹爪偏移是**每条增强 episode 的一个常数** |(I−FR)q0|(实测与闭式预测逐条吻合:本条被抓 episode |q0|=0.009 m,三个增强克隆偏移 4.4 / 8.6 / 6.3 px vs 预测 4.5 / 8.5 / 6.1;ORIG 恒 0.2 px)。

但被抓的这条 episode 恰好 q0 极小。**6 次 fresh reset 实测 |q0| = 4.5~32.8 px(mean 17.8 px)** → 典型偏移 ≈ 1.27·|q0| ≈ **10~40 px,worst-case 66 px**(θ=180°、半宽 100 px)。即:离流形分量真实存在、常见几十像素,但形式是**每条 episode 一个常数平移**,不是初版说的「随 pull 递增、关键帧最脏」。

**相机系论证(修复依然免费且严格正确)**:夹爪居中相机下,任何全局世界旋转在相机系里恒等于**绕原点(=图心=夹爪)**的旋转;绕 q0 旋转 = 绕原点旋转 + 假的常数平移 (I−FR)q0。锚点改为夹爪/原点即可把离流形分量精确清零。

**诚实的不确定性**:CNN 平移等变可能部分吸收常数平移,所以「几十 px 常数偏移」对性能的实际伤害**未证实**;92% 合成 / 仅 15 个独立场景的**稀释**(根因 D)仍是并列嫌疑。锚点修复 A/B 依然是最便宜的判决实验,但对其收益的预期应下调。

图:`context/plan/viz_mea_v2_trajs.png`(1 条真实轨迹 + 3 条增强克隆,含动作箭头/夹爪标记),`diag_mea_v2.png`(初版,注意其 T2 用了逐帧质心,量级偏大)。

---

## 7.b 真·根因(2026-07-17,**用户发现**):增强动作标签坐标系错位 → v2 全部实验臂作废

**发现路径**:用户看 §7 的轨迹可视化,指出**绿色动作箭头与 ORIG 和增强轨迹的表观运动都不符**。追查确认这不止是画图问题,而是训练管线的真 bug。

### 实测证据(`bash/check_pc_action_frame.py`)

夹爪居中相机下,**静止物体在 pc 帧反向运动** `d_obj_pc = −s·M·a[1:3]`。用接近段(物体静止、可见)拟合:

```
M = [[0, 1], [1, 0]]   (x/y 对调),det(M) = −1,scale ≈ 0.78~0.92
```

即 **pc/图像系是世界系的 x/y 对调镜像副本**(俯视相机的标准现象)。`DummyEnv.step` 的 `(−a[2], −a[1])` 物体反向平移**早已编码**这一约定——v1 走的就是它。

### 三个错(全部在 v2 专属代码里)

| 项 | 应该(M-共轭) | 旧 v2 代码 | 后果 |
|---|---|---|---|
| 旋转 | pc 转 +θ ⇒ 动作转 **−θ**(det(M)=−1 翻手性) | 动作转 **+θ** | **每条增强 demo 动作标签错 2θ**,θ~U(−π,π) |
| 镜像 | pc x-mirror ≡ 世界 y-mirror ⇒ 翻 **a[2]** | 翻 **a[1]** | 镜像 demo 动作错 180° 旋转 |
| 锚点 | **pc 原点**(=夹爪=图心;任何全局世界旋转在 pc 系的像) | 场景质心 q0 | 夹爪偏心 (I−FR)q0(§7.a,次要) |

**为何 §8 的「验证」没抓到**:`test_sym_v2.py` 只拟合 action↔世界系(gripper_pos),而增强旋转的是 pc 系;夹爪永远钉在 pc 原点,恰好是**唯一测不出 M 的实体**。

**v1 为何免疫**:`generate_sym3` 先变换动作、再用 `DummyEnv.step`(内置正确 M)**回放动作生成观测帧** → obs/action 构造上自洽;v2 是几何旋转观测 + 另行变换动作,两边约定不一致才炸。

### 对既有结果的重新解读

- **§6 V2ROT/V2+REFL、§7 V2+REFL-normal 全部作废**:那些臂 92% 的专家数据带系统性错误动作标签,其负结果反映「错标签的危害」,不是「对称增强的价值」。
- 「旋转增强主动有害(V2ROT 0/3)」从此有了最简单的解释:**标签错 2θ**。
- **BASE(equi/normal)与 V1 曲线仍有效**;BASE-normal vs BASE-equi 的架构价值对比仍可用。
- §7.a 的 anchor 偏移是并存的次要缺陷,同批修复。

### 修复与验证(本 commit)

- `transform_action_se2`:M-共轭(默认 `action_sign=−1`,mirror 翻 a[2]);`generate_sym_v2`/`Sym`/yml:`anchor=origin`。
- `check_pc_action_frame.py` 断言通过:新变换 vs 要求 **误差 0**(200 随机抽样);旧变换最大偏差 0.099 ≈ 2|a|。
- 修复后可视化 `context/plan/viz_mea_v2_trajs_fixed.png`:箭头随场景**共旋/共镜**,增强轨迹夹爪 **0.2 px 居中**(与真实观测一致);对比旧图 `viz_mea_v2_trajs.png` 可见旧箭头系统性错向。

### 下一步

**用修复后代码重跑 v2 A/B**(equi 网,3 seed × {V2FIX+REFL, V2FIX-ROT},新前缀 `v2fgr_`/`v2fg_`,`bash/bench_train.sh` 已更新;BASE/V1 复用)。分析:`python bash/analyze_screening.py`(已支持 6 臂 + label-fix 对照行)。这将是**第一次干净测量**「合法对称增强 vs C4 内建等变」;若 V2FIX 仍 ≈ BASE,冗余结论才真正成立;若 V2FIX+REFL > BASE,反射假设复活。

### 关键推论:这其实指向一个**可修复的实现缺陷**,而非方法死局

- 绕**夹爪/图心**旋转**整个场景**:① 夹爪留在中心(**保持 on-manifold**);② 刚体 → block 间距不变 → **关系型 reward 严格不变**;③ 非 90° 连续角 → C4 网**没有**的多样性。**三者同时满足** → 这才是本任务+相机下唯一合法且非冗余的旋转增强。
- 当前代码用 `scene_centroid_xy(obs[t])` 当锚点 → 破坏 ①,是**实现 bug**,不是根因 B/E。之前 design 文档把「global SE(2) ≡ C4 内建对称」当成冗余,其实两者旋转中心不同(场景质心 vs 图心),真实后果是**离流形**而非干净冗余。

### 下一步(便宜、判决性)

**改锚点重跑 V2**:`se2_about` 锚点从 `scene_centroid_xy` → 夹爪 xy(≈原点),equi 网 3 seed A/B(V2gr-fixed vs BASE)。
- 若 fixed 版**追平/超过 baseline** → 之前的负结果主要是**旋转中心 bug**造成的离流形损伤,方法本身可用(至少不再有害)。
- 若 fixed 版**仍 ≈ baseline** → 对 C4 网,连续角+反射的增量确实填不动内建对称的冗余(回到根因 E),但这次是**干净的**冗余结论。

- 反射 A/B(equi 网):完成,§6。
- normal 网络对照:**完成,本节**(证伪「增强替代架构」)。
- 机理:**旋转中心 bug → 夹爪离流形**(`diagnose_mea_v2.py` 实证)。
- 锚点修复版 A/B:**TODO,推荐下一个跑**。
