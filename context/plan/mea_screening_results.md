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
