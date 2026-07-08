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
- normal 网络对照:**TODO**(建议下一个跑,`bench_train.sh` 加 `--actor_type normal --critic_type normal` 变体)。
- mea_v2 增强范围重设计:研究方向,视上面实验结果决定是否启动。
