# MEA v2 当前状态快照

- **更新**:2026-08-07 · 分支 `meav2` · 目标:MEA v2 重投 ICLR(novelty=超过 baseline=同一个机制)
- **配套**:计划 [[b2_plan]]、证伪全记录 [[falsification_and_pivot]]、策略 [[iclr_strategy]]、理论 [[new_idea]]、任务选型 `sym_manip_tasks_baselines.md`。待办见同目录 `todo.md`。

---

## 0. 一句话现状

augmentation-beats-baseline 路线(v1→v2 的数据增强侧)**已在两个栈上先后证伪**;方向已转到 **B2 = context 进网络**;但 B2 只在"extrinsic-equivariance floor 真实存在"的任务上才可能涨,而现有证据(2×2)指向 **Square 没有 floor**。当前正在跑**决定性一枪**:宽朝向 **Square_D2** 上的 floor 探测(C8 vs C1 的 gap 是否收窄)。

---

## 1. 走到这里的完整弧线(时间序)

| 阶段 | 结论 | 记录 |
|---|---|---|
| **RL 侧**(equi-rl-for-pomdps / BulletArm / Equi-RSAC) | mea ≈ baseline,**null**;根因 = 与 C4 等变网**自我冗余**(全局对称增强对已等变网络无信息) | [[mea_screening_results]]、[[mea_augmentation_issues]] |
| **转 diffusion**(EquiDiff on MimicGen Square) | 换宿主:算力可行 + 落在审稿人期待的 EquiDiff 对比上 | `sym_manip_tasks_baselines.md §3` |
| **对抗证伪** | 3 条增强臂 2 条死(global 冗余、approach 被相对帧碾压);唯一活的内核 = 离散物体点群 **keyed-C4 ENGAGE 增强** | [[falsification_and_pivot]] §1 |
| **per-k 可行性**(2026-07-31)| 抓手柄翻 180/270° 够不到 → **C4 在 Panda 上塌到可达子集 ~{0°,90°}**(+90° 增强,非干净子群) | falsification §29 |
| **完整 2×2**(2026-08-06)| **keyed-C4 对两个宿主都惰性**:DP 0.736→0.748(+1.2pt)、EquiDiff 0.866→0.844(−2.2pt),**全在噪声内 → augmentation 路线证伪** | falsification §2026-08-06 |
| **方向决策**(2026-08-07)| 转 **B2(context 进网络)**;augmentation 线只留作 negative-analysis 材料 | 本文件 / [[b2_plan]] |
| **B2 计划 + 任务决策** | floor 必须先存在;2×2 是 floor 探测器读数 → Square 无 floor;选任务 **A = 宽朝向 Square_D2** 找臂运动学 floor | [[b2_plan]] |
| **当前** | 准备 Square_D2 abs 数据(下 core → 转 abs)→ 跑 C8-vs-C1 floor 探测 | `todo.md` |

---

## 2. 已确立的关键事实(别重犯)

- **报告口径**:`test/mean_score` 的**末 10 档平均**(单档=50 rollout,二项 SE≈±5pt;末10 压噪)。
- **训练必须跑满**:schedule = `num_epochs=50000/n_demo`(n_demo=100 → **500 epoch**),`rollout_every=1000/n_demo`(→ 每 10 ep,满档 50 次 eval)。**任何 arm 必须到 `max_epoch=499` 再比**——首个 base run 在 ep65/500 被 Ctrl-C → 假 delta 0.84-vs-0.34(踩过一次)。
- **群阶 N**:`diffusion_equi_unet_cnn_enc_policy.py:33 N=8`(=C8),**硬编码、不在 yaml**;yaml 的 `n_groups:8` 是 GroupNorm、无关。改等变阶用 hydra `+policy.N=4`(待验证构造函数吃)。
- **无 floor 的证据**:EquiDiff-C8(0.866) ≫ DP-C1(0.736),+13pt 单调 → 加等变净有益 → Square 无 extrinsic floor(GIC:全局 z 旋转全程合法)。
- **Curie no-op**:不变标量喂等变网 = 仍全局等变 → 结构上破不了对称 → "最小 B2(相位标量写 layer1)"是死路,只配当反面对照 B2a。真 escape = **群退化 / 自由通路**。

## 3. 基建 / 环境(远端 `ssh ubuntu-ws`)

- **两 conda env**:`equidiff`(训练/渲染,pointW robomimic 8aad5b3)+ `mimicgen_datagen`(datagen,ARISE robomimic d0b37cf)。robosuite b9d8d3de(v1.4.1)+ mujoco 2.3.2 两边相同 → 状态可确定性重放。
- **激活**:先 `source ~/ssd/miniconda3/bin/activate` 再 `conda activate equidiff`(conda 不在默认 PATH)。
- **路径**:训练根 `~/ssd/code/research/mea/equidiff`;数据 `data/robomimic/datasets/`(有 `square_d0`〔1000 demo,带图像〕、`square_mea_base`、`square_mea_keyed`;`square_d2` 目录空)。mimicgen 下载脚本 `~/ssd/code/research/mea/mimicgen/mimicgen/scripts/download_datasets.py`。
- **wandb**:`WANDB_MODE=offline` + `logging.mode=offline`;显存用 `dataloader.batch_size=64`(≈11G,别改 `enc_n_hidden` 以保公平)。
- **读结果**:`mea_readout.py`(扫 `data/outputs/**/logs.json.txt`,认 equi/dp + base/keyed,打末10/peak/final)。
- **Claude 沙箱够不到远端 GPU** → 训练只能你发命令、我读数;subagent/workflow 帮不上训练本身。

---

## 4. 当前 2×2(Square, n_demo=100, 1 seed, 末10)

| host | base | keyed | Δ |
|---|---|---|---|
| DP(C1)| 0.736 | 0.748 | +0.012 |
| EquiDiff(C8)| **0.866** | 0.844 | −0.022 |

EquiDiff−DP = +13pt(等变优势,复现文献 → setup 正确)。两个 keyed Δ 均噪声内 → keyed 惰性。
