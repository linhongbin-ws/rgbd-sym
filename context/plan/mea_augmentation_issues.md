# MEA 数据增强问题分析:为什么 mea ≈ baseline

- **日期**：2026-07-03
- **分支**：`meav2`
- **对照**：`configs/block_pull/mea-rnn-equi-all.yml`(`mea_expert_eps: 12`) vs `configs/block_pull/rnn-equi-all.yml`(`mea_expert_eps: 0`);两份配置**逐行 diff 只差 `mea_expert_eps` 一处**。
- **现象**：开启 mea 后最终性能常常与 baseline 基本持平。
- **验证方式**：5 路独立读码代理找问题 → 逐条对抗式验证 → 人工复核头号问题(证据均为核实过的 `file:line`)。
- **进度**：**A 已修复**(2026-07-03,见下方 A 节「✅ 已修复」);B(方法 limitation)暂缓;C/D/E/F 待处理。

---

## 结论(TL;DR)

mea ≈ baseline **不是调参问题**,而是两个结构性缺陷叠加:

1. **【A / 高 ✅ 已修复】对称变换后的动作 `new_sym_actions` 根本没进 buffer** —— 整套 `action_sym / action_inverse` 对训练是死代码。
2. **【B / 高】每条增强轨迹只改了抓取前约 2 帧接近段,抓取+拉动整段逐帧照抄** —— 12 条增强几乎是原轨迹的副本。

即使修好 A、B,下面的 C/D/E 仍会限制增益上限。

---

## 根因排序

| # | 问题 | 严重度 | 判定 | 为什么导致 ≈ baseline |
|---|---|---|---|---|
| **A** | 对称变换后的动作 `new_sym_actions` 没进 buffer | 🔴 高 | ✅ **已修复** | mea 的价值 =(变换观测,变换动作)配对,该配对被两头掐断 |
| **B** | 只增强抓取前 ~2 帧,抓取+拉动段照抄 | 🔴 高 | CONFIRMED | 12 条增强彼此/与原轨迹几乎相同,几乎无新信息 |
| **C** | `--algo sac` 不用 `expert_masks`(无 BC/模仿损失) | 🟡 中 | CONFIRMED | 专家(含增强)只当普通 replay,缺“示范驱动”的放大通道 |
| **D** | buffer 被 ~92% 增强近重复灌满、均匀采样、更新预算不变 | 🟡 中 | CONFIRMED | 相同梯度步数摊在被重复数据主导的 buffer 上,稀释真实 80 条 |
| **E** | mea 旋转与等变网络 + seq_rot 旋转增强冗余 | 🟡 中→低 | PARTIAL | 最大自由度(旋转)已被免费提供,边际信息本就小 |
| **F** | 变换质量(每步 ±1rad yaw 噪声、DummyEnv 无物理/欠旋转、O(n²) 重渲染) | ⚪ 低 | PARTIAL | 只影响那 ~2 帧,且因 A 动作被丢弃而基本无害 |

已被 **驳倒(REFUTED)** 的猜想:`sym_end_step==0` 全等副本(block_pull 中夹爪从 z=0.2 复位,不会一开始就 <0.15);接缝处 reward 断裂(重建锚定在抓取位姿、按构造连续)。

---

## A(头号 bug,逐行复核)—— 对称动作是死代码

**链条**：

1. `generate_sym3` 算出的 `new_sym_actions` 只被塞进 `obs['sym_action']`
   —— `rgbd_sym/env/wrapper/sym.py:126`
2. wrapper 顺序:`Sym` 在内、`GymRegularizer` 在外
   —— `rgbd_sym/api.py:18-19`(`env = Sym(...)` 后 `env = GymRegularizer(...)`)
3. `GymRegularizer._obs_proc` 把 dict 观测压成裸 `(2,84,84)` 数组返回,**丢弃 `sym_action` 及整个 dict**
   —— `rgbd_sym/env/wrapper/gym_regularizer.py:31-49`
   —— 全仓 `obs['sym_action']` 只被赋值、从不被读取
4. learner 存进 buffer 的动作**恒为** `query_expert(expert_ep_cnt)`,从不读 `obs['sym_action']`
   —— `rgbd_sym/rl/learner.py:621-648`(`action=query_expert(...)` → `act_list.append(action)` → `add_episode(actions=act_buffer, expert_masks=1)`)
5. replay 增强 episode 时 `Sym.step` 不 step 底层 env → 底层仿真冻结在上一条真实轨迹末态,`query_expert` 读的是冻结态
   —— `rgbd_sym/env/wrapper/sym.py:70-73`

**后果**:每条增强 episode =（重渲染的对称观测图）+（来自冻结仿真的错位动作)。在 SAC 里这些 `(s, a, r, s')` 甚至相当于告诉 critic「从对称状态做那个错位动作能拿到照抄来的成功回报」——不仅无益,还可能轻微污染 critic。

### ✅ 已修复(2026-07-03)

**实现**(比原方案更小,未改 `GymRegularizer`):

- `rgbd_sym/env/wrapper/sym.py`:新增 `_sym_action` 跟踪(`__init__` + reset/step 的 4 个分支各 `self._sym_action = obs['sym_action']`)与 `sym_action` 属性(与 `sym_state` 并列,经 `BaseWrapper.__getattr__` 透传,故**无需改 `GymRegularizer`**)。
- `rgbd_sym/rl/learner.py`:`collect_expert_rollouts` 与 `collect_rollouts` 在 `sym_state==1` 时 `store_action = ptu.FloatTensor([self.train_env.sym_action])` 存入 buffer,替代 `query_expert`;真实 episode(`sym_state==0`)保持不变。

**为何无需单位换算**:实测 `ptu.get_numpy` 与 tensor **共享内存**,`block_pulling.step` 的原地缩放会回灌到 learner 的 `act_list` → 真实 episode 的 buffer 动作本就是**物理量**,`new_sym_actions` 也是物理量,同一空间,直接存即可。

**验证**(`--num_expert_episodes 1 --mea_expert 2`,`WANDB_MODE=offline`):

- `query_expert` 在 replay 全程恒为 `[1, -0.089, 1, -0.005, -0.597]`(坐实「底层仿真冻结、动作陈旧」)。
- 存入 buffer 的动作现在是逐步不同、两条增强轨迹也不同的**变换后动作**(物理尺度 xyz≈±0.05)。
- 21 步 ×2 增强轨迹全部跑通,无崩溃,对齐正确(每条增强 = `L-1` 个动作,与真实等长)。

---

## B —— 只增强 ~2 帧,其余照抄

- `sym_end_step` = 夹爪首次 `gripper_pos[2] < 0.15` 的帧 —— `rgbd_sym/tool/sym.py:551-556`(阈值 `sym_end_z_thres=0.15`)。
- 夹爪从 `z=0.2` 复位、每步降 ≤0.05 → 约第 2 步越过 0.15;block_pull 专家轨迹约 10–15 步。
- 拼接:`new_sym_obs = obs[:0] + 重渲染的~2帧 + obs[sym_end_step:]`;`new_sym_actions = ... + origin_actions[sym_end_step:]`(照抄)—— `rgbd_sym/tool/sym.py:584-585`。
- reward/done/info 逐帧照抄原轨迹 —— `rgbd_sym/env/wrapper/sym.py:122-124`。

**后果**:任务关键的抓取+拉动段在 12 条增强里完全相同 → 近重复,几乎无新样本。

**修复(P1)**:把对称变换扩展到**整条轨迹**(像 seq_rot 那样一致旋转 block+抓取+拉动),或调大 `sym_end_z_thres` 让更多帧被重渲染。

---

## C —— `--algo sac` 忽略 `expert_masks`(无 BC 损失)

- `sac.py` 的 `actor_loss` 收了 `expert_masks` 却从不使用,`policy_loss = -min_q + alpha*log_probs`
  —— `ext/equi-rl-for-pomdps/policies/rl/sac.py:220,242-246`
- 只有 `sacfd.py` 有模仿损失 `imitation_loss = ((new_actions[:-1]-actions[1:])*expert_masks[1:])**2`
  —— `ext/equi-rl-for-pomdps/policies/rl/sacfd.py:244-247`
- 配置 `policy.algo_name: sac`;所有专家/增强 episode 以 `expert_masks=1` 入 buffer(`learner.py:675`)。

**后果**:示范数据只走 replay,没有「示范驱动」通道。注意 baseline 也是 sac,这不是 mea-vs-baseline 的差异项,但它掐掉了让好示范放大作用的通道。

**修复(P2)**:mea 实验改用 `--algo sacfd`,让 `expert_masks` 驱动 BC(前提是先修 A,让动作标签正确)。

---

## D —— buffer 稀释,且 mea 不增加更新步数

- `num_rollouts = 80 + 12*80 = 1040` —— `rgbd_sym/rl/learner.py:590`;每条再被 seq_rot ×5(原始+4 旋转副本)。
- 采样均匀:`sample_weight_baseline=0` → 每 episode 权重为 1,无专家/真实加权 —— `ext/equi-rl-for-pomdps/buffers/seq_vanilla.py:134-188`。
- 更新步数由 `_n_env_steps_total` 决定,而只有真实 episode(`sym_state==0`)推进它 —— `learner.py:682-683, 437-438`。
- `mea_normal_eps=0`,增强只在一次性初始采集发生;`1040×5×~50` 步溢出 `1e5` FIFO buffer → 前期被增强数据灌满,后期被在线数据逐步挤出。

**后果**:相同更新量摊在被近重复数据主导的 buffer 上 → 最好中性,可能轻微稀释真实 80 条。

**修复(P2)**:给专家样本加采样权重(`sample_weight_baseline>0` 或专家优先采样),或按总存储量而非仅真实步数放大更新次数。

---

## E —— 与已有对称机制冗余(部分成立)

- 等变网络(`actor_type/critic_type=equi`, `num_rotations:4`)+ seq_rot(`num_aug_episode:4`,**正确** relabel)已覆盖全局旋转 —— `ext/equi-rl-for-pomdps/buffers/seq_rot.py:79-83`, `utils/helpers.py:373`。
- mea 的旋转分量(`sym_trans_rot ~ uniform(0, 2π)`)高度冗余;但 mea 是绕**固定抓取点**旋转接近方向,并非全局 SE(2) 旋转,所以并非完全冗余。非冗余部分(径向/竖直 ≤25% 缩放 + 重渲染)信息量小,且因 A 被丢弃。
- 判定 PARTIAL:解释「边际增益变小」,不足以单独解释「完全无增益」。

---

## F —— 变换质量问题(低优先级)

- 每步给 yaw 加独立 `uniform(-1,1)*sym_rot`,`sym_rot=1`(config 覆盖了 git `4a1a309` 调到 0.3 的默认)—— `rgbd_sym/tool/sym.py:604`,配置 `sym_rot_low/high=1`。
- DummyEnv 只做刚性点云变换、无碰撞、yaw 欠旋转且只转夹爪云;O(n²) 重渲染每帧从不同真实点云重新种子 —— `rgbd_sym/env/embodied/dummy/env.py:37-64`, `rgbd_sym/tool/sym.py:571-583`。
- 只影响那 ~2 帧,且因 A 动作被丢弃,实际影响很小。

**修复(P3)**:`sym_rot` 调回 0.3、yaw 改为每 episode 一个偏移;DummyEnv 一致旋转所有点云并修正旋转尺度。

---

## 建议:先验证,再修

**便宜的验证实验(确认 A、B)**:

1. 在 `generate_sym3` 打印 `sym_end_step`(预期 ≈2)。
2. 对一条 sym-replay episode,打印真正写进 buffer 的 `actions`,确认它等于 `query_expert` 输出而**不等于** `new_sym_actions`。

**修复优先级**:

- ~~**P0 修 A**~~:✅ **已完成**(2026-07-03)—— 见上方 A 节「✅ 已修复」。
- **P1 修 B**:对称变换扩展到整条轨迹,或调大 `sym_end_z_thres`。
- **P2**:`--algo sacfd` + 专家采样加权。
- **P3**:`sym_rot=0.3` + yaw 改每-episode 偏移;DummyEnv 一致旋转所有点云。

---

## 附:mea 机制回顾(已验证)

- `mea_expert_eps>0` 时,learner 调 `train_env.mea_rollouts(mea)`(设 `Sym._sym_aug_new_eps=mea`)+ `set_sym(True)`;`num_rollouts = N + mea*N`。
- 每跑完 1 条真实轨迹,`Sym._on_end_gt_eps` 用 `generate_sym3` 生成 `mea` 条对称合成轨迹,replay 入 buffer(`expert_masks=1`)。
- 只有真实轨迹(`sym_state==0`)推进 env-step 计数;合成轨迹进 buffer 但不计入环境交互预算。
- `generate_sym3` 只对抓取前接近段做随机对称变换(z/径向/旋转),用独立 `dummy_env` 重渲染观测,抓取点之后原样保留。
