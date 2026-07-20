# 原版 equi-rl-for-pomdps baseline 与 fork(mea_v2)可比性审计

2026-07-20。起因:准备用 `ext/equi-rl-for-pomdps-original` 复现原始 baseline,
作为 mea_v2 的对照。审计问题:两边的成功率曲线能否直接对比?

方法:5 路并行审计(任务动力学 / 观测 / 协议 / learner 循环 / 低 demo 缩放)
+ 对每条"blocking"差异做对抗式复核(要求主动 refute)。以下结论均已复核。

## 结论(一句话)

**不能直接对比。** 任务动力学、奖励、协议键都一致,但**网络看到的观测是不同模态**——
原版是 pybullet 原生深度高度图,fork 是点云重渲的占据图。因此:

- 「mea_v2 vs baseline」的论断 → 必须用 **fork 自己的 BASE 臂**(`mea_expert=0`),
  这是合法对照(见 §3);
- 「我们复现了原论文」的论断 → 用 `bench_original.sh`,但它的数值**不能**和 fork 的
  曲线画在同一张图上。

## 1. 确认成立的差异(已复核,refuted=false)

| 差异 | 原版 | fork | 影响 |
|---|---|---|---|
| **观测模态(核心)** | channel0 = pybullet 原生深度高度图,84×84 直接进编码器 | channel0 = `obs['occup_image']`,即 Occup wrapper 从点云重建的占据图,cv2 降到 84×84 | **阻断对比** |
| 深度编码 | `getHeightmap` 返回 `abs(depth - max(depth))`,再 `-heightmap + gripper_pos[2]`(夹爪相对高度) | 不同编码路径 | material |
| 夹爪像素填充值 | `heightmap[gripper_img==1] = 0` | `= 1 - gripper_pos[2]` | material |
| 渲染分辨率 | img_size 84 | img_size 600(再降采样) | 渲染成本/点云保真度;**网络输入形状不变** |
| 渲染后端 | `ER_TINY_RENDERER`(CPU) | `ER_BULLET_HARDWARE_OPENGL` + 分割掩码 | minor,深度值不保证逐位相同 |

证据:`rgbd_sym/env/wrapper/gym_regularizer.py:27-38`(occup 分支 + resize 到 84);
`rgbd_sym/api.py:10-25`(PomdpEnv → Occup → Sym → GymRegularizer);
`rgbd_sym/env/embodied/pomdp/close_loop_env.py:215-240` vs
`ext/.../close_loop_envs/close_loop_env.py:215-226`(夹爪填充值)。

注:`obs_type: image` 分支存在,切过去会接近原版的高度图路径(但夹爪填充值仍不同)。
若要做"同模态"对照,这是最小改动的入口。

## 2. 被复核**推翻**的差异(不必担心)

- **"观测张量变成 (2,600,600)"** — 假。`GymRegularizer` 硬编码 `s=84` 并返回固定
  `Box(shape=(2,84,84))`,learner 从最外层 wrapper 读 shape,两边网络输入都是 84×84。
  600 只影响渲染成本和点云保真度。
- **"初始 20 条随机 rollout 在低 demo 下是 fork 特有的坑"** — 假。相关文件在两仓库
  逐字节相同,不是 fork 差异(但作为**实验设计**问题仍然成立,见 §4)。
- **"专家采样比例是 fork 特有差异"** — 假,机制两边一致。
- **"MEA 稀释是 blocking"** — 不成立。`mea_expert_eps` 是 flag 门控,
  `configs/block_pull/rnn-equi-all.yml:23` 设为 0;为 0 时代码走 `set_sym(False)`、
  `num_rollouts + 0*num_rollouts`,与原版行为等价。**这正是 BASE 臂合法的原因。**

## 3. 为什么 fork 的 BASE 臂是合法对照

审计确认:`mea_expert=0` 时 fork 的 learner 路径与原版**行为等价**(不进入任何增强
代码)。所以 `scr_base_` 系列 run 与 `v2fgr_` 系列共享完全相同的观测管线、协议、
种子——配对差值是干净的。跨仓库对比才有问题,仓库内对比没有。

## 4. 遗留的实验设计注意点(非代码缺陷,两仓库共有)

1. **初始随机 rollout 不随 demo 数缩放**:`num_init_rollouts_pool: 20` 固定。
   d80 时专家占 80/100,d15 时 15/35,d5 时 **5/25**——"数据稀缺"实验里有一部分
   效应来自随机 rollout 占多数,而非 demo 数本身。`bench_lowdemo.sh` 保持默认值
   以便与 d15 筛选可比,但解读时须记住这一点。
2. **专家 rollout 计入全局 step 预算**:`while _n_env_steps_total < n_env_steps_total`,
   而 `n_env_steps_total = 50*(20+num_iters)`。demo 越少,留给在线阶段的预算越多。
   量级:ITERS=500 → 预算 26000 步;d15 专家约 225 步、d5 约 75 步,差约 0.6%,
   **可忽略**。但 d80(约 1200 步)与 d5 相比就有约 4% 偏移,跨 demo 数比较时留意。
3. **协议不同**:`bench_original.sh` 用 readme 的 80 demos / 800 iters,
   `bench_train.sh` 用 15 demos / 500 iters。x 轴长度和 offset 都不同。

## 5. 审计中顺带发现的 fork 缺陷(不影响 block_pull 训练)

- `rgbd_sym/env/embodied/pomdp/close_loop_env.py:351-356`:`camera_center_xyr` 分支的
  `depth` 赋值被注释掉但 `return depth` 还在 → `NameError`。block_pull 用的是
  `camera_center_xyz`,不触发,但该 view type 已不可用。
- `rgbd_sym/env/embodied/pomdp/sensor.py:59-62`:`getPointCloud` 里插入了无条件的
  open3d `draw_geometries`(阻塞调用)。block_pull 路径没调用它
  (`close_loop_env.py:367` 已注释),但任何人调用都会卡住。
