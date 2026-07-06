# Run-efficiency plan (make MEA experiments cheaper under GPU scarcity)

- **日期**：2026-07-05
- **分支**：`meav2`
- **目标**:在**不改变实验结果**的前提下降低每个 run 的墙钟/成本,把紧缺的 GPU 只花在真正的训练上。
- **背景**:mea vs baseline 对照要多 seed 才有信号,而每个 run 的 **prefill(采集+增强)是 CPU 活、GPU 全程闲置**,是最大浪费。相关诊断见 [[mea_augmentation_issues]]。

---

## Part A — prefill 增强的「行为等价」提速(#2,PLANNED,暂不做)

### 现状与实测(2026-07-03)
`generate_sym3`(`rgbd_sym/tool/sym.py`)里重渲染增强轨迹的循环是 **O(n²)**,且**每一步 `dummy_env.step` 都经 `Occup` 重算一次 84³ 占据栅格(~0.1s)**。实测(`num_expert=2 mea=12`,block_pull):
```
sym_end=5  →  dummy_steps=30  →  但每个增强只保留 5 帧
每个增强 render ≈ 2–3.3s；每条真实轨迹 12 个增强 ≈ 26–44s
```
即每个增强算了 ~30 次占据栅格却**丢弃 ~25 次**(循环里每个输出帧从真实点云重新前向模拟,只留最后一帧)。占据栅格是最贵的部分。

**推算完整 prefill**:`num_expert=80 mea=12` → 960 个增强 × ~3s ≈ **~48 分钟**(纯 CPU,GPU 闲置)。

### 提议的修复(行为等价,输出逐比特不变)
- 循环里**会被丢弃的中间帧只做点云刚性变换(廉价),不算占据栅格**;占据栅格只在真正保留的帧上算一次。
  - 实现上:中间步用**裸 `DummyEnv.step`(仅变换点云)**,只对每个 `_i` 迭代保留的那一帧走 `Occup` 算占据栅格。
- 可选进一步:把 O(n²) 的「每帧从真实点云重种子再前向」改成 **O(n) 单次前向**——但需确认与逐帧重种子在数值上一致(刚性变换可复合,理论上等价,需验证),风险略高,先不做。

### 预期收益
占据栅格计算 ~30→~5 次 → **约 6× 提速**。`mea=12` prefill ~48min→**~8min**;`mea=4` ~16min→~3min。

### 验证(必须)
- 固定随机种子,优化前后 `generate_sym3` 产出的 `new_sym_obs` / `new_sym_actions` 数组**逐比特相等**(断言),确认没改坏行为再合入。

### 状态
**PLANNED,暂缓**(用户 2026-07-05 指示先不做)。不改增强语义、不影响对照可比性、与 mea_v2 方向无关。

---

## Part B — pool 缓存 / checkpoint-resume(#3,代码已内建大部分)

### 代码现状(已核实)
仓库**已内建完整 checkpoint/resume**(`rgbd_sym/rl/learner.py`):
- `--checkpoint_dir <path.pt>`:启动时若该文件存在 → `joblib.load` 成 `chkpt_dict`。
- **恢复时跳过 prefill**:`learner.py:423 / 442` 的 `... and self.chkpt_dict is None` 守卫——`chkpt_dict` 非空则不采集 expert / init pool。
- **buffer 可完整序列化**:`policy_storage.get_state_dict()` / `load_from_state_dict()`(`ext/equi-rl-for-pomdps/buffers/seq_vanilla.py:257/273`,seq_rot 继承),checkpoint 里存 `buffer_dict`(含增强 pool)+ agent + RNG + 计数器 + `wandb_id`。
- **保存时机**:训练循环里 `if (time.time()-start)/3600 > time_limit:` → dump 整个 checkpoint 后 `exit(0)`(`learner.py:469-501`)。`--time_limit` 默认 1000(小时),即默认从不触发。

### 立即可用(无需改代码):时间分段跑
GPU 紧缺时把长训练**切成多段**:
```sh
# 首次:设一个小时预算,到点自动存 checkpoint 并退出
python ./rgbd_sym/rl/main.py --cfg ... --checkpoint_dir ckpt/run_seedX.pt --time_limit 0.5 ...
# 续跑(自动跳过 prefill,从 buffer/agent/wandb 续上)
python ./rgbd_sym/rl/main.py --cfg ... --checkpoint_dir ckpt/run_seedX.pt --time_limit 0.5 ...
```
注意:checkpoint 保存点在**首轮 warmup update 之后**(prefill→warmup update→init pool→进训练循环才检查 time_limit),所以第一段仍会付一次 warmup update 成本。

### 需要小改动:pool-only 复用(跨不同训练配置)
现机制恢复的是**整段 run**(连 agent/计数器/wandb_id 一起,属「续跑」)。若想**只复用增强 pool**、换不同下游训练设置(如 `sac` vs `sacfd`、不同 lr)而不重跑昂贵的增强:
- 加一个「pool-only」模式:prefill 后**只 dump `buffer_dict`**;新 run 加载它到**全新 agent/wandb**并跳过 prefill。
- 基于已有的 `get_state_dict`/`load_from_state_dict` + 那两个 `chkpt_dict is None` 守卫,改动很小(约几十行:新增 `--pool_cache <path>`,存/取仅 buffer,不碰 agent/RNG/计数器)。

### 状态
- 时间分段跑:**开箱可用**。
- pool-only 复用:**小改动 TODO**(需要时再做)。

---

## 优先级小结
1. 现在先跑筛选实验(`bash/bench_train.sh`:data-scarce + 3 配对 seed)看有没有信号。
2. 若要压 GPU 成本:先用 **Part B 的时间分段跑**(零改动);多配置扫参再考虑 **pool-only 复用**。
3. **Part A 提速**在需要大量重复 prefill(如多 seed × 多配置)时再做,做时带逐比特校验。
