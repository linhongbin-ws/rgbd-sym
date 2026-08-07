# MEA v2 TODO(接 STATUS.md / b2_plan.md)

- **更新**:2026-08-07 · 分支 `meav2`
- 优先级:🔴 现在做 · 🟡 探测出结果后 · 🟢 下游/收尾 · ⚠️ 杂项/安全

---

## 🔴 P0 —— 正在做:Square_D2 floor 探测(决定 B2 有没有靶子)

- [ ] **① 拿 D2 abs 数据**(远端 equidiff env):
  - `python <mimicgen>/scripts/download_datasets.py --dataset_type core --tasks square_d2`(脚本在 `~/ssd/code/research/mea/mimicgen/mimicgen/scripts/`)
  - 带图像 → 直接 `robomimic_dataset_conversion.py -i square_d2.hdf5 -o square_d2_abs.hdf5 -n 12`;
  - 只有 low-dim → 先 `dataset_states_to_obs.py --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84` 再转 abs。
- [ ] **② 跑 C8(EquiDiff)on D2**:`train_equi_diffusion_unet_abs task_name=square_d2 dataset_path=.../square_d2_abs.hdf5 n_demo=100 batch64 满 500ep`。
- [ ] **③ 跑 C1(plain DP)on D2**:`train_diffusion_unet` 同参数、同数据。
- [ ] **④ 判 floor**:`mea_readout.py` 读末10 → 算 **(EquiDiff−DP) on D2**,和 D0 的 **+13pt** 比。
  - **收窄/反转** → floor 现身,B2 有靶子 → 进 P1。
  - **维持/变大** → 全观测 sim 压不出 floor → 见 P2(诚实分叉)。

---

## 🟡 P1 —— 若 D2 现出 floor:定位 escape 群阶 + 建 B2b

- [ ] **群阶梯**:确认 `+policy.N=4` 能改等变阶(先 `training.debug` 或短跑看 param count 变);在 D2 上补 **C4 / C2**。非单调峰值处 = 该退化到的群阶。
- [ ] **建 B2b(相位门控群退化,非喂标量)**:`escnn.RestrictionModule` 把 C_N→C1(或峰值阶),或并联 `trivialOnR2()` 自由通路,λ=相位后验单调 schedule;load-balance/entropy 正则防坍缩。见 [[b2_plan]] §4。
- [ ] **消融阶梯** B0 / **B2a**(不变标量→等变网,Curie no-op 反面对照,应≈B0)/ **B2b**(群退化,应>B0)/ B3(+匹配增强)。叙事:B2a≈B0≪B2b。
- [ ] **补 seed**:关键臂 3 seed,AUC/末10 判据,给误差棒。

## 🟡 P1' —— 若 D2 无 floor:诚实分叉(二选一,交用户定)

- [ ] **走任务 B**:构造 place-to-固定世界目标(goal 世界系固定 + 部分不可观测)→ 教科书式 floor;工程量中。
- [ ] **或 证据移到 partial-obs/真机**:sim 全观测下 GIC 使全局等变几乎总 intrinsic;context/gauge 的价值在无 GT-mask 的真机最大(iclr_strategy §4.4)。paper 重心转真机小实验。

---

## 🟢 P2 —— 下游/写作(有正信号后)

- [ ] **理论对接**:把 new_idea 的 per-context value-invariance/policy-equivariance 定理,与 partial-equivariance 误差界 `(ε_R+γρε_P)/(1−γ)`(2411.04225)接上,形式化"逃 floor"的增益。
- [ ] **related work 定位**:显著区分 **PE-SAC(2512.00915)**(空间门控/全观测)vs 本文(相位索引群退化 + belief + POMDP + 数据侧);正面回应 **GIC(2308.14984)**(定位成 stabilizer reduction 不是"contact breaks symmetry")。
- [ ] **negative-analysis 章节**:把 2×2 + 两个构造性死因 + eval 单模态论证,写成"何时物体对称增强对 diffusion BC 有效/无效"的诚实分析(即便主线走 B2)。
- [ ] **floor-detector 方法**:群阶梯(C8/C4/C2/C1 是否单调)本身是可复用的"floor 探测器",可作方法贡献。

---

## ⚠️ 杂项 / 安全

- [ ] **🔒 撤销泄露的 wandb key**:会话里贴过一个真实 `wandb_v1_...` key(已暴露在对话/日志)→ 去 wandb settings **reset/revoke**。别再用它。
- [ ] **清理残留目录**:`data/robomimic/datasets/square_mea_baseline/`(早先错路径 `mkdir` 出来的空/半成品,和真正的 `square_mea_base` 撞名,易混)——确认无用后删。
- [ ] **datagen hook 状态**:`mea_diff/mimicgen_c4_hook.py::PATCH` 已落地在真 repo 的 `data_generator.py`;若不再用 keyed 生成,记得回退/门控(`MEA_C4` 环境变量默认关,问题不大)。
