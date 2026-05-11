# RL Python 文件版本分析

日期：2026-05-11

## 1. 总览

`rl/` 目录当前包含 19 个 Python 文件。整体可以分为 5 类：

1. GSPO 主线版本：`my_gspo.py`、`multi_obj_gspo.py`、`multi_obj_gspo_4_21.py`、`multi_obj_gspo_5_6.py`、`multi_obj_gspo_4_21_5_6*.py`
2. GSPO 派生实验：`multi_obj_gspo_beam.py`、`multi_obj_gspo_rankloss.py`、`multi_obj_gspo_4_21_pareto.py`、`multi_obj_gspo_4_27_reward_plateau.py`、`multi_obj_gspo_elite_aux_04_21.py`
3. 其他 RL baseline：`multi_obj_gdpo.py`、`multi_obj_reinforce.py`、`rl_test.py`
4. 工具文件：`utils.py`、`sascorer.py`、`my_sascore.py`
5. 诊断工具：`multi_obj_gspo_4_21_diagnostics.py`

## 2. 文件逐项说明

| 文件 | 类型 | 核心内容 | 相对前序版本的主要变化 |
| --- | --- | --- | --- |
| `rl_test.py` | 算法玩具验证 | 用简单 MLP policy 在 gym 环境里演示 GSPO/ratio/KL | 与分子生成无关，主要用于理解 GSPO 序列级思想 |
| `sascorer.py` | SA score 工具 | Ertl SA score 计算 | 第三方/传统工具代码，供奖励函数调用 |
| `my_sascore.py` | SA CSV 工具 | 给 CSV 批量计算 SA score | 小型数据处理脚本，不参与 RL 主训练 |
| `utils.py` | RL 通用工具 | SELFIES rollout、Vina reward、复合 reward、GDPO reward | 奖励 schema 从旧的 `vina/sa/logp` 扩展到 `vina_raw/vina_reward/sa_raw/sa_reward/logp_raw/logp_reward/status` |
| `my_gspo.py` | 早期 GSPO | 基础 GSPO loss、Vina 单目标 reward、简单日志 | 早期版本，结构简单，只有 policy/KL/ratio，日志字段少 |
| `multi_obj_gdpo.py` | GDPO/GSPO 变体 | 使用 GDPO 风格复合 reward 和 GSPO loss | 与 `my_gspo.py` 相比，奖励改为多目标 GDPO schema，训练入口仍较简单 |
| `multi_obj_gspo.py` | 多目标 GSPO | 复合 reward、ReplayBuffer、scheduler、更多日志 | 相比 `my_gspo.py`，增加多目标 reward、经验回放、学习率/total_loss 日志 |
| `multi_obj_reinforce.py` | REINFORCE baseline | REINFORCE loss、baseline、ReplayBuffer | 不使用 PPO clip/KL；作为非 GSPO baseline |
| `multi_obj_gspo_beam.py` | Beam + GSPO | Beam search 采样结合 GSPO 更新 | 将采样策略从随机 SELFIES 采样替换/扩展为 beam search |
| `multi_obj_gspo_rankloss.py` | Rank loss 实验 | hybrid advantage / rank-aware reward | 试图把排序信号并入 advantage，区别于纯 reward z-score |
| `multi_obj_gspo_4_21.py` | 4/21 GSPO 稳定化 | 显式 sequence logprob、old logprob batch、ratio clip 范围 | 相比 `multi_obj_gspo.py`，更重视 old policy logprob 的一致性和 ratio 稳定 |
| `multi_obj_gspo_elite_aux_04_21.py` | Elite aux 实验 | Elite replay memory、aux likelihood loss、scaffold diversity bonus | 引入高奖励样本辅助似然学习和 scaffold 多样性奖励 |
| `multi_obj_gspo_4_21_diagnostics.py` | 诊断工具/训练变体 | sampling logprob、true-token-KL、ratio diagnostics、EMA/plateau | 从训练脚本中抽出大量诊断函数，是后续 5_6 版本的重要依赖 |
| `multi_obj_gspo_4_21_pareto.py` | Pareto 训练 | ParetoArchive、non-dominated sort、crowding distance | 把 reward 标量优化改成多目标 Pareto 排序/拥挤距离训练 |
| `multi_obj_gspo_4_27_reward_plateau.py` | Reward plateau 版本 | reward/top5 EMA 平台检测和早停 | 重点解决 reward 停滞问题，但没有后续 5_6 的完整诊断体系 |
| `multi_obj_gspo_5_6.py` | GSPO 5_6 早版 | EliteBuffer、RewardMovingBaseline、dedup、advantage clip、KL beta 自适应 | 相比 4/21 系列，形成 5_6 主线：elite buffer、aux loss、plateau early stop、更多日志 |
| `multi_obj_gspo_4_21_5_6_20260508.py` | 20260508 稳定版 | unique rollout refill、true-token-KL beta、temperature/top_k 增强 | 在 `multi_obj_gspo_5_6.py` 上强化防塌缩：补采样直到唯一 batch，KL beta 使用 `true_token_kl_mean` |
| `multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | no-aux 版本 | 保留 GSPO 与 elite buffer 记录，移除 aux loss 训练路径 | 从 20260508 baseline 复制，删除 `elite_aux_likelihood_loss`、`aux_*` 参数和日志字段 |
| `multi_obj_gspo_4_21_5_6.py` | 当前活跃 5_6 版本 | min unique gate、raw_unique_rate、collapse early stop、精简日志 | 相比 20260508 版，不再补采样 128 次，而是采样后低多样性直接跳过/早停；日志字段被精简 |

## 3. GSPO 主线演化

### 3.1 `my_gspo.py`

这是早期 GSPO 分子优化脚本。核心是：

- `compute_gspo_loss_batch(...)`
- `train_gspo(...)`
- Vina reward 函数
- 简单 CSV 日志

主要特点是简单直接，适合作为算法原型，但 reward、日志、路径和稳定性控制都比较粗。

### 3.2 `multi_obj_gspo.py`

相比 `my_gspo.py`，主要改动：

- 从单一 Vina reward 扩展到 Vina/QED/SA/LogP 复合 reward
- 增加 `ReplayBuffer`
- 增加 scheduler 支持
- 日志增加 `total_loss`、`lr`、`kl_beta`、`replay_size`
- 训练主流程更接近真正多目标分子优化

diff 概况：相对 `my_gspo.py`，约 411 行新增、70 行删除。

### 3.3 `multi_obj_gspo_4_21.py`

相比 `multi_obj_gspo.py`，主要变化：

- 引入 `compute_sequence_logprob_from_logits(...)`
- 引入 `compute_old_logprob_batch(...)`
- 对 sequence ratio 增加 `ratio_min/ratio_max` 过滤
- 主程序默认 batch 增大到 16，`mu=3`，`clip_eps=0.25`
- 更强调 old policy 与当前 policy 的 logprob 对齐

这个版本是后续诊断和 5_6 版本的重要基础。

### 3.4 `multi_obj_gspo_5_6.py`

这是 5_6 主线早版。相比 4/21 版本，新增：

- `EliteBuffer`
- `elite_aux_likelihood_loss(...)`
- `RewardMovingBaseline`
- `deduplicate_rollout_batch(...)`
- `prepare_advantages(...)`
- `adjust_kl_beta(...)`
- reward/top5 EMA
- plateau early stopping
- KL stop patience
- 更详细 CSV 日志

主要目的：提升训练稳定性，缓解 reward 方差和高奖励样本遗忘。

### 3.5 `multi_obj_gspo_4_21_5_6_20260508.py`

这是 20260508 稳定变体，也是当前最重要 baseline 之一。相比 `multi_obj_gspo_5_6.py`：

- 新增顶部说明，明确三项稳定化改动
- 新增 `sample_unique_rollout_batch(...)`
- 默认 `temperature=1.15`、`top_k=80`
- `target_kl=0.05`、`kl_beta_min=1e-3`、`kl_beta_max=2.0`
- KL beta 调节改用 `true_token_kl_mean`
- 主程序 `mu=2`

目的：解决 5 月 8 日日志中观察到的采样塌缩和 KL 代理失真。

### 3.6 `multi_obj_gspo_4_21_5_6_20260508_no_aux.py`

这是 no-aux 派生版。相比 20260508 baseline：

- 删除 `elite_aux_likelihood_loss(...)`
- 删除 `aux_loss`
- 删除 `aux_coef`
- 删除 `aux_batch_size`
- 删除 `aux_start`
- 删除 `aux_reward_temperature`
- `total_loss = gspo_loss`
- 日志和 checkpoint 命名使用 `no_aux`

保留内容：

- GSPO objective
- true-token-KL beta
- unique rollout refill
- elite buffer 记录和导出

### 3.7 `multi_obj_gspo_4_21_5_6.py`

这是当前活跃版本。相比 20260508 baseline：

- 增加 `min_unique_samples`
- 增加 `min_raw_unique_rate`
- 增加 `collapse_stop_patience`
- 低多样性 batch 不再继续更新，而是跳过并累计塌缩计数
- 日志新增/保留 `raw_unique_rate`、`collapse_skip_hits`
- CSV 字段被精简，不再记录大量可推导或常量字段
- 当前主程序参数为 `M=500`、`mu=4`、`temperature=1.1`、`top_k=30`、`aux_coef=0.0`

与 20260508 的核心差异：20260508 通过补采样尽量凑唯一 batch；当前版本通过门槛直接阻止低多样性 batch 参与训练。

## 4. 派生实验线

### 4.1 Beam 版本：`multi_obj_gspo_beam.py`

核心变化：

- 使用 beam search 生成候选分子
- 仍使用 GSPO 训练结构
- 保留 ReplayBuffer 风格

适合比较随机采样与 beam 采样对 reward 和 diversity 的影响。

### 4.2 Rank loss 版本：`multi_obj_gspo_rankloss.py`

核心变化：

- 新增 `compute_hybrid_advantage(...)`
- 将 reward 与 rank 信息结合
- 用排序信号增强 advantage

适合研究 batch 内排序是否比 reward z-score 更稳定。

### 4.3 Elite aux 版本：`multi_obj_gspo_elite_aux_04_21.py`

核心变化：

- `EliteReplayMemory`
- `compute_aux_likelihood_loss(...)`
- `apply_scaffold_diversity_bonus(...)`
- scaffold diversity bonus
- target KL 调节

这是 5_6 `EliteBuffer + aux loss` 的早期探索版本。

### 4.4 Pareto 版本：`multi_obj_gspo_4_21_pareto.py`

核心变化：

- 不再只看标量 reward
- 使用 `ParetoArchive`
- 使用 non-dominated sorting
- 使用 crowding distance
- 将 Vina/QED/SA/LogP 视为多目标优化

适合分析多目标 trade-off，而不是单一加权和。

### 4.5 Reward plateau 版本：`multi_obj_gspo_4_27_reward_plateau.py`

核心变化：

- `compute_plateau_metrics(...)`
- reward/top5 窗口改善率
- plateau early stop

这是早停机制的中间实验版，后续思想进入 5_6 系列。

## 5. 非 GSPO baseline

### 5.1 `multi_obj_gdpo.py`

虽然函数仍叫 `compute_gspo_loss_batch` / `train_gspo`，但 reward 使用 GDPO 风格 schema 和 z-score 归一化。更适合作为 GDPO reward 线的实验，而不是标准 GSPO 主线。

### 5.2 `multi_obj_reinforce.py`

使用 REINFORCE：

- 无 PPO clip
- 无 reference KL
- 使用 baseline 降低方差
- 使用 ReplayBuffer 混合历史样本

适合作为简单 RL baseline。

### 5.3 `rl_test.py`

非分子任务，用 gym 和简单 MLP 演示 GSPO ratio/KL。它是算法教学/验证脚本，不应作为分子训练入口。

## 6. 工具文件

### 6.1 `utils.py`

当前职责较多：

- SELFIES batch 采样
- Vina reward 转换
- SA reward 转换
- LogP reward 转换
- 复合 reward
- GDPO reward

建议后续拆分为：

- `rl/sampling_utils.py`
- `rl/reward_utils.py`
- `rl/reward_schema.py`

但短期不建议直接移动，避免破坏旧脚本 import。

### 6.2 `sascorer.py`

SA score 基础实现。建议保持不动。

### 6.3 `my_sascore.py`

CSV 批量 SA 计算脚本。建议归为数据处理/评估工具，不属于 RL 主训练。

## 7. 版本差异统计

| 比较 | diff 概况 |
| --- | --- |
| `my_gspo.py` -> `multi_obj_gspo.py` | 411 insertions, 70 deletions |
| `multi_obj_gspo.py` -> `multi_obj_gspo_4_21.py` | 268 insertions, 360 deletions |
| `multi_obj_gspo_4_21.py` -> `multi_obj_gspo_elite_aux_04_21.py` | 599 insertions, 393 deletions |
| `multi_obj_gspo_4_21.py` -> `multi_obj_gspo_4_21_diagnostics.py` | 722 insertions, 278 deletions |
| `multi_obj_gspo_4_21.py` -> `multi_obj_gspo_4_21_pareto.py` | 417 insertions, 373 deletions |
| `multi_obj_gspo_4_21.py` -> `multi_obj_gspo_4_27_reward_plateau.py` | 214 insertions, 320 deletions |
| `multi_obj_gspo_5_6.py` -> `multi_obj_gspo_4_21_5_6_20260508.py` | 169 insertions, 44 deletions |
| `multi_obj_gspo_4_21_5_6_20260508.py` -> `multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | 19 insertions, 82 deletions |
| `multi_obj_gspo_4_21_5_6_20260508.py` -> `multi_obj_gspo_4_21_5_6.py` | 100 insertions, 183 deletions |
| `multi_obj_gspo.py` -> `multi_obj_gspo_beam.py` | 45 insertions, 351 deletions |
| `multi_obj_gspo.py` -> `multi_obj_gspo_rankloss.py` | 101 insertions, 427 deletions |
| `multi_obj_gspo.py` -> `multi_obj_gdpo.py` | 89 insertions, 407 deletions |
| `multi_obj_gspo.py` -> `multi_obj_reinforce.py` | 88 insertions, 435 deletions |

## 8. 推荐使用关系

| 目的 | 推荐文件 |
| --- | --- |
| 当前继续调 GSPO 5_6 | `multi_obj_gspo_4_21_5_6.py` |
| 复现 20260508 稳定版 | `multi_obj_gspo_4_21_5_6_20260508.py` |
| 不使用 aux loss 的 20260508 版 | `multi_obj_gspo_4_21_5_6_20260508_no_aux.py` |
| 查看 ratio/KL/logprob 诊断实现 | `multi_obj_gspo_4_21_diagnostics.py` |
| 做 Pareto 多目标实验 | `multi_obj_gspo_4_21_pareto.py` |
| 做 beam search 对照 | `multi_obj_gspo_beam.py` |
| 做 REINFORCE baseline | `multi_obj_reinforce.py` |
| 查看奖励/采样公共工具 | `utils.py` |

## 9. 维护建议

1. 将 `multi_obj_gspo_4_21_5_6.py` 标记为当前活跃版本。
2. 将 `multi_obj_gspo_4_21_5_6_20260508.py` 标记为重要 baseline。
3. 每次新实验都复制 baseline 并生成 Markdown 变更记录。
4. 不建议继续在根目录新增 `grpo_*.py` 或 `gspo_*.py`。
5. 后续应把稳定下来的函数从训练脚本中抽出，例如：
   - `EliteBuffer`
   - `RewardMovingBaseline`
   - `deduplicate_rollout_batch`
   - `prepare_advantages`
   - `adjust_kl_beta`
6. `utils.py` 中 reward 和 sampling 职责混杂，建议后续拆分，但保持旧 import 兼容。
