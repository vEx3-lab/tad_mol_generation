# RL 目录说明

本目录保存强化学习训练脚本、GSPO/GDPO/REINFORCE 实验版本、RL 采样工具和 SA score 辅助文件。

## 推荐入口

| 文件 | 用途 |
| --- | --- |
| `multi_obj_gspo_4_21_5_6.py` | 当前活跃 GSPO 5_6 训练脚本 |
| `multi_obj_gspo_4_21_5_6_20260508.py` | 20260508 GSPO 5_6 baseline |
| `multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | 从 20260508 baseline 复制出的 no-aux 版本 |
| `multi_obj_gspo_4_21_diagnostics.py` | GSPO loss、旧策略 logprob、ratio/KL 诊断等公共诊断工具 |
| `utils.py` | SELFIES rollout、复合奖励和奖励缩放工具 |

## 历史实验脚本

以下文件建议保留，但不要作为默认入口继续扩展：

- `my_gspo.py`
- `multi_obj_gspo_5_6.py`
- `multi_obj_gspo_elite_aux_04_21.py`
- `multi_obj_gspo_4_27_reward_plateau.py`
- `multi_obj_gspo_rankloss.py`
- `multi_obj_gdpo.py`
- `multi_obj_reinforce.py`

## 维护约定

- 新实验脚本优先从明确 baseline 复制，并在 Markdown 中记录差异。
- 不直接覆盖历史实验脚本。
- 日志字段发生变化时，同步更新诊断脚本或记录兼容说明。
- 稳定下来的通用函数应逐步沉淀到 `utils.py` 或单独的公共模块。
