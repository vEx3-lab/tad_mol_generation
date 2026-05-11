# 项目代码整理方案

日期：2026-05-11

---

## 一、项目总览

本项目是一个基于 SELFIES + Decoder-Only Transformer 的分子生成框架，使用 GSPO/GRPO/REINFORCE 等强化学习算法进行多目标分子优化（Vina 对接评分 + QED + SA + LogP）。

**当前问题概况**：代码约 104 个 Python 文件，存在大量实验版本副本、废弃暂存文件、重复函数、命名混乱等问题。根目录散落 20+ 个实验脚本，`rl/` 目录有 15+ 个 GSPO 变体，`train/` 有 5 个几乎相同的训练脚本。

---

## 二、文件分类

### 2.1 核心文件（当前活跃使用）

| 文件 | 功能 | 状态 |
|------|------|------|
| `main_selfies.py` | SELFIES 预训练入口 | 核心 |
| `fine_tuning_selfies.py` | SELFIES 微调入口 | 核心 |
| `generate_selfies.py` | SELFIES grammar 约束生成 | 核心 |
| `evaluate.py` | 分子评估 CLI 工具 | 核心 |
| `project_paths.py` | 路径解析工具 | 核心 |
| `config/load_config.py` | YAML 配置加载 | 核心 |
| `config/decoder_only_tfm_config.yaml` | 主模型配置 | 核心 |
| `data/data_utils.py` | SELFIES 数据/词表处理 | 核心 |
| `data/data_aug.py` | SMILES 数据增强 | 核心 |
| `model/decoder_only_tfm.py` | 主模型：Decoder-Only Transformer | 核心 |
| `model/bi_lstm.py` | 备选模型：BiLSTM | 核心 |
| `sample/sample.py` | SELFIES top-k 采样 | 核心 |
| `sample/sample_beam.py` | SELFIES beam search 采样 | 核心 |
| `feedback/utils.py` | 分子规范化 + SMILES→PDBQT | 核心 |
| `feedback/vina_scores.py` | Vina 批量对接 + 缓存 | 核心 |
| `rl/utils.py` | RL 采样 + 复合奖励函数 | 核心 |
| `rl/multi_obj_gspo_4_21_diagnostics.py` | GSPO 诊断工具库 | 核心 |
| `rl/multi_obj_gspo_4_21_5_6.py` | **当前活跃 GSPO 5_6 版本** | 核心 |
| `rl/multi_obj_gspo_4_21_5_6_20260508.py` | GSPO 5_6 重要 baseline | 核心 |
| `eval metrics/cal_metrics.py` | 批量评估（QED/SA/Novelty） | 核心 |
| `eval metrics/SA_Score/sascorer.py` | SA Score 计算（原始位置） | 核心 |
| `scripts/plot_gspo_5_6_diagnostics.py` | GSPO 5_6 日志诊断绘图 | 核心 |
| `scripts/plot_all_training_metrics.py` | 通用训练曲线绘图 | 核心 |
| `scripts/smoke_test.py` | 基础导入/模型 smoke test | 核心 |
| `scripts/project_inventory.py` | 项目文件盘点工具 | 核心 |

### 2.2 历史实验文件（保留但不作为默认入口）

| 文件 | 说明 |
|------|------|
| `main.py` | SMILES 流程入口（已被 SELFIES 流程取代） |
| `train.py` | SMILES K-Fold 训练 |
| `train_selfies.py` | SELFIES K-Fold 训练 |
| `fine_tuning.py` | SMILES 微调 |
| `generate.py` | SMILES 生成（无 grammar 约束） |
| `dataloader.py` | SMILES 数据加载（已被 data_utils.py 取代） |
| `smile_gen.py` | SMILES 批量生成脚本 |
| `selfies_gen.py` | SELFIES 批量生成脚本 |
| `g_v2.py` | GRPO + Vina RL 训练（早期版本） |
| `grpo_.py` | GRPO 基线版本 |
| `grpo_1_5.py` | GRPO + batch-padded + ref model |
| `grpo_12_16_.py` | GRPO + 修复 ratio + 大批量 |
| `grpo_alth1.py` | GRPO pseudocode 精确版（3 层嵌套循环） |
| `grpo_lr.py` | GRPO + k3 KL estimator |
| `grpo_new.py` | GRPO + ref model k3 KL（ref 未更新，有 bug） |
| `rl_grpo_ft.py` | 根目录 GRPO 微调脚本 |
| `rl/my_gspo.py` | 早期单目标 GSPO |
| `rl/multi_obj_gspo.py` | 多目标 GSPO 初版 |
| `rl/multi_obj_gspo_4_21.py` | GSPO 4/21 稳定版 |
| `rl/multi_obj_gspo_5_6.py` | GSPO 5_6 早期版 |
| `rl/multi_obj_gspo_beam.py` | Beam search + GSPO |
| `rl/multi_obj_gspo_rankloss.py` | Rank-aware advantage GSPO |
| `rl/multi_obj_gdpo.py` | GDPO 风格 reward 归一化 |
| `rl/multi_obj_reinforce.py` | REINFORCE baseline |
| `rl/multi_obj_gspo_elite_aux_04_21.py` | Elite aux + scaffold diversity |
| `rl/multi_obj_gspo_4_21_pareto.py` | Pareto 多目标 GSPO |
| `rl/multi_obj_gspo_4_27_reward_plateau.py` | Reward plateau 早停版 |
| `model/decoder_only_tfm_v2.py` | 从头实现 Transformer |
| `model/decoder_only_tfm_v3.py` | v2 + RoPE + SwiGLU |
| `model/v4.py` | 稳定版 + RoPE graft |
| `model/v5.py` | 稳定版 + Pre-LN |
| `model/v6.py` | 清洁重写: RoPE + GELU + Pre-LN |
| `model/v7.py` | 清洁重写: RoPE + SwiGLU + Pre-LN |
| `train/train_v2.py` ~ `train_v5.py` | 不同模型版本的 K-Fold 训练 |

### 2.3 可删除/归档文件

| 文件 | 原因 |
|------|------|
| `debug.py` | 仅 `print(1+2)` |
| `temp.py` | 临时暂存文件 |
| `12_14_1712.py` | 临时暂存文件（与 temp.py 重复） |
| `fix_encoding_temp.py` | 一次性补丁脚本（已执行完毕） |
| `trainer_selfies.py` | 不完整的 Trainer 类（无训练方法） |
| `model/tfm.py` | 空占位文件（仅注释） |
| `model/rnn.py` | 空占位文件（仅注释） |
| `feedback/demo.py` | 早期原型（已被 grpo.py 取代） |

### 2.4 重复文件

| 文件 | 重复于 | 建议 |
|------|--------|------|
| `eval metrics/sascorer.py` | `eval metrics/SA_Score/sascorer.py` | 删除根级副本，改为 re-import |
| `train/config.py` | `config/load_config.py` | 删除，改为 import |
| `rl/my_sascore.py` | `eval metrics/SA_Score/my_sascore.py` | 功能近似，保留 SA_Score 版本 |

---

##三、需要用户确认的结构性操作清单

### 3.1 删除文件（8 个）

| # | 文件路径 | 删除原因 |
|---|----------|----------|
| 1 | `debug.py` | 仅 `print(1+2)`，无意义 |
| 2 | `temp.py` | 临时暂存文件，与 `12_14_1712.py` 重复 |
| 3 | `12_14_1712.py` | 临时暂存文件，含日期标记 |
| 4 | `fix_encoding_temp.py` | 一次性补丁脚本，已执行完毕 |
| 5 | `model/tfm.py` | 空占位文件，仅 5 行注释 |
| 6 | `model/rnn.py` | 空占位文件，仅 5 行注释 |
| 7 | `trainer_selfies.py` | 不完整，仅 `__init__` 无训练方法 |
| 8 | `feedback/demo.py` | 早期原型，已被 `grpo.py` 取代 |

### 3.2 删除重复文件（3 个）

| # | 文件路径 | 重复原因 |
|---|----------|----------|
| 9 | `eval metrics/sascorer.py` | 与 `eval metrics/SA_Score/sascorer.py` 100% 相同 |
| 10 | `train/config.py` | 是 `config/load_config.py` 的旧版副本 |
| 11 | `rl/my_sascore.py` | 功能与 `eval metrics/SA_Score/my_sascore.py` 近似 |

### 3.3 重命名文件（8 个）

| # | 当前路径 | 建议新路径 | 理由 |
|---|----------|-----------|------|
| 12 | `g_v2.py` | `rl/train_grpo_vina.py` | 移到 rl/，文件名体现功能 |
| 13 | `rl_grpo_ft.py` | `rl/train_grpo_ft.py` | 移到 rl/ 目录 |
| 14 | `rl/my_gspo.py` | `rl/gspo_baseline.py` | 去掉无意义前缀 `my_` |
| 15 | `grpo_.py` | `rl/archive/grpo_v1_baseline.py` | 历史版本归档 |
| 16 | `grpo_1_5.py` | `rl/archive/grpo_v2_batch_ref.py` | 历史版本归档 |
| 17 | `grpo_12_16_.py` | `rl/archive/grpo_v3_large_batch.py` | 历史版本归档 |
| 18 | `grpo_alth1.py` | `rl/archive/grpo_v4_nested_loop.py` | 历史版本归档 |
| 19 | `grpo_changduguiyi.py` | `rl/archive/grpo_v5_length_norm.py` | 历史版本归档（有 bug） |
| 20 | `grpo_lr.py` | `rl/archive/grpo_v6_k3_kl.py` | 历史版本归档 |
| 21 | `grpo_new.py` | `rl/archive/grpo_v7_ref_k3.py` | 历史版本归档 |

### 3.4 重命名 RL 版本文件（7 个）

| # | 当前路径 | 建议新路径 | 理由 |
|---|----------|-----------|------|
| 22 | `rl/multi_obj_gdpo.py` | `rl/gdpo_reward_norm.py` | 简化命名 |
| 23 | `rl/multi_obj_reinforce.py` | `rl/reinforce_baseline.py` | 简化命名 |
| 24 | `rl/multi_obj_gspo_beam.py` | `rl/gspo_beam_search.py` | 简化命名 |
| 25 | `rl/multi_obj_gspo_rankloss.py` | `rl/gspo_rank_advantage.py` | 简化命名 |
| 26 | `rl/multi_obj_gspo_elite_aux_04_21.py` | `rl/archive/gspo_elite_aux_scaffold.py` | 历史版本归档 |
| 27 | `rl/multi_obj_gspo_4_27_reward_plateau.py` | `rl/archive/gspo_plateau_early_stop.py` | 历史版本归档 |
| 28 | `rl/multi_obj_gspo_4_21_pareto.py` | `rl/gspo_pareto.py` | 简化命名 |

### 3.5 移动文件位置（5 个）

| # | 当前路径 | 建议新路径 | 理由 |
|---|----------|-----------|------|
| 29 | `plot/cns-mpo.py` | `eval metrics/cns_mpo.py` | 这是评估工具，不是绘图脚本 |
| 30 | `plot_docking_score_density.py` | `scripts/plot_docking_density.py` | 绘图脚本应集中到 scripts/ |
| 31 | `feedback/logs/plot.py` | `scripts/plot_grpo_training.py` | 绘图脚本应集中到 scripts/ |
| 32 | `feedback/logs/result_plot.py` | `scripts/plot_grpo_training_by_epoch.py` | 绘图脚本应集中到 scripts/ |
| 33 | `rl/feedback/logs/plot.py` | `scripts/plot_gspo_training.py` | 绘图脚本应集中到 scripts/ |
| 34 | `rl/feedback/logs/plot2.py` | `scripts/plot_gspo_training_v2.py` | 绘图脚本应集中到 scripts/ |

### 3.6 合并文件（2 组）

| # | 涉及文件 | 合并方式 | 理由 |
|---|----------|----------|------|
| 35 | `train/train.py` + `train/train_v2.py` ~ `train_v5.py`（5 个文件） | 合并为 `train/train_cv.py`，模型通过参数传入 | 5 个文件 `train_cv()` 函数完全相同，仅 model import 不同 |
| 36 | `plot/plot_property.py` + `plot/two_model_plot_property.py` | 提取公共函数到 `scripts/plot_property_utils.py`，保留两个入口脚本 | `compute_properties()` 函数完全重复 |

### 3.7 目录结构调整（2 项）

| # | 操作 | 理由 |
|---|------|------|
| 37 | 创建 `rl/archive/` 目录，移入历史 GSPO/GRPO 版本 | 将活跃版本与历史版本分离 |
| 38 | `eval metrics/` 重命名为 `eval_metrics/`（去掉空格） | 避免路径中的空格问题 |

---

## 四、命名问题汇总

### 4.1 命名不清晰的文件

| 当前名称 | 问题 | 建议 |
|----------|------|------|
| `g_v2.py` | 不体现功能 | `train_grpo_vina.py` |
| `grpo_.py` | 尾部下划线无意义 | 归档为 `grpo_v1_baseline.py` |
| `grpo_changduguiyi.py` | 中文拼音，应为英文 | 归档为 `grpo_v5_length_norm.py` |
| `grpo_alth1.py` | "alth" 含义不明 | 归档为 `grpo_v4_nested_loop.py` |
| `my_gspo.py` | "my_" 前缀无意义 | `gspo_baseline.py` |
| `12_14_1712.py` | 纯时间戳，不体现功能 | 删除 |
| `multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | 文件名过长（46 字符） | 保留为重要 baseline，但新文件应简化 |

### 4.2 模型文件命名

| 当前名称 | 问题 | 建议 |
|----------|------|------|
| `model/decoder_only_tfm.py` | "tfm" 缩写不够清晰 | 可保留（已广泛引用），或改为 `model/transformer.py` |
| `model/decoder_only_tfm_v2.py` ~ `v3.py` | 版本号递增 | 保留但标记为实验 |
| `model/v4.py` ~ `v7.py` | 单纯数字版本号，不体现改动 | 保留但标记为实验 |

---

## 五、代码重复问题

### 5.1 完全重复的函数

| 函数 | 重复位置 |
|------|----------|
| `train_cv()` | `train/train.py`, `train/train_v2.py` ~ `train_v5.py`（5 份完全相同的副本） |
| `compute_properties()` | `plot/plot_property.py`, `plot/two_model_plot_property.py` |
| `evaluate_generated_smiles_counts()` | `plot/plot_v_v&n_v&u&n.py`, `plot/two_model_plot_unv.py` |
| `sample_selfies_batch_from_generate_selfies()` | `g_v2.py`, `temp.py`, `12_14_1712.py`（3 份） |
| `sascorer.py` 全部内容 | `eval metrics/sascorer.py` = `eval metrics/SA_Score/sascorer.py` |
| `load_config()` | `config/load_config.py` ≈ `train/config.py` |

### 5.2 重复的 reward/sampling 逻辑

RL 脚本中以下函数在多个文件中被重复定义：
- `compute_gspo_loss_batch` — 出现在 10+ 个文件中，各版本有微小差异
- `ReplayBuffer` / `EliteBuffer` — 多个变体
- `adjust_kl_beta` — 多个变体
- `prepare_advantages` — 多个变体

**建议**：后续将稳定版本沉淀到 `rl/utils.py` 或新建 `rl/gspo_utils.py`。

---

## 六、其他问题

### 6.1 已知 Bug

| 文件 | 问题 |
|------|------|
| `grpo_changduguiyi.py` | `compute_grpo_loss` 函数缺失（仅在注释中），运行时会 NameError |
| `grpo_new.py` | ref_agent 创建后从未同步更新，形同虚设 |
| `train/train_v5.py` | save_dir 后缀写死为 `'v4'` 而非 `'v5'`，与 train_v4.py 冲突 |
| `smile_gen.py` | 模型 load_state_dict 被注释掉，从随机权重生成 |
| `model/bi_lstm.py` | 类名是 `bi_lstm` 但 `bidirectional=False`，实际是单向 LSTM |

### 6.2 硬编码路径

多个文件存在硬编码路径：
- `selfies_gen.py`: checkpoint 路径 `'./rl/feedback/best_models/best_reward_20260323_173854.pt'`
- `feedback/vina_scores.py`: receptor center/box 硬编码
- `feedback/logs/plot.py` 和 `result_plot.py`: CSV 文件名硬编码

### 6.3 混乱的 Import

- `feedback/grpo.py` 有 5 组重复的 import 块
- `generate.py` 导入了 `bi_lstm` 但从未使用
- 多个文件混合使用 `../` 相对路径和绝对路径

---

## 七、推荐整理路线

### 阶段 1：清理（低风险，立即执行）
1. 删除 2.3 节列出的 8 个可删除文件
2. 删除 3 个重复文件（保留一个，删除其他）
3. 创建 `rl/archive/` 目录

### 阶段 2：重命名与移动（中风险，需确认后执行）
4. 执行 3.3~3.5 节的文件重命名和移动
5. 更新所有受影响文件的 import 路径
6. 将 `eval metrics/` 重命名为 `eval_metrics/`

### 阶段 3：合并与抽取（中高风险，需测试后执行）
7. 合并 `train/train_v*.py` 为一个参数化脚本
8. 提取重复的 plot 工具函数到公共模块
9. 将稳定的 RL 工具函数沉淀到 `rl/utils.py`

### 阶段 4：深度重构（高风险，长期计划）
10. 统一配置管理（将 RL 超参从脚本抽到 YAML）
11. 统一日志字段 schema
12. 修复已知 bug

---

## 八、整理后推荐的项目结构

```
mycode_raw/
├── config/
│   ├── load_config.py
│   ├── decoder_only_tfm_config.yaml
│   ├── decoder_only_tfm_config_chembridge_mpo.yaml
│   └── bi_lstm_config.yaml
├── data/
│   ├── data_utils.py              # SELFIES 数据处理
│   ├── data_aug.py                # SMILES 数据增强
│   ├── data_process.py
│   ├── sel2smi.py
│   ├── run.py
│   ├── sample_data.py
│   └── *.csv                      # 数据文件
├── model/
│   ├── transformer.py             # 主模型 (原 decoder_only_tfm.py)
│   ├── bi_lstm.py                 # 备选模型
│   ├── experimental/              # 实验模型变体
│   │   ├── v3_rope_swiglu.py
│   │   ├── v6_rope_gelu.py
│   │   └── v7_rope_swiglu.py
│   └── checkpoints/               # 模型权重集中存放
├── train/
│   ├── train_cv.py                # 参数化 K-Fold 训练
│   ├── fine_tuning.py             # 微调训练
│   └── checkpoints/               # 训练产物
├── sample/
│   ├── sample.py                  # Top-k 采样
│   └── sample_beam.py             # Beam search 采样
├── rl/
│   ├── utils.py                   # RL 工具函数
│   ├── gspo_utils.py              # GSPO 专用工具 (EliteBuffer/MovingBaseline等)
│   ├── train_gspo_5_6.py          # 当前活跃 GSPO 版本
│   ├── train_gspo_5_6_baseline.py # 20260508 baseline
│   ├── train_gspo_pareto.py       # Pareto 多目标
│   ├── train_gspo_beam.py         # Beam search GSPO
│   ├── train_gspo_rank.py         # Rank advantage GSPO
│   ├── train_reinforce.py         # REINFORCE baseline
│   ├── train_gdpo.py              # GDPO 风格
│   ├── train_grpo_vina.py         # GRPO + Vina
│   ├── sascorer.py                # SA Score 计算
│   └── archive/                   # 历史版本
│       ├── gspo_baseline.py
│       ├── gspo_elite_aux_scaffold.py
│       ├── gspo_plateau_early_stop.py
│       ├── grpo_v1_baseline.py
│       ├── grpo_v2_batch_ref.py
│       ├── grpo_v3_large_batch.py
│       ├── grpo_v4_nested_loop.py
│       ├── grpo_v5_length_norm.py
│       ├── grpo_v6_k3_kl.py
│       └── grpo_v7_ref_k3.py
├── feedback/
│   ├── utils.py                   # 分子工具 (canonicalize, smi2pdbqt)
│   ├── vina_scores.py             # Vina 对接打分
│   └── __init__.py
├── eval_metrics/
│   ├── cal_metrics.py             # 批量评估
│   ├── cns_mpo.py                 # CNS-MPO 计算
│   ├── sascorer.py                # SA Score（原 SA_Score/）
│   └── SA_Score/                  # SA Score 相关
├── scripts/
│   ├── plot_training_metrics.py
│   ├── plot_gspo_diagnostics.py
│   ├── plot_docking_density.py
│   ├── plot_property.py
│   ├── plot_property_utils.py     # 抽取的公共函数
│   ├── project_inventory.py
│   └── smoke_test.py
├── docs/
├── main_selfies.py
├── fine_tuning_selfies.py
├── generate_selfies.py
├── evaluate.py
├── project_paths.py
└── README.md
```

---

## 九、验证清单

整理完成后需验证：
- [ ] `python scripts/smoke_test.py` 通过
- [ ] `python scripts/project_inventory.py` 正常运行
- [ ] 核心训练入口 `main_selfies.py` 可正常导入
- [ ] RL 训练入口 `rl/train_gspo_5_6.py`（原 multi_obj_gspo_4_21_5_6.py）可正常导入
- [ ] 所有被重命名文件的 import 路径已更新
- [ ] Git 历史可通过 `git log --follow` 追溯
