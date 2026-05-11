# 项目工程化整理说明

日期：2026-05-11

## 1. 本次整理目标

本次整理采用低风险、非破坏性策略：不移动、不删除、不重命名现有核心训练脚本；优先补充项目索引、模块职责说明、路径工具注释和可复现的盘点脚本。这样可以提升可读性和后续维护效率，同时保留当前所有实验入口和历史结果。

## 2. 当前项目扫描结果

使用 `rg --files` 统计时，项目可检索文件约 369 个，其中 Python 文件约 164 个。
新增的 `scripts/project_inventory.py` 会跳过 `.git`、`.idea`、`__pycache__` 后扫描更完整的工作区，
当前可见文件约 777 个，其中 Python 文件约 104 个。两个数字差异主要来自 `.gitignore`、
未跟踪产物、模型权重和日志文件的统计口径不同。

主要文件类型包括：

| 类型 | 观察 |
| --- | --- |
| `.py` | 训练、采样、评估、RL、绘图和实验脚本混在根目录、`rl/`、`train/`、`scripts/` 中 |
| `.csv` | 数据集、训练日志、评估结果数量较多，主要在 `data/`、`plot/`、`rl/feedback/logs/` |
| `.pt` / `.ckpt` | 模型权重较多，主要在 `model/`、`train/model/`、`rl/feedback/best_models/` |
| `.png` | 训练曲线、诊断图、论文图和历史结果混在根目录、`figures/`、`scripts/` |
| `.yaml` | 主项目配置和 ACEGEN 子项目配置并存 |

当前 Git 工作区已有大量修改、重命名和未跟踪文件。因此本次整理没有执行目录迁移或删除操作，避免覆盖或打断已有实验状态。

## 3. 核心模块职责

| 模块/目录 | 当前职责 | 维护建议 |
| --- | --- | --- |
| `config/` | YAML 配置和配置加载 | 保留为统一配置入口；后续新增 RL 配置建议放入 `config/rl/` |
| `data/` | 原始分子数据、SELFIES 数据、数据增强结果 | 保留数据文件；建议新增 `data/README.md` 说明列名和来源 |
| `model/` | Decoder-only Transformer、BiLSTM 和历史模型变体 | `decoder_only_tfm.py` 是主模型；`v*.py` 建议标记为历史实验 |
| `sample/` | SELFIES 采样、beam search 采样 | 保留为生成接口；建议把通用采样函数集中到 `sample/` |
| `feedback/` | Vina 打分、分子规范化和反馈工具 | 保留为奖励/打分基础层；不要在这里混入训练循环 |
| `rl/` | GSPO、GDPO、REINFORCE 等 RL 训练脚本 | 当前最活跃也最混乱；建议继续保留历史脚本，并新增文档标记推荐入口 |
| `scripts/` | 绘图、诊断、smoke test 等辅助脚本 | 建议作为所有命令行工具的统一位置 |
| `train/` | 预训练脚本和训练结果目录 | 保留预训练流程；建议避免继续新增 `train_v*.py`，改用配置区分实验 |
| `eval metrics/` | SA/QED/Novelty 等评估工具 | 目录名含空格，短期保留；长期建议迁移到 `eval_metrics/` |
| `acegen-open-main/` | 外部参考项目 | 建议视为第三方参考，不纳入主项目重构 |

## 4. 主要入口文件

| 入口 | 用途 | 状态判断 |
| --- | --- | --- |
| `main_selfies.py` | SELFIES 预训练入口 | 核心入口 |
| `fine_tuning_selfies.py` | 监督微调入口 | 核心入口 |
| `evaluate.py` | 模型评估入口，带 argparse | 核心入口 |
| `generate_selfies.py` | 生成逻辑 | 核心工具 |
| `rl/multi_obj_gspo_4_21_5_6.py` | 当前正在使用/调参的 GSPO 5_6 训练脚本 | 活跃实验入口 |
| `rl/multi_obj_gspo_4_21_5_6_20260508.py` | 20260508 GSPO 5_6 稳定变体 | 重要 baseline |
| `rl/multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | 基于 20260508 baseline 生成的 no-aux 版本 | 新增实验入口 |
| `scripts/plot_gspo_5_6_diagnostics.py` | GSPO 5_6 日志诊断绘图 | 核心辅助工具 |
| `scripts/plot_all_training_metrics.py` | 通用训练曲线绘图 | 核心辅助工具 |
| `scripts/smoke_test.py` | 基础导入/模型 smoke test | 核心检查工具 |

## 5. 文件分类建议

### 5.1 核心文件

- `config/load_config.py`
- `project_paths.py`
- `data/data_utils.py`
- `model/decoder_only_tfm.py`
- `sample/sample.py`
- `sample/sample_beam.py`
- `feedback/utils.py`
- `feedback/vina_scores.py`
- `rl/utils.py`
- `rl/multi_obj_gspo_4_21_diagnostics.py`
- `rl/multi_obj_gspo_4_21_5_6.py`
- `rl/multi_obj_gspo_4_21_5_6_20260508.py`
- `scripts/plot_gspo_5_6_diagnostics.py`
- `scripts/smoke_test.py`

### 5.2 历史实验脚本

这些文件应保留，但建议不要继续作为默认入口：

- 根目录 `grpo_*.py`
- 根目录 `g_v2.py`
- `rl/my_gspo.py`
- `rl/multi_obj_gspo_5_6.py`
- `rl/multi_obj_gspo_elite_aux_04_21.py`
- `rl/multi_obj_gspo_4_27_reward_plateau.py`
- `train/train_v2.py` 到 `train/train_v5.py`
- `model/v4.py` 到 `model/v7.py`

### 5.3 可归档候选

这些文件目前看起来更像临时调试或生成产物。建议先归档，不建议直接删除：

- `debug.py`
- `temp.py`
- `fix_encoding_temp.py`
- 根目录训练曲线 PNG：`kl_loss.png`、`policy_loss.png`、`ratio_mean.png`、`reward_metrics.png`、`top_rewards.png`、`total_reward.png`
- 各级 `__pycache__/`
- `scripts/training_metric_plots/*.png`
- `rl/feedback/logs/*_diagnostics.png`

## 6. 识别到的主要问题

| 问题 | 影响 | 处理建议 |
| --- | --- | --- |
| 根目录实验脚本过多 | 新入口难找，历史版本和当前版本混淆 | 先文档标记，再分阶段归档 |
| `rl/` 中 GSPO 版本很多 | 不清楚哪个是 baseline、哪个是当前推荐版本 | 使用 Markdown 记录每次实验文件来源和差异 |
| 配置注释和 README 存在编码异常 | 文档可读性差 | 新增 UTF-8 文档；后续单独修复编码 |
| 路径写法混杂 `../`、相对路径和绝对拼接 | 换工作目录运行时容易失败 | 统一使用 `project_paths.resolve_project_path` |
| 奖励函数和采样逻辑有重复 | RL 脚本之间重复实现，维护成本高 | 后续把稳定工具沉淀到 `rl/utils.py` 或 `feedback/` |
| 训练日志字段多且版本间不一致 | 分析脚本需要兼容多种 CSV | 推荐每个实验脚本固定日志字段，并在文档记录 |
| 第三方 ACEGEN 子项目与主项目混在一起 | 全局搜索和统计被干扰 | 文档上标记为 external reference |

## 7. 本次已实施修改

| 文件 | 修改 | 影响 |
| --- | --- | --- |
| `project_paths.py` | 增加中文模块说明、`PROJECT_ROOT`、`ensure_dir`，保留 `BASE_DIR`、`project_path`、`resolve_project_path`、`ensure_parent` | 不改变旧接口；路径职责更清楚 |
| `config/load_config.py` | 增加中文说明，加载配置前先用 `resolve_project_path` 解析配置文件路径 | 兼容相对路径配置文件；目录创建逻辑保持不变 |
| `scripts/project_inventory.py` | 新增项目盘点脚本 | 可复现地查看文件类型、入口候选、大型 Python 文件和近期 GSPO 日志 |
| `scripts/smoke_test.py` | 注入项目根目录到 `sys.path`，并将临时目录放到项目目录内 | 支持直接运行 `python scripts/smoke_test.py` |
| `docs/project_organization_20260511.md` | 新增本整理说明文档 | 记录模块职责、问题、分类和后续路线 |

## 8. 后续推荐整理路线

### 阶段 1：文档化和冻结入口

- 在 `rl/README.md` 中明确推荐训练入口。
- 在 `scripts/README.md` 中明确诊断脚本用法。
- 在 `config/README.md` 中说明每个 YAML 的用途。
- 不再新增根目录实验脚本。

### 阶段 2：轻量配置统一

- 新增 `config/rl/gspo_5_6.yaml`，把 `batch_size`、`M`、`temperature`、`top_k`、`target_kl`、`reward_weights` 等从脚本中抽出。
- 保留脚本默认值，配置只作为覆盖项。
- 日志目录和模型保存目录统一通过配置控制。

### 阶段 3：工具函数沉淀

- 把稳定的 batch 去重、advantage 准备、KL beta 调节沉淀到 `rl/utils.py` 或新模块 `rl/gspo_utils.py`。
- 保留旧脚本，但让新脚本调用公共函数。
- 对 reward schema 增加单元级 smoke test，确保 `vina_raw/vina_reward/qed/sa_reward/logp_reward/reward` 字段稳定。

### 阶段 4：归档历史实验

- 新建 `archive/experiments/` 或 `rl/archive/`。
- 只移动已确认不再直接运行的历史脚本。
- 每次移动都在文档中记录旧路径、新路径、移动原因。

## 9. 当前推荐运行命令

项目盘点：

```bash
python scripts/project_inventory.py
```

GSPO 5_6 当前活跃版本：

```bash
python rl/multi_obj_gspo_4_21_5_6.py
```

GSPO 5_6 no-aux 版本：

```bash
python rl/multi_obj_gspo_4_21_5_6_20260508_no_aux.py
```

GSPO 5_6 日志诊断：

```bash
python scripts/plot_gspo_5_6_diagnostics.py --csv rl/feedback/logs/<training_log>.csv
```

## 10. 风险说明

- 本次没有移动或删除历史文件，因此不会破坏已有入口。
- `project_paths.py` 和 `config/load_config.py` 只做兼容性增强；如果某个旧脚本依赖非常规工作目录，建议用 smoke test 单独确认。
- README 原文件存在编码异常，本次没有覆盖它，避免误伤原始内容；建议后续单独做 README UTF-8 修复。

## 11. 本次验证

已执行：

```bash
python -m py_compile project_paths.py config/load_config.py scripts/project_inventory.py
python scripts/project_inventory.py
python scripts/smoke_test.py
```

结果：

- 新增盘点脚本可以正常运行。
- 路径工具和配置加载工具语法检查通过。
- `scripts/smoke_test.py` 在项目环境中通过，输出 `smoke_test: ok`。
- 本次未执行完整训练，也未移动或删除数据、模型和历史实验文件。
