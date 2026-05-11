# Config 目录说明

本目录保存项目配置和配置加载工具。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `decoder_only_tfm_config.yaml` | Decoder-only Transformer 主配置 |
| `decoder_only_tfm_config_chembridge_mpo.yaml` | ChemBridge/MPO 相关配置 |
| `bi_lstm_config.yaml` | BiLSTM 配置 |
| `load_config.py` | YAML 配置加载与输出目录创建 |

## 维护约定

- 路径字段统一放在 `paths` 下。
- 相对路径建议以项目根目录为基准，并通过 `project_paths.resolve_project_path` 解析。
- 后续 RL 实验配置建议放到 `config/rl/`，例如 `config/rl/gspo_5_6.yaml`。
- 不建议继续把大量实验超参硬编码在训练脚本底部。
