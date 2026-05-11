# Scripts 目录说明

本目录保存项目级辅助命令，不承载模型核心逻辑。

## 当前工具

| 文件 | 用途 |
| --- | --- |
| `project_inventory.py` | 扫描项目文件、入口候选、大型 Python 文件和近期 GSPO 日志 |
| `plot_gspo_5_6_diagnostics.py` | 绘制 GSPO 5_6 训练日志诊断图 |
| `plot_all_training_metrics.py` | 绘制通用训练指标图 |
| `plot_method_architecture.py` | 生成方法架构示意图 |
| `smoke_test.py` | 基础导入和模型 smoke test |

## 维护约定

- 新增命令行工具优先放在本目录。
- 脚本应尽量提供 `argparse` 参数，避免硬编码输入输出文件。
- 生成图片或报告时，输出目录应明确，不要默认写到项目根目录。
