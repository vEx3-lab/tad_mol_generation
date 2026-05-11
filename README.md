# SELFIES-Based Molecular Generation with Decoder-Only Transformer and Reinforcement Learning

## 1. 项目简介

本项目是一个面向**分子/候选药物分子生成**的深度学习框架，核心目标是利用深度生成模型结合强化学习方法，设计具有更好化学性质和靶点结合能力的候选分子。

**技术路线**：
- 使用 **SELFIES** (Self-Referencing Embedded Strings) 作为分子序列表示，保证生成分子的化学有效性
- 采用 **Decoder-Only Transformer** 进行自回归分子序列生成
- 引入 **GSPO/GRPO/REINFORCE** 等强化学习算法进行分子优化微调
- 结合 **AutoDock Vina** 对接评分、QED、SA、LogP 等多目标奖励函数

## 2. 项目特点

- **SELFIES 分子表示**：基于 SELFIES 语法，生成分子 100% 化学有效
- **Decoder-Only Transformer**：自回归生成模型，支持 temperature、top-k 采样
- **强化学习微调**：支持 GSPO、GRPO、REINFORCE 等多种 RL 算法
- **多目标奖励函数**：Vina 对接得分 + QED 类药性 + SA 合成可及性 + LogP
- **分子评估指标**：Validity、Uniqueness、Novelty、QED、SA、Vina Score 等
- **Beam Search 支持**：可选的 Beam Search 生成策略

## 3. 项目结构

```
D:\Desktop\mycode\
├── config/                          # 配置文件
│   ├── decoder_only_tfm_config.yaml
│   ├── decoder_only_tfm_config_chembridge_mpo.yaml
│   └── load_config.py
├── data/                            # 数据处理
│   ├── data_utils.py               # SelfiesDataset, selfies_vocab
│   ├── data_process.py
│   └── *.csv                       # 训练数据 (SMILES/SELFIES)
├── model/                           # 模型定义
│   ├── decoder_only_tfm.py         # 主模型：Decoder-Only Transformer
│   ├── decoder_only_tfm_v2.py
│   ├── decoder_only_tfm_v3.py
│   ├── bi_lstm.py
│   ├── tfm.py
│   └── v4.py, v5.py, v6.py, v7.py  # 历史版本
├── train/                           # 训练脚本
│   ├── train.py
│   ├── train_v2.py ~ train_v5.py
│   └── train_selfies.py            # K-Fold 预训练入口
├── sample/                          # 采样生成
│   ├── sample.py                   # Top-k 采样
│   └── sample_beam.py              # Beam Search 采样
├── rl/                              # 强化学习微调
│   ├── multi_obj_gspo.py           # GSPO (PPO) 微调
│   ├── multi_obj_gdpo.py           # GDPO 微调
│   ├── multi_obj_reinforce.py      # REINFORCE 微调
│   ├── multi_obj_gspo_beam.py      # Beam Search + GSPO
│   ├── utils.py                    # 采样与奖励函数
│   └── my_gspo.py                  # 早期版本
├── feedback/                        # 奖励与评估
│   ├── vina_scores.py              # AutoDock Vina 批量打分
│   ├── utils.py                    # 分子转换工具
│   └── grpo.py                     # GRPO 实现
├── eval metrics/                    # 评估指标
│   ├── cal_metrics.py              # 分子评估 (QED, SA, Novelty)
│   └── SA_Score/                   # SA Score 计算
├── main_selfies.py                  # 预训练入口
├── generate_selfies.py              # 生成器核心逻辑
├── fine_tuning_selfies.py           # 监督微调入口
└── README.md
```

## 4. 环境依赖

**核心依赖**（建议使用 conda 或虚拟环境）：

- Python ≥ 3.8
- PyTorch ≥ 1.9
- RDKit
- SELFIES
- NumPy, pandas
- scikit-learn
- PyYAML
- tqdm

**分子对接依赖**：
- AutoDock Vina（需单独安装并加入环境变量）
- Open Babel（用于 PDBQT 转换）

**可选依赖**：
- matplotlib（绘图）
- tensorboard（训练可视化）

## 5. 数据准备

### 5.1 分子数据

项目使用 CSV 格式的分子数据，需要包含 SMILES 或 SELFIES 列：

```
# data/htvs_molecules_with_selfies.csv 示例结构：
selfies
[C][=C][C]...
[C][C][C]...
...
```

### 5.2 受体文件

进行 Vina 对接需要准备受体 PDBQT 文件：

```
# 放置在 feedback/ 目录下
feedback/8sc7.pdbqt
```

### 5.3 配置文件

修改 `config/decoder_only_tfm_config.yaml` 中的路径配置：

```yaml
paths:
  data_file: "../data/htvs_molecules_with_selfies.csv"
  save_dir: "./model/decoder_only_tfm"
```

## 6. 模型训练

### 6.1 预训练 (K-Fold Cross Validation)

```bash
python main_selfies.py
```

预训练使用 3-Fold 交叉验证，模型保存至 `train/model/decoder_only_tfm/`。

### 6.2 监督微调

```bash
python fine_tuning_selfies.py
```

### 6.3 配置参数说明

主要训练参数在 `config/decoder_only_tfm_config.yaml` 中：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| d_model | Transformer 隐藏层维度 | 256 |
| n_heads | 注意力头数 | 4 |
| n_layers | Transformer 层数 | 3 |
| max_len | 最大序列长度 | 80 |
| batch_size | 批大小 | 64 |
| epochs | 训练轮数 | 200 |
| lr | 学习率 | 1e-3 |

## 7. 分子生成

### 7.1 Top-k 采样生成

```python
from rl.utils import sample_selfies_batch_from_generate_selfies
from data.data_utils import selfies_vocab

batch_token_ids, batch_smiles = sample_selfies_batch_from_generate_selfies(
    model=agent,
    vocab=vocab,
    batch_size=8,
    max_len=80,
    temperature=1.0,
    top_k=20,
    device="cuda"
)
```

### 7.2 Beam Search 生成

```python
from sample.sample_beam import generate_selfies_beam

result = generate_selfies_beam(
    model=agent,
    vocab=vocab,
    beam_width=5,
    max_len=80,
    temperature=1.0,
    device="cuda"
)
# result["smiles"] 为最佳 SMILES
# result["beams"] 为所有 Beam 结果
```

### 7.3 生成参数

| 参数 | 说明 |
|------|------|
| temperature | Softmax 温度，越高越随机 |
| top_k | Top-k 采样，限制候选 token 数量 |
| beam_width | Beam Search 宽度 |
| max_len | 最大生成长度 |

## 8. 强化学习微调

### 8.1 GSPO (Gradient-based Self-Positioning Optimization)

基于 PPO 的序列级策略优化：

```bash
python rl/multi_obj_gspo.py
```

### 8.2 REINFORCE

简化的策略梯度方法：

```bash
python rl/multi_obj_reinforce.py
```

### 8.3 奖励函数

奖励由多目标加权组成（可配置）：

```python
weights = {
    'vina': 3.0,   # 对接评分 (Vina score)
    'qed': 1.0,    # 类药性 (0-1)
    'sa': 0.5,     # 合成可及性 (1-10)
    'logp': 0      # 脂水分配系数
}
reward = make_composite_reward(smiles, vina_results, weights)
```

### 8.4 强化学习训练流程

1. **采样**：使用当前策略模型生成分子
2. **评估**：计算 Vina 对接得分、QED、SA 等指标
3. **奖励计算**：根据权重计算综合奖励
4. **策略更新**：执行 GSPO/REINFORCE 梯度更新
5. **保存**：定期保存最优模型和训练日志

## 9. 评价指标

### 9.1 生成质量指标

| 指标 | 说明 |
|------|------|
| Validity | 有效 SMILES 比例 |
| Uniqueness | 唯一 SMILES 比例 |
| Novelty | 相对于训练集的新颖性 |
| QED | 类药性评分 (0-1，越高越好) |
| SA | 合成可及性评分 (1-10，越低越好) |
| LogP | 脂水分配系数 |

### 9.2 对接评分

| 指标 | 说明 |
|------|------|
| Vina Score | AutoDock Vina 对接评分 (越负越好) |

### 9.3 训练监控指标

| 指标 | 说明 |
|------|------|
| policy_loss | 策略梯度损失 |
| kl_loss | KL 散度损失 |
| ratio_mean | PPO importance sampling ratio |
| baseline | REINFORCE 基准线 |

## 10. 配置文件说明

### decoder_only_tfm_config.yaml

```yaml
model_name: "decoder_only_tfm"
device: "cuda"

model:
  d_model: 256
  n_heads: 4
  n_layers: 3
  max_len: 80
  dropout: 0.25

train:
  batch_size: 64
  epochs: 200
  lr: 1e-3
  k_folds: 3

generate:
  num_samples: 1000
  temperature: 1.0
  top_k: 10
  max_len: 80
```

## 11. 输出文件说明

训练和生成过程会产生以下文件：

| 类型 | 位置 | 说明 |
|------|------|------|
| 模型权重 | `train/model/` | `.pt` 文件，建议不提交 |
| 训练日志 | `train/model/*/logs/` | CSV 格式训练记录 |
| RL 日志 | `rl/feedback/logs/` | 强化学习训练日志 |
| 生成分子 | `feedback/vina_results/` | 生成分子的 CSV |
| Docking 结果 | `feedback/vina_results/` | 对接输出文件 |

**注意**：大文件、模型权重、日志文件不建议提交到 Git，建议加入 `.gitignore`。

## 12. 注意事项

1. **路径问题**：Windows 环境下路径中避免空格和中文
2. **Vina 配置**：需要正确安装 AutoDock Vina 并设置环境变量
3. **受体文件**：准备正确的 PDBQT 格式受体文件
4. **GPU 内存**：大 batch size 可能导致 OOM，需根据显存调整
5. **数据格式**：CSV 文件需包含正确的 SMILES/SELFIES 列名

## 13. 后续改进方向

- [ ] 统一训练入口，减少重复代码
- [ ] 统一配置管理，支持 YAML/JSON 配置
- [ ] 整理日志输出，增加 TensorBoard 支持
- [ ] 增加更多 RL 算法实现（如 PPO、DPO）
- [ ] 增加单元测试
- [ ] 添加 Docker/conda 环境配置文件
- [ ] 增强分子可视化与筛选报告功能

## 14. 快速开始示例

```python
# 1. 加载数据和模型
import pandas as pd
from data.data_utils import selfies_vocab
from model.decoder_only_tfm import decoder_only_tfm

data = pd.read_csv("data/htvs_molecules_with_selfies.csv")["selfies"].tolist()
vocab = selfies_vocab(data)

# 2. 加载预训练模型
model = decoder_only_tfm(vocab_size=len(vocab), d_model=256, n_heads=4, n_layers=3)
model.load_state_dict(torch.load("train/model/decoder_only_tfm/best_model_fold1.pt"))

# 3. 生成分子
from sample.sample import generate_selfies
result = generate_selfies(model, vocab, device="cuda", max_len=80, temperature=1.0, top_k=20)
print(result["smiles"])  # 输出 SMILES 字符串
```

---

*本 README 基于项目代码自动生成，部分路径和参数可能需根据本地环境调整。*