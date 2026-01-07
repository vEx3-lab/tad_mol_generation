from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt


def plot_two_models_counts(counts_tfm, counts_lstm, labels=("TFM", "Bi-LSTM")):
    """
    将两个模型的生成统计结果画在同一张图中

    counts_tfm / counts_lstm 结构为：
    {
        "valid_count": ...,
        "valid_unique_count": ...,
        "novel_count": ...
    }
    """

    categories = ["Valid", "Valid & Unique", "Valid & Unique & Novel"]
    tfm_values = [
        counts_tfm["valid_count"],
        counts_tfm["valid_unique_count"],
        counts_tfm["novel_count"]
    ]
    lstm_values = [
        counts_lstm["valid_count"],
        counts_lstm["valid_unique_count"],
        counts_lstm["novel_count"]
    ]

    x = range(len(categories))
    width = 0.35  # 柱宽

    plt.figure(figsize=(8, 5))

    # 两类模型并排画
    plt.bar([p - width / 2 for p in x], tfm_values, width,
            label=labels[0], color="#4c72b0")
    plt.bar([p + width / 2 for p in x], lstm_values, width,
            label=labels[1], color="#c44e52")

    # 数值标签
    for p, val in zip([p - width / 2 for p in x], tfm_values):
        plt.text(p, val + 1, str(val), ha='center')
    for p, val in zip([p + width / 2 for p in x], lstm_values):
        plt.text(p, val + 1, str(val), ha='center')

    plt.xticks(x, categories)
    plt.ylabel("Count")
    plt.title("TFM vs Bi-LSTM — SMILES Generation Statistics")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_generated_smiles_counts(generated_smiles, train_smiles=None):
    """
    计算生成 SMILES 的 valid、valid&unique、valid&unique&novel 数量。

    Args:
        generated_smiles (list or pd.Series): 生成的 SMILES 列表
        train_smiles (list, optional): 训练集中的 SMILES，用于计算 novel

    Returns:
        dict: {"valid_count", "valid_unique_count", "novel_count"}
    """
    # 转列表
    if isinstance(generated_smiles, pd.Series):
        generated_smiles = generated_smiles.dropna().tolist()
    elif isinstance(generated_smiles, pd.DataFrame):
        generated_smiles = generated_smiles.iloc[:, 0].dropna().tolist()

    # 1️⃣ valid
    valid_smiles = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))  # 标准化

    # 2️⃣ valid & unique
    valid_unique_smiles = list(set(valid_smiles))

    # 3️⃣ valid & unique & novel
    if train_smiles is not None:
        train_set = set(train_smiles)
        novel_smiles = [s for s in valid_unique_smiles if s not in train_set]
    else:
        novel_smiles = []

    result = {
        "valid_count": len(valid_smiles),
        "valid_unique_count": len(valid_unique_smiles),
        "novel_count": len(novel_smiles),
        "total_generated": len(generated_smiles)
    }

    print("=== Generation Evaluation Counts ===")
    print(f"Total Generated: {result['total_generated']}")
    print(f"Valid: {result['valid_count']}")
    print(f"Valid & Unique: {result['valid_unique_count']}")
    print(f"Valid & Unique & Novel: {result['novel_count']}")

    return result
if __name__ == '__main__':

    # ====== 读 TFM 生成结果 ======
    tfm_gen = pd.read_csv("../model/decoder_only_tfm/pretrained_generated_smiles.csv")
    tfm_counts = evaluate_generated_smiles_counts(
        tfm_gen["smiles"],
        train_smiles=pd.read_csv("../data/htvs_molecules_with_selfies.csv")["smiles"].tolist()
    )

    # ====== 读 Bi-LSTM 生成结果 ======
    lstm_gen = pd.read_csv("../model/bi_lstm/pretrained_generated_smiles.csv")
    lstm_counts = evaluate_generated_smiles_counts(
        lstm_gen["smiles"],
        train_smiles=pd.read_csv("../data/htvs_molecules_with_selfies.csv")["smiles"].tolist()
    )

    # ====== 两个模型并排可视化 ======
    plot_two_models_counts(tfm_counts, lstm_counts)
