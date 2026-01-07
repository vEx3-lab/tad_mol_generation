from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt

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
        else:
            print(smi)

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

def plot_generation_counts(counts_dict, title="SMILES Generation Counts"):
    """
    绘制 Valid / Valid&Unique / Valid&Unique&Novel 样本数量柱状图
    """
    labels = ['Valid', 'Valid & Unique', 'Valid & Unique & Novel']
    values = [
        counts_dict.get('valid_count', 0),
        counts_dict.get('valid_unique_count', 0),
        counts_dict.get('novel_count', 0)
    ]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['#4c72b0', '#55a868', '#c44e52'])

    # 在柱上显示数值
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}', ha='center', fontsize=11)

    plt.ylabel("Count")
    plt.title(title)
    plt.ylim(0, max(values) * 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gen_df = pd.read_csv("../model/decoder_only_tfm/2_fine_generated_smiles.csv")
    train_df = pd.read_csv("../data/sp_molecules_with_selfies.csv")
    print(len(gen_df))
    counts = evaluate_generated_smiles_counts(
        generated_smiles=gen_df["smiles"],
        train_smiles=train_df["smiles"].tolist()
    )

    plot_generation_counts(counts, title="Fold 1 Generated SMILES Counts")
