from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_generated_smiles(generated_smiles, train_smiles=None):
    """
    计算生成 SMILES 的 valid、unique、novel 比例。

    Args:
        generated_smiles (list or pd.Series): 生成的 SMILES 列表
        train_smiles (list, optional): 训练集中的 SMILES，用于计算 novel

    Returns:
        dict: {"valid": float, "unique": float, "novel": float}
    """
    if isinstance(generated_smiles, pd.Series):
        generated_smiles = generated_smiles.dropna().tolist()
    elif isinstance(generated_smiles, pd.DataFrame):
        generated_smiles = generated_smiles.iloc[:, 0].dropna().tolist()

    # ---- 1️⃣ 检查 valid ----
    valid_smiles = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))  # 标准化

    valid_ratio = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0

    # ---- 2️⃣ unique ----
    unique_smiles = list(set(valid_smiles))
    unique_ratio = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    # ---- 3️⃣ novel ----
    if train_smiles is not None:
        train_smiles_set = set(train_smiles)
        novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
        novel_ratio = len(novel_smiles) / len(valid_smiles) if valid_smiles else 0
    else:
        novel_ratio = None  # 如果没有训练集，无法计算 novel

    result = {
        "valid": round(valid_ratio, 4),
        "unique": round(unique_ratio, 4),
        "novel": round(novel_ratio, 4) if novel_ratio is not None else None,
        "valid_count": len(valid_smiles),
        "total_generated": len(generated_smiles)
    }

    print("=== Generation Evaluation ===")
    print(f"Valid: {result['valid'] * 100:.2f}%  ({result['valid_count']}/{result['total_generated']})")
    print(f"Unique: {result['unique'] * 100:.2f}%")
    if novel_ratio is not None:
        print(f"Novel: {result['novel'] * 100:.2f}%")

    return result


def plot_generation_metrics(metrics_dict, title="SMILES Generation Metrics"):
    """
    绘制 Valid / Unique / Novel 条形图

    Args:
        metrics_dict: dict, 需包含 'valid', 'unique', 'novel' 三个 key
    """
    labels = ['Valid', 'Unique', 'Novel']
    values = [
        metrics_dict.get('valid', 0),
        metrics_dict.get('unique', 0),
        metrics_dict.get('novel', 0)
    ]

    # 转换为百分比
    values_pct = [v * 100 if v is not None else 0 for v in values]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values_pct, color=['#4c72b0', '#55a868', '#c44e52'])

    # 在每个柱上显示数值
    for bar, val in zip(bars, values_pct):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', fontsize=11)

    plt.ylim(0, 110)
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # gen_df = pd.read_csv("../model/decoder_only_tfm_1111_1313/generated_smiles.csv")
    gen_df = pd.read_csv("../model/decoder_only_tfm_1111_1313/fold2_generated_smiles.csv")
    train_df = pd.read_csv("../data/htvs_molecules_with_selfies.csv")

    result = evaluate_generated_smiles(
        generated_smiles=gen_df["generated_smiles"],
        train_smiles=train_df["smiles"].tolist()
    )
    print(result)

    plot_generation_metrics(result, title="Fold 1 Generated SMILES Metrics")
