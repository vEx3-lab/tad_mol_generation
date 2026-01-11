from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm

def randomize_smile(sml):
    """
    随机化单个 SMILES 序列。
    Args:
        sml (str): 输入的 SMILES。
    Returns:
        str 或 None: 随机化后的 SMILES；若无效返回 None。
    """
    try:
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            return None
        atom_indices = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_indices)
        new_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(new_mol, canonical=False)
    except Exception:
        return None


def augment_smiles(smiles_list, n_aug=5, drop_invalid=True, seed=42):
    """
    对一批 SMILES 序列进行数据增强。
    Args:
        smiles_list (list[str]): 原始 SMILES 序列列表。
        n_aug (int): 每条 SMILES 随机化次数。
        drop_invalid (bool): 是否丢弃无效 SMILES。
        seed (int): 随机种子。
    Returns:
        pd.DataFrame: 包含原始 SMILES 与增强后 SMILES 的 DataFrame。
    """
    np.random.seed(seed)
    augmented_data = []

    for s in tqdm(smiles_list, desc="Augmenting SMILES"):
        for _ in range(n_aug):
            new_smi = randomize_smile(s)
            if new_smi is None and drop_invalid:
                continue
            augmented_data.append((s, new_smi))

    df_aug = pd.DataFrame(augmented_data, columns=["original", "augmented"])
    df_aug.dropna(inplace=True)
    df_aug.drop_duplicates(subset=["augmented"], inplace=True)
    return df_aug


if __name__ == '__main__':
    smiles_list = pd.read_csv("./sp_molecules_with_selfies.csv")['smiles'].tolist()
    n_aug = 1
    df_aug = augment_smiles(smiles_list, n_aug=n_aug)

    df_aug.to_csv(f"./augmented_{n_aug}_smiles_sp.csv", index=False)
    print(f"✅ 已保存增强后的 SMILES，共 {len(df_aug)} 条")