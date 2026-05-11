from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def compute_properties(smiles_list):
    """计算一组SMILES分子的基本化学性质"""
    props = {
        "MolWt": [],
        "LogP": [],
        "NumHDonors": [],
        "NumHAcceptors": [],
        "NumRings": [],
        "TPSA": []
    }

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        props["MolWt"].append(Descriptors.MolWt(mol))
        props["LogP"].append(Descriptors.MolLogP(mol))
        props["NumHDonors"].append(Descriptors.NumHDonors(mol))
        props["NumHAcceptors"].append(Descriptors.NumHAcceptors(mol))
        props["NumRings"].append(Descriptors.RingCount(mol))
        props["TPSA"].append(Descriptors.TPSA(mol))

    return pd.DataFrame(props)


def evaluate_generated_smiles_counts(generated_smiles, train_smiles=None):
    """
    计算生成 SMILES 的 valid、valid&unique、valid&unique&novel 数量。
    """
    if isinstance(generated_smiles, pd.Series):
        generated_smiles = generated_smiles.dropna().tolist()
    elif isinstance(generated_smiles, pd.DataFrame):
        generated_smiles = generated_smiles.iloc[:, 0].dropna().tolist()

    valid_smiles = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))

    valid_unique_smiles = list(set(valid_smiles))

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
