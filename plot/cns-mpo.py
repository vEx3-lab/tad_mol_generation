from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import math
import pandas as pd

# ---------- CNS-MPO 单项评分（sigmoid 类型评分函数） ----------
def score_clogp(x):
    return 1 - abs(x - 3) / 3 if 0 <= x <= 6 else 0

def score_clogd(x):
    return 1 - abs(x - 3) / 3 if 0 <= x <= 6 else 0

def score_mw(x):
    return 1 - abs(x - 460) / 140 if 320 <= x <= 600 else 0

def score_tpsa(x):
    return 1 - x / 90 if 0 <= x <= 90 else 0

def score_hbd(x):
    return 1 - x / 3 if 0 <= x <= 3 else 0

def score_pka(x):
    return 1 - abs(x - 9) / 3 if 6 <= x <= 12 else 0

def calc_cns_mpo(mol):

    clogp = Crippen.MolLogP(mol)
    clogd = clogp  # 简化假设 pH=7.4 下相同
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)

    # pKa 简化为最强碱性原子的预测（如果没有 pKa 模型）
    # 若你有更准确 pKa 模型，可替换这里
    pka = 7.5

    # 打分
    total = (
        score_clogp(clogp)
        + score_clogd(clogd)
        + score_mw(mw)
        + score_tpsa(tpsa)
        + score_hbd(hbd)
        + score_pka(pka)
    )

    return total



# ----------------- 批量处理 SMILES -----------------
def compute_scores(smiles_list):
    results = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append([smi, None, None])
            continue

        cns_mpo = calc_cns_mpo(mol)

        results.append([smi, cns_mpo])

    df = pd.DataFrame(results, columns=["SMILES", "CNS-MPO"])
    return df


def add_cns_mpo_to_csv(input_csv, output_csv=None, smiles_col="SMILES"):
    df = pd.read_csv(input_csv)

    cns_scores = []

    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            cns_scores.append(None)
        else:
            cns_scores.append(calc_cns_mpo(mol))

    df["CNS_MPO"] = cns_scores

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df
df = add_cns_mpo_to_csv(
    input_csv="pretrained_generated_smiles_with_sa.csv",
    output_csv="pretrained_generated_smiles_with_sa_mpo.csv",
    smiles_col="smiles"
)

print(df.head())
df = add_cns_mpo_to_csv(
    input_csv="htvs_molecules_with_sa.csv",
    output_csv="htvs_molecules_with_sa_mpo.csv",
    smiles_col="smiles"
)

print(df.head())

df = add_cns_mpo_to_csv(
    input_csv="sp_molecules_with_sa.csv",
    output_csv="sp_molecules_with_sa_mpo.csv",
    smiles_col="smiles"
)

print(df.head())

df = add_cns_mpo_to_csv(
    input_csv="grpo_generated_smiles_with_sa.csv",
    output_csv="grpo_generated_smiles_with_sa_mpo.csv",
    smiles_col="smiles"
)

print(df.head())
