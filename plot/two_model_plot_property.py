import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

sns.set_style("whitegrid")
sns.set_context("talk")  # 更适合论文/汇报


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


def plot_property_distributions(real_smiles, tfm_smiles, lstm_smiles, properties=None):
    """绘制训练集与两个模型生成集的化学性质分布（论文级美化）"""

    if properties is None:
        properties = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "NumRings", "TPSA"]

    real_df = compute_properties(real_smiles)
    tfm_df = compute_properties(tfm_smiles)
    lstm_df = compute_properties(lstm_smiles)

    # 颜色方案（colorbrewer 风格）
    colors = {
        "Training": "#4C72B0",  # 蓝
        "TFM": "#DD8452",       # 橙
        "Bi-LSTM": "#55A868"    # 绿
    }

    for prop in properties:
        plt.figure(figsize=(7, 5))

        sns.kdeplot(real_df[prop], label="Training Set", fill=True, alpha=0.35, color=colors["Training"])
        sns.kdeplot(tfm_df[prop], label="TFM Generated", fill=True, alpha=0.35, color=colors["TFM"])
        sns.kdeplot(lstm_df[prop], label="Bi-LSTM Generated", fill=True, alpha=0.35, color=colors["Bi-LSTM"])

        plt.title(f"{prop} Distribution Comparison", fontsize=16)
        plt.xlabel(prop, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
real_smiles = pd.read_csv("../data/htvs_molecules_with_selfies.csv")["smiles"].dropna().tolist()
tfm_smiles  = pd.read_csv("../model/decoder_only_tfm/pretrained_generated_smiles.csv")["smiles"].dropna().tolist()
lstm_smiles = pd.read_csv("../model/bi_lstm/pretrained_generated_smiles.csv")["smiles"].dropna().tolist()

plot_property_distributions(real_smiles, tfm_smiles, lstm_smiles)
