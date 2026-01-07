import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_property_distributions(real_smiles, generated_smiles, properties=None):
    """绘制训练集与生成集的性质分布"""
    if properties is None:
        properties = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "NumRings", "TPSA"]

    real_df = compute_properties(real_smiles)
    gen_df = compute_properties(generated_smiles)

    for prop in properties:
        plt.figure(figsize=(6,4))
        sns.kdeplot(real_df[prop], label="Training set", fill=True, alpha=0.4)
        sns.kdeplot(gen_df[prop], label="Generated set", fill=True, alpha=0.4)
        plt.title(f"{prop} Distribution")
        plt.xlabel(prop)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

real_smiles = pd.read_csv('../data/sp_molecules_with_selfies.csv')["smiles"].tolist()
generated_smiles = pd.read_csv('../model/decoder_only_tfm/1_fine_generated_smiles.csv')["smiles"].tolist()

plot_property_distributions(real_smiles, generated_smiles)