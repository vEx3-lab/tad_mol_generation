import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取两个 CSV
df_a = pd.read_csv("pretrained_generated_smiles_with_sa_mpo.csv")
df_b = pd.read_csv("grpo_generated_smiles_with_sa_mpo.csv")
df_c = pd.read_csv('sp_molecules_with_sa_mpo.csv')
df_d = pd.read_csv('htvs_molecules_with_sa_mpo.csv')
scores_a = df_a["CNS_MPO"].dropna()
scores_b = df_b["CNS_MPO"].dropna()
scores_c = df_c["CNS_MPO"].dropna()
scores_d = df_d["CNS_MPO"].dropna()
plt.figure(figsize=(8, 5))

sns.kdeplot(scores_a, label="tfm", fill=True, alpha=0.4)
sns.kdeplot(scores_b, label="tfm_grpo", fill=True, alpha=0.4)
sns.kdeplot(scores_c, label="sp_data", fill=True, alpha=0.4)
sns.kdeplot(scores_d, label="htvs_data", fill=True, alpha=0.4)
plt.xlabel("CNS-MPO")
plt.ylabel("Density")
plt.title("KDE of CNS-MPO Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.show()