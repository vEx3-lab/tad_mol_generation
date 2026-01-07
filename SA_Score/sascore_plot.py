import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取两个 CSV
df_a = pd.read_csv('./htvs_molecules_wih_sa.csv')
df_b = pd.read_csv("./sp_molecules_with_sa.csv")
df_c = pd.read_csv('./pretrained_generated_smiles_with_sa.csv')
df_d = pd.read_csv('./grpo_generated_smiles_with_sa.csv')
scores_a = df_a["sa_score"].dropna()
scores_b = df_b["sa_score"].dropna()
scores_c = df_c["sa_score"].dropna()
scores_d = df_d["sa_score"].dropna()
plt.figure(figsize=(8, 5))

sns.kdeplot(scores_a, label="htvs_data", fill=True, alpha=0.4)
sns.kdeplot(scores_b, label="sp_data", fill=True, alpha=0.4)
sns.kdeplot(scores_c, label="tfm", fill=True, alpha=0.4)
sns.kdeplot(scores_d, label="tfm_grpo", fill=True, alpha=0.4)
plt.xlabel("SA_Score")
plt.ylabel("Density")
plt.title("KDE of SA_Score Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(8, 5))
