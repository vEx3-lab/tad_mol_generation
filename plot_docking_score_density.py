import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取两个 CSV
df_a = pd.read_csv("./docking/12_12_vina_results/vina_docking_scores.csv")
df_b = pd.read_csv("./docking/12_16_vina_results/vina_docking_scores.csv")
df_c = pd.read_csv('./docking/sp_vina_results/vina_docking_scores.csv')
df_d = pd.read_csv('./docking/htvs_vina_results/vina_docking_scores.csv')
scores_a = df_a["Score"].dropna()
scores_b = df_b["Score"].dropna()
scores_c = df_c["Score"].dropna()
scores_d = df_d["Score"].dropna()
plt.figure(figsize=(8, 5))

sns.kdeplot(scores_a, label="tfm", fill=True, alpha=0.4)
sns.kdeplot(scores_b, label="tfm_grpo", fill=True, alpha=0.4)
sns.kdeplot(scores_c, label="sp_data", fill=True, alpha=0.4)
sns.kdeplot(scores_d, label="htvs_data", fill=True, alpha=0.4)
plt.xlabel("Binding Energy (Vina Score)")
plt.ylabel("Density")
plt.title("KDE of Binding Energy Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
plt.figure(figsize=(8, 5))

plt.hist(scores_a, bins=20, alpha=0.6, label="tfm", density=True)
plt.hist(scores_b, bins=20, alpha=0.6, label="tfm_grpo", density=True)
plt.hist(scores_c, bins=20, alpha=0.6, label="htvs_data", density=True)

plt.xlabel("Binding Energy (Vina Score)")
plt.ylabel("Density")
plt.title("Histogram of Binding Energy")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
df_plot = pd.DataFrame({
    "Score": pd.concat([scores_a, scores_b]),
    "Group": ["tfm"] * len(scores_a) + ["tfm_grpo"] * len(scores_b)
})

plt.figure(figsize=(6, 5))
sns.violinplot(x="Group", y="Score", data=df_plot)

plt.title("Violin Plot of Binding Energy")
plt.grid(alpha=0.3)
plt.show()


# 定义函数计算描述性统计
def describe_scores(scores, name=""):
    desc = {
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "min": scores.min(),
        "25%": scores.quantile(0.25),
        "50%": scores.quantile(0.5),
        "75%": scores.quantile(0.75),
        "max": scores.max(),
        "count": len(scores)
    }
    print(f"==== {name} ====")
    for k, v in desc.items():
        print(f"{k}: {v:.4f}")
    return desc

# 统计信息
desc_a = describe_scores(scores_a, name="tfm")
desc_b = describe_scores(scores_b, name="tfm_grpo")
desc_c = describe_scores(scores_c, name="sp")
desc_d = describe_scores(scores_d, name="htvs")
def top10_stats(scores, name=""):
    best10 = scores.nsmallest(10)   # 结合能最好（最负）
    worst10 = scores.nlargest(10)   # 数值最大（最不负）

    print(f"\n==== {name} ====")
    print(f"Top-10 BEST (lowest score) mean: {best10.mean():.4f}")
    print(f"Top-10 WORST (highest score) mean: {worst10.mean():.4f}")

    return {
        "best10_mean": best10.mean(),
        "worst10_mean": worst10.mean()
    }

stats_a = top10_stats(scores_a, "tfm")
stats_b = top10_stats(scores_b, "tfm_grpo")
stats_c = top10_stats(scores_c, "sp")
stats_d = top10_stats(scores_d, "htvs")