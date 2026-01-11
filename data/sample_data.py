import pandas as pd

# 读取原始 CSV
df = pd.read_csv("htvs_molecules_with_selfies.csv")

# 随机采样 1000 条（不放回）
df_sampled = df.sample(n=1000, random_state=45)

# 保存为新的 CSV
df_sampled.to_csv("sampled_1000_htvs_molecules.csv", index=False)
