import pandas as pd
import matplotlib.pyplot as plt

# ===== 读取 CSV =====
csv_path = "grpo_training_log_20260104_145854.csv"  # 改成你的实际路径
df = pd.read_csv(csv_path)

epochs = df["epoch"]

# ===== 1. Mean Reward =====
plt.figure()
plt.plot(epochs, df["mean_reward"],linewidth=2)#marker='o',
plt.xlabel("Epoch")
plt.ylabel("Mean Reward")
plt.title("Mean Reward vs Epoch")
plt.grid(True)
plt.show()

# ===== 2. Policy Loss =====
plt.figure()
plt.plot(epochs, df["policy_loss"], linewidth=2)#marker='o',
plt.xlabel("Epoch")
plt.ylabel("Policy Loss")
plt.title("Policy Loss vs Epoch")
plt.grid(True)
plt.show()

# ===== 3. KL Loss =====
plt.figure()
plt.plot(epochs, df["kl_loss"],  linewidth=2)#marker='o',
plt.xlabel("Epoch")
plt.ylabel("KL Loss")
plt.title("KL Loss vs Epoch")
plt.grid(True)
plt.show()
