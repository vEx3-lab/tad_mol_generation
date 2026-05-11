import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 1. 读取 CSV
# ======================
csv_path = "./grpo_training_log_20260105_212356.csv"   # 改成你的路径
df = pd.read_csv(csv_path)

# ======================
# 2. 构造 global_step
# ======================
if "step" in df.columns:
    df["global_step"] = (
        df["iteration"].astype(str) + "_" + df["step"].astype(str)
    )
    df["global_step"] = range(len(df))
else:
    df["global_step"] = df["iteration"]

# 排序（保险）
df = df.sort_values("global_step").reset_index(drop=True)

# ======================
# 3. 可选：滑动平均（RL 常用）
# ======================
SMOOTH_WINDOW = 5  # 不想平滑就设为 1

def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1).mean()

# ======================
# 4. Reward 曲线
# ======================
plt.figure()
plt.plot(df["global_step"], smooth(df["mean_reward"], SMOOTH_WINDOW), label="mean_reward")
plt.plot(df["global_step"], smooth(df["top1_reward"], SMOOTH_WINDOW), label="top1_reward")
plt.plot(df["global_step"], smooth(df["top5_mean_reward"], SMOOTH_WINDOW), label="top5_mean_reward")
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("Reward Curve")
plt.legend()
plt.grid(True)
plt.show()

# ======================
# 5. Policy Loss
# ======================
plt.figure()
plt.plot(df["global_step"], smooth(df["policy_loss"], SMOOTH_WINDOW))
plt.xlabel("Training Step")
plt.ylabel("Policy Loss")
plt.title("Policy Loss")
plt.grid(True)
plt.show()

# ======================
# 6. KL Loss
# ======================
plt.figure()
plt.plot(df["global_step"], smooth(df["kl_loss"], SMOOTH_WINDOW))
plt.xlabel("Training Step")
plt.ylabel("KL Loss")
plt.title("KL Divergence")
plt.grid(True)
plt.show()

# ======================
# 7. Ratio Mean（最关键的稳定性指标）
# ======================
plt.figure()
plt.plot(df["global_step"], smooth(df["ratio_mean"], SMOOTH_WINDOW))
plt.xlabel("Training Step")
plt.ylabel("Ratio Mean")
plt.title("PPO / GRPO Ratio Mean")
plt.grid(True)
plt.show()
