import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = r"D:\Desktop\mycode_raw\rl\feedback\logs\gspo_training_log_20260506_111156.csv"

EMA_ALPHA = 0.1
RAW_ALPHA = 0.25


def ema(series, alpha=EMA_ALPHA):
    """Exponential moving average: EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.ewm(alpha=alpha, adjust=False).mean()


def add_global_step(dataframe):
    dataframe = dataframe.copy()

    if "global_step" in dataframe.columns:
        dataframe["global_step"] = pd.to_numeric(dataframe["global_step"], errors="coerce")
    elif {"iteration", "step"}.issubset(dataframe.columns):
        iteration = pd.to_numeric(dataframe["iteration"], errors="coerce")
        step = pd.to_numeric(dataframe["step"], errors="coerce")
        max_step = int(step.max())
        dataframe["global_step"] = (iteration - 1) * max_step + step
    elif "step" in dataframe.columns:
        dataframe["global_step"] = pd.to_numeric(dataframe["step"], errors="coerce")
    else:
        dataframe["global_step"] = range(1, len(dataframe) + 1)

    return dataframe.sort_values("global_step").reset_index(drop=True)


def show_plot(title, ylabel):
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_raw_ema(df, column, title, ylabel, label=None, ideal_line=None):
    if column not in df.columns:
        return

    label = label or column
    plt.figure()
    plt.plot(df["global_step"], df[column], alpha=RAW_ALPHA, label=f"{label}_raw")
    plt.plot(df["global_step"], ema(df[column]), linewidth=2, label=f"{label}_ema")

    if ideal_line is not None:
        plt.axhline(ideal_line, linestyle="--", label=f"ideal={ideal_line}")

    show_plot(title, ylabel)


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


df = pd.read_csv(CSV_PATH)
df = add_global_step(df)

# Reward metrics. Prefer normalized reward-space columns from diagnostics logs.
metric_columns = [
    (first_existing_column(df, ["mean_vina_reward", "mean_vina"]), "vina"),
    (first_existing_column(df, ["mean_qed"]), "qed"),
    (first_existing_column(df, ["mean_sa_reward", "mean_sa"]), "sa"),
    (first_existing_column(df, ["mean_logp_reward", "mean_logp"]), "logp"),
]

plt.figure()
has_metric = False
for column, label in metric_columns:
    if column is None:
        continue
    plt.plot(df["global_step"], ema(df[column]), linewidth=2, label=f"{label}_ema")
    has_metric = True

if has_metric:
    show_plot("Reward Metrics with EMA", "Score")
else:
    plt.close()

# Raw docking score is useful when diagnostics logs include it.
plot_raw_ema(
    df,
    "mean_vina_raw",
    "Vina Raw Docking Score with EMA",
    "Vina Score (kcal/mol)",
    label="vina_raw",
)

# Total reward. Use mean_reward when available; legacy fallback is kept only for old logs.
if "mean_reward" in df.columns:
    df["total_reward"] = pd.to_numeric(df["mean_reward"], errors="coerce")
elif {"mean_vina", "mean_qed", "mean_sa"}.issubset(df.columns):
    df["total_reward"] = -df["mean_vina"] + df["mean_qed"] - df["mean_sa"]

plot_raw_ema(df, "total_reward", "Total Reward Curve with EMA", "Total Reward", label="reward")

# Elite reward curves.
plt.figure()
has_top_reward = False
for column in ["top1_reward", "top5_mean_reward"]:
    if column not in df.columns:
        continue
    plt.plot(df["global_step"], ema(df[column]), linewidth=2, label=f"{column}_ema")
    has_top_reward = True

if has_top_reward:
    show_plot("Top Reward Curves with EMA", "Reward")
else:
    plt.close()

# Loss and stability curves.
plot_raw_ema(df, "policy_loss", "Policy Loss with EMA", "Policy Loss", label="policy_loss")
plot_raw_ema(df, "kl_loss", "KL Divergence with EMA", "KL Loss", label="kl_loss")
plot_raw_ema(
    df,
    "ratio_mean",
    "PPO / GSPO Ratio Mean with EMA",
    "Ratio Mean",
    label="ratio_mean",
    ideal_line=1.0,
)
