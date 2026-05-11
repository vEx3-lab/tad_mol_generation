import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "rl" / "feedback" / "logs"
DEFAULT_PATTERN = "gspo_5_6_training_log_*.csv"
# DEFAULT_PATTERN  = 'gspo_5_6_training_log_20260508_095722.csv'

def find_latest_log(log_dir=DEFAULT_LOG_DIR, pattern=DEFAULT_PATTERN):
    paths = sorted(
        Path(log_dir).glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No logs matched {Path(log_dir) / pattern}")
    return paths[0]


def numeric(df, column):
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def smooth(series, window):
    if series is None:
        return None
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def add_global_step(df):
    df = df.copy()
    if "global_step" in df.columns:
        df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    elif {"iteration", "step"}.issubset(df.columns):
        iteration = pd.to_numeric(df["iteration"], errors="coerce")
        step = pd.to_numeric(df["step"], errors="coerce")
        max_step = int(step.max())
        df["global_step"] = (iteration - 1) * max_step + step
    elif "step" in df.columns:
        df["global_step"] = pd.to_numeric(df["step"], errors="coerce")
    else:
        df["global_step"] = range(1, len(df) + 1)
    return df.sort_values("global_step").reset_index(drop=True)


def first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def plot_series(ax, df, columns, title, ylabel, window, raw_alpha=0.18):
    x = df["global_step"]
    plotted = False
    for column, label in columns:
        series = numeric(df, column)
        if series is None:
            continue
        ax.plot(x, series, alpha=raw_alpha, linewidth=0.8)
        ax.plot(x, smooth(series, window), linewidth=1.8, label=label)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("global_step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "missing columns", ha="center", va="center", transform=ax.transAxes)


def summarize(df):
    lines = []
    lines.append(f"rows={len(df)}")
    if len(df) == 0:
        return lines

    for column in [
        "mean_reward",
        "mean_reward_ema",
        "top5_mean_reward",
        "top5_reward_ema",
        "mean_vina_raw",
        "mean_vina_reward",
        "kl_beta",
        "ratio_mean",
        "aux_loss",
        "grad_norm",
    ]:
        series = numeric(df, column)
        if series is None or series.dropna().empty:
            continue
        values = series.dropna()
        lines.append(
            f"{column}: first={values.iloc[0]:.6g} "
            f"last={values.iloc[-1]:.6g} mean={values.mean():.6g} "
            f"min={values.min():.6g} max={values.max():.6g}"
        )

    window = min(30, max(1, len(df) // 3))
    if len(df) >= window * 2:
        for column in ["mean_reward", "top5_mean_reward", "mean_vina_raw", "aux_loss"]:
            series = numeric(df, column)
            if series is None:
                continue
            values = series.dropna()
            if len(values) < window * 2:
                continue
            lines.append(
                f"{column} window: first{window}={values.iloc[:window].mean():.6g} "
                f"last{window}={values.iloc[-window:].mean():.6g}"
            )
    return lines


def build_plot(df, csv_path, output_path, smooth_window):
    df = add_global_step(df)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)
    axes = axes.ravel()

    reward_columns = [
        ("mean_reward", "mean_reward"),
        ("mean_reward_ema", "mean_reward_ema"),
        ("top5_mean_reward", "top5_mean_reward"),
        ("top5_reward_ema", "top5_reward_ema"),
    ]
    plot_series(axes[0], df, reward_columns, "Reward Trend", "reward", smooth_window)

    component_columns = [
        (first_existing(df, ["mean_vina_reward", "mean_vina"]), "vina_reward"),
        ("mean_qed", "qed"),
        (first_existing(df, ["mean_sa_reward", "mean_sa"]), "sa_reward"),
        (first_existing(df, ["mean_logp_reward", "mean_logp"]), "logp_reward"),
    ]
    component_columns = [(c, label) for c, label in component_columns if c is not None]
    plot_series(axes[1], df, component_columns, "Reward Components", "component score", smooth_window)

    plot_series(
        axes[2],
        df,
        [("mean_vina_raw", "vina_raw")],
        "Raw Docking Score",
        "kcal/mol (lower is better)",
        smooth_window,
    )
    axes[2].invert_yaxis()

    plot_series(
        axes[3],
        df,
        [("policy_loss", "policy"), ("gspo_loss", "gspo"), ("kl_loss", "kl")],
        "Policy / KL Loss",
        "loss",
        smooth_window,
    )

    plot_series(
        axes[4],
        df,
        [("aux_loss", "aux"), ("total_loss", "total")],
        "Auxiliary and Total Loss",
        "loss",
        smooth_window,
    )

    plot_series(
        axes[5],
        df,
        [("ratio_mean", "ratio_mean"), ("kl_beta", "kl_beta")],
        "GSPO Ratio and KL Beta",
        "value",
        smooth_window,
    )
    axes[5].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ratio=1")
    axes[5].legend(fontsize=8)

    sample_columns = [
        ("valid_count", "valid"),
        ("dedup_count", "dedup"),
        ("raw_sample_count", "raw"),
        ("kept_samples", "kept"),
        ("duplicate_removed", "dup_removed"),
    ]
    plot_series(axes[6], df, sample_columns, "Sample Health", "count", smooth_window)

    stability_columns = [
        ("grad_norm", "grad_norm"),
        ("adv_std", "adv_std"),
        ("adv_max_abs", "adv_max_abs"),
        ("plateau_hits", "plateau_hits"),
    ]
    plot_series(axes[7], df, stability_columns, "Optimization Stability", "value", smooth_window)

    fig.suptitle(f"GSPO 5_6 Diagnostics: {csv_path.name}", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GSPO 5_6 training diagnostics.")
    parser.add_argument("--csv", type=Path, default=None, help="Training log CSV. Defaults to latest GSPO 5_6 log.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--smooth", type=int, default=10, help="Rolling mean window.")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else find_latest_log()
    csv_path = csv_path.resolve()
    df = pd.read_csv(csv_path)
    df = add_global_step(df)

    output_path = args.out
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_diagnostics.png")
    output_path = output_path.resolve()

    build_plot(df, csv_path, output_path, max(1, args.smooth))

    print(f"csv={csv_path}")
    print(f"output={output_path}")
    for line in summarize(df):
        print(line)


if __name__ == "__main__":
    main()
