import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMOOTH_WINDOW = 10
METRICS_PER_PAGE = 24
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "rl/feedback/logs/gspo_5_6_training_log_20260507_161438.csv"


def add_reward_contributions(df):
    weights = {
        "vina": ("mean_vina_reward", 1.2),
        "qed": ("mean_qed", 0.2),
        "sa": ("mean_sa_reward", 0.15),
        "logp": ("mean_logp_reward", 0.0),
    }
    for name, (column, weight) in weights.items():
        if column in df.columns:
            df[f"contrib_{name}"] = pd.to_numeric(df[column], errors="coerce") * weight


def choose_x_column(df):
    for column in ("global_step", "step", "iteration"):
        if column in df.columns:
            return column
    df.insert(0, "row_index", np.arange(len(df)))
    return "row_index"


def numeric_metric_columns(df, x_column):
    metrics = []
    for column in df.columns:
        if column == x_column:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            df[column] = values
            metrics.append(column)
    return metrics


def plot_metric_page(df, x_column, metrics, output_path, page_index):
    rows = math.ceil(len(metrics) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(20, max(4, rows * 3)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    x = pd.to_numeric(df[x_column], errors="coerce")

    for ax, metric in zip(axes, metrics):
        y = pd.to_numeric(df[metric], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.any():
            ax.plot(x[mask], y[mask], color="#9ecae1", alpha=0.35, linewidth=0.8)
            if mask.sum() >= SMOOTH_WINDOW:
                smooth = y[mask].rolling(SMOOTH_WINDOW, min_periods=1).mean()
                ax.plot(x[mask], smooth, color="#f28e2b", linewidth=1.6)
        ax.set_title(metric, fontsize=10)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(f"Training Metrics Page {page_index}", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def available_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output filename near {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot simple metric curves for a training CSV.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Training CSV file. Defaults to {DEFAULT_CSV}.",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    add_reward_contributions(df)
    x_column = choose_x_column(df)
    metrics = numeric_metric_columns(df, x_column)

    if not metrics:
        raise ValueError(f"No numeric metrics found in {csv_path}")

    output_paths = []
    output_dir = Path.cwd() / "training_metric_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(metrics), METRICS_PER_PAGE):
        page = len(output_paths) + 1
        page_metrics = metrics[start:start + METRICS_PER_PAGE]
        suffix = "" if len(metrics) <= METRICS_PER_PAGE else f"_page{page}"
        output_path = available_path(output_dir / f"metrics{suffix}.png")
        plot_metric_page(df, x_column, page_metrics, output_path, page)
        output_paths.append(output_path)

    print(f"csv={csv_path}")
    print(f"x={x_column}")
    print(f"metrics={len(metrics)}")
    for path in output_paths:
        print(f"saved={path}")


if __name__ == "__main__":
    main()
