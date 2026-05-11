from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "paper"


def add_box(ax, xy, width, height, title, lines, facecolor, edgecolor="#243447", lw=1.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#16212b",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#243447",
        linespacing=1.35,
    )
    return patch


def add_arrow(ax, start, end, color="#425466", lw=2.0, style="-|>", mutation_scale=16, ls="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=ls,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def add_label(ax, xy, text, fontsize=10, color="#425466", ha="center"):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color, ha=ha, va="center")


def draw_method_architecture():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.03,
        0.965,
        "Method Overview of the Molecular Generation Framework",
        fontsize=20,
        fontweight="bold",
        color="#102030",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.932,
        "SELFIES-based decoder-only generation with on-policy GSPO, elite replay auxiliary learning, "
        "Vina caching, and scaffold-aware diversity control",
        fontsize=11.5,
        color="#4b5b6b",
        ha="left",
        va="top",
    )

    # Panel labels
    ax.text(0.03, 0.87, "(a) Pretraining and model backbone", fontsize=13, fontweight="bold", color="#102030")
    ax.text(0.03, 0.47, "(b) Reinforcement learning fine-tuning", fontsize=13, fontweight="bold", color="#102030")

    # Top panel
    box_data = add_box(
        ax,
        (0.05, 0.63),
        0.22,
        0.18,
        "Data Preparation",
        [
            "SMILES -> SELFIES",
            "vocabulary construction",
            "padded autoregressive dataset",
            "train / fine-tune split",
        ],
        facecolor="#e9f5ff",
    )
    box_model = add_box(
        ax,
        (0.39, 0.60),
        0.24,
        0.24,
        "Decoder-Only Transformer",
        [
            "token embedding + position embedding",
            "causal self-attention encoder blocks",
            "language-model head over SELFIES tokens",
            "next-token generation",
        ],
        facecolor="#eef7ea",
    )
    box_supervised = add_box(
        ax,
        (0.74, 0.63),
        0.21,
        0.18,
        "Supervised Optimization",
        [
            "cross-entropy next-token loss",
            "K-fold pretraining",
            "task-specific fine-tuning",
            "initialize RL policy",
        ],
        facecolor="#fff3df",
    )

    add_arrow(ax, (0.27, 0.72), (0.39, 0.72))
    add_arrow(ax, (0.63, 0.72), (0.74, 0.72))
    add_label(ax, (0.33, 0.755), "SELFIES token stream")
    add_label(ax, (0.685, 0.755), "cross-entropy training")

    # Separator
    ax.plot([0.03, 0.97], [0.53, 0.53], color="#d7dee5", linewidth=1.5, linestyle="--")

    # Bottom panel boxes
    box_policy = add_box(
        ax,
        (0.05, 0.24),
        0.20,
        0.16,
        "Policy Rollout",
        [
            "old policy samples molecules",
            "SELFIES-constrained decoding",
            "temperature / top-k sampling",
        ],
        facecolor="#e9f5ff",
    )
    box_reward = add_box(
        ax,
        (0.31, 0.18),
        0.21,
        0.28,
        "Multi-Objective Scoring",
        [
            "AutoDock Vina score with cache",
            "QED, SA, LogP reward terms",
            "composite scalar reward",
            "scaffold bonus / duplicate penalty",
        ],
        facecolor="#fff3df",
    )
    box_gspo = add_box(
        ax,
        (0.59, 0.28),
        0.19,
        0.14,
        "On-Policy GSPO",
        [
            "sequence-level clipped ratio",
            "KL regularization to reference policy",
        ],
        facecolor="#eef7ea",
    )
    box_memory = add_box(
        ax,
        (0.59, 0.10),
        0.19,
        0.12,
        "Elite Replay Memory",
        [
            "high-reward molecules only",
            "scaffold-aware filtering",
        ],
        facecolor="#f9eef8",
    )
    box_aux = add_box(
        ax,
        (0.82, 0.12),
        0.13,
        0.16,
        "Auxiliary Likelihood",
        [
            "reward-weighted NLL",
            "replay gradients",
            "stabilize policy drift",
        ],
        facecolor="#f1f6fe",
    )
    box_update = add_box(
        ax,
        (0.82, 0.34),
        0.13,
        0.12,
        "Updated Agent",
        [
            "best model checkpoints",
            "training log export",
        ],
        facecolor="#eef7ea",
    )

    # RL arrows
    add_arrow(ax, (0.25, 0.32), (0.31, 0.32))
    add_arrow(ax, (0.52, 0.34), (0.59, 0.35))
    add_arrow(ax, (0.52, 0.23), (0.59, 0.16))
    add_arrow(ax, (0.78, 0.35), (0.82, 0.39))
    add_arrow(ax, (0.78, 0.16), (0.82, 0.20))
    add_arrow(ax, (0.885, 0.28), (0.885, 0.34), style="-|>", lw=2.0)
    add_arrow(ax, (0.82, 0.40), (0.25, 0.40), color="#5d6d7e", ls="--")
    add_label(ax, (0.69, 0.435), "parameter update", fontsize=10)
    add_label(ax, (0.885, 0.305), "L_total", fontsize=10)
    add_label(ax, (0.36, 0.355), "dock / evaluate / reward", fontsize=10)
    add_label(ax, (0.56, 0.235), "store elite molecules", fontsize=10)
    add_label(ax, (0.80, 0.26), "replay only enters auxiliary branch", fontsize=9.8, color="#8b3f5c")

    # Connection from supervised model to RL policy
    add_arrow(ax, (0.51, 0.60), (0.15, 0.40), color="#4b5b6b", ls="--")
    add_label(ax, (0.27, 0.515), "pretrained / fine-tuned weights", fontsize=10)

    # Formula band
    formula_box = FancyBboxPatch(
        (0.23, 0.035),
        0.56,
        0.09,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.5,
        facecolor="#f7fafc",
        edgecolor="#c8d2dc",
    )
    ax.add_patch(formula_box)
    ax.text(
        0.51,
        0.09,
        "Training objective",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#102030",
    )
    ax.text(
        0.51,
        0.058,
        "L_total = L_GSPO(on-policy rollout, clipped ratio, KL) + lambda_aux * L_aux(elite replay likelihood)",
        ha="center",
        va="center",
        fontsize=11,
        color="#22313f",
    )

    # Footnote
    ax.text(
        0.03,
        0.012,
        "Source modules: data/data_utils.py, model/decoder_only_tfm.py, sample/sample.py,\n"
        "feedback/vina_scores.py, rl/multi_obj_gspo.py, rl/multi_obj_gspo_elite_aux_04_21.py",
        fontsize=9.2,
        color="#607080",
        ha="left",
        va="bottom",
        linespacing=1.25,
    )

    png_path = OUT_DIR / "fig_method_architecture.png"
    pdf_path = OUT_DIR / "fig_method_architecture.pdf"
    svg_path = OUT_DIR / "fig_method_architecture.svg"

    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    manifest = OUT_DIR / "fig_method_architecture_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "Figure: Method architecture overview",
                f"PNG: {png_path}",
                f"PDF: {pdf_path}",
                f"SVG: {svg_path}",
                "",
                "Code sources referenced:",
                f"- {ROOT / 'data' / 'data_utils.py'}",
                f"- {ROOT / 'model' / 'decoder_only_tfm.py'}",
                f"- {ROOT / 'sample' / 'sample.py'}",
                f"- {ROOT / 'feedback' / 'vina_scores.py'}",
                f"- {ROOT / 'rl' / 'multi_obj_gspo.py'}",
                f"- {ROOT / 'rl' / 'multi_obj_gspo_elite_aux_04_21.py'}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {manifest}")


if __name__ == "__main__":
    draw_method_architecture()
