import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "test", "val"]
COLORS = {"train": "#1f77b4", "test": "#d62728", "val": "#2ca02c"}


def configure_axis(ax):
    """
    Clean axis formatting without overlapping ticks
    """
    ax.grid(True, linestyle="--", alpha=0.3)

    # Let matplotlib decide smart ticks
    ax.tick_params(axis='both', labelsize=12)

    # Avoid cutting off labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')


def plot_merged():

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    metric_info = [
        ("fid.csv",   "fid",       "FID",                    None),
        ("kid.csv",   "kid_mean",  "KID",                    ("kid_mean", "kid_std")),
        ("is.csv",    "is_mean",   "Inception Score (IS)",   ("is_mean", "is_std")),
        ("lpips.csv", "lpips_mean","LPIPS",                  None),
    ]

    for ax, (filename, value_col, ylabel, std_cols) in zip(axes, metric_info):

        for split in SPLITS:
            csv_path = os.path.join(ROOT_DIR, split, filename)
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df = df.sort_values("epoch")

            # Handle FID capitalization
            if filename == "fid.csv":
                value_col = "FID" if "FID" in df.columns else "fid"

            epochs = df["epoch"]
            values = df[value_col]

            ax.plot(
                epochs,
                values,
                linewidth=2.5,
                label=split.capitalize(),
                color=COLORS[split],
            )

            if std_cols is not None:
                mean_col, std_col = std_cols
                if mean_col in df.columns and std_col in df.columns:
                    ax.fill_between(
                        epochs,
                        df[mean_col] - df[std_col],
                        df[mean_col] + df[std_col],
                        alpha=0.2,
                        color=COLORS[split],
                    )

        ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_title(ylabel, fontsize=15, fontweight="bold")

        configure_axis(ax)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, fontsize=12)
    fig.suptitle(
        "Metrics comparisons for train-test-val sets on synthesized images",
        fontsize=18,
        fontweight="bold",
        y=0.98
        )

    # Proper spacing so ticks are NOT cut off
    plt.subplots_adjust(top=0.75, bottom=0.2, wspace=0.3)

    save_path = os.path.join(ROOT_DIR, "merged_metrics_row.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ Saved merged_metrics_row.pdf")


if __name__ == "__main__":
    plot_merged()