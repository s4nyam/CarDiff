import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================ CONFIGURATION - ALL CONTROL VARIABLES HERE ============================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "test", "val"]
COLORS = {"train": "#1f77b4", "test": "#d62728", "val": "#2ca02c"}
CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_', '1', '2', '3', '4']

# ============================================ PLOT CUSTOMIZATION VARIABLES (UPDATED) ============================================

# Legend configuration
LEGEND_FONT_SIZE = 15 # Updated
LEGEND_N_COLS = 7 # Updated to 7
LEGEND_BBOX_TO_ANCHOR_Y = 1 # Adjusted as requested to 1.01

# Tick configuration
TICK_FONT_SIZE = 13 # Updated
TICK_ROTATION = 45

# Axis and Title Font Sizes
AXIS_LABEL_FONT_SIZE = 14 # New
AXIS_TITLE_FONT_SIZE = 14 # New
SUPTITLE_FONT_SIZE = 15 # New
SUPTITLE_Y = 1.05 # Adjusted as requested to 1.05

# Combined merged plot figure size (Updated)
COMBINED_MERGED_FIG_WIDTH = 25 # Updated
COMBINED_MERGED_FIG_HEIGHT = 8 # Updated

# Subplot size within combined merged plot (all 4 subplots will have same size)
SUBPLOT_WIDTH = 12
SUBPLOT_HEIGHT = 12

# Individual classwise plot size
INDIVIDUAL_PLOT_WIDTH = 12
INDIVIDUAL_PLOT_HEIGHT = 8

# Tight layout rect parameter
TIGHT_LAYOUT_RECT_TOP = 0.9 # Adjusted as requested to 0.9

# ============================================ CLASSWISE CONFIG ============================================
CLASSWISE_DIR = os.path.join(ROOT_DIR, "test-classwise")
OUTPUT_CLASSWISE_DIR = os.path.join(ROOT_DIR, "test-classwise-plots")

os.makedirs(OUTPUT_CLASSWISE_DIR, exist_ok=True)

def get_available_classes():
    files = os.listdir(CLASSWISE_DIR)
    classes = set()
    for f in files:
        if f.endswith(".csv"):
            parts = f.replace(".csv", "").split("_", 1)
            if len(parts) == 2:
                classes.add(parts[1])
    return sorted(classes)

# ===================================================== HELPER: Auto-configure ticks based on data =====================================================
def auto_configure_ticks(ax, x_data, y_data, num_ticks=8):
    """Automatically set ticks based on data range"""
    # X-axis ticks
    if len(x_data) > 0:
        x_min, x_max = min(x_data), max(x_data)
        x_range = x_max - x_min
        x_ticks = np.linspace(x_min, x_max, min(num_ticks, len(x_data)))
        ax.set_xticks(x_ticks)
        ax.set_xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)
        ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)

    # Y-axis ticks
    if len(y_data) > 0:
        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        y_ticks = np.linspace(y_min, y_max, min(num_ticks, len(np.unique(y_data))))
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    if len(x_data) > 10:
        ax.tick_params(axis="x", rotation=TICK_ROTATION)

# ===================================================== COMBINED + MERGED PLOT (All metrics in one row with multiple classes) (IMPROVED) =====================================================
def plot_combined_classwise_merged_improved():
    """
    Create a 1x4 grid where each subplot shows multiple classes for one metric
    """
    classes = get_available_classes()

    metric_info = [
        ("fid",   "fid",       "FID",                    None),
        ("kid",   "kid_mean",  "KID",                    ("kid_mean", "kid_std")),
        ("is",    "is_mean",   "Inception Score (IS)",   None),
        ("lpips", "lpips_mean","LPIPS",                  None),
    ]

    # Create figure with specified total size (using updated global variables)
    fig, axes = plt.subplots(1, 4, figsize=(COMBINED_MERGED_FIG_WIDTH, COMBINED_MERGED_FIG_HEIGHT))

    # Removed: ax.set_position loop (plt.tight_layout will handle this more efficiently)

    all_handles = []
    all_labels = []

    for ax_idx, (ax, (prefix, value_col, ylabel, std_cols)) in enumerate(zip(axes, metric_info)):

        color_idx = 0
        valid_classes = []
        all_epochs = []
        all_values = []

        for cls in classes:
            filename = f"{prefix}_{cls}.csv"
            csv_path = os.path.join(CLASSWISE_DIR, filename)

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            if "epoch" not in df.columns:
                continue

            df = df.sort_values("epoch")
            epochs = df["epoch"].values

            if prefix == "fid":
                values = df["fid"].values if "fid" in df.columns else None
            elif prefix == "kid":
                values = df["kid_mean"].values if "kid_mean" in df.columns else None
            elif prefix == "is":
                values = df["is_mean"].values if "is_mean" in df.columns else None
            elif prefix == "lpips":
                values = df["lpips_mean"].values if "lpips_mean" in df.columns else None

            if values is None:
                continue

            valid_classes.append(cls)
            all_epochs.extend(epochs)
            all_values.extend(values)

            color = CLASS_COLORS[color_idx % len(CLASS_COLORS)]
            marker = MARKERS[color_idx % len(MARKERS)]

            line = ax.plot(epochs, values, linewidth=2.0, marker=marker,
                           markersize=5, color=color, label=f"{cls}", # Markersize reduced for improved visibility with more data
                           markeredgecolor='black', markeredgewidth=0.5)[0]

            # Only add std for KID
            if std_cols is not None and prefix == "kid":
                mean_col, std_col = std_cols
                if mean_col in df.columns and std_col in df.columns:
                    ax.fill_between(
                        epochs,
                        df[mean_col] - df[std_col],
                        df[mean_col] + df[std_col],
                        alpha=0.15,
                        color=color,
                    )

            if ax_idx == 0:
                all_handles.append(line)
                all_labels.append(cls)

            color_idx += 1

        if not valid_classes:
            ax.text(0.5, 0.5, f"No data for {ylabel}",
                   ha='center', va='center', transform=ax.transAxes, fontsize=AXIS_LABEL_FONT_SIZE)
            ax.set_title(f"{ylabel}", fontsize=AXIS_TITLE_FONT_SIZE)
            continue

        ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold") # Updated font size
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold") # Updated font size
        ax.set_title(f"{ylabel} - All Classes", fontsize=AXIS_TITLE_FONT_SIZE, fontweight="bold") # Updated font size
        ax.grid(True, linestyle="--", alpha=0.3)

        # Auto-configure ticks based on all data for this subplot
        # auto_configure_ticks uses the global TICK_FONT_SIZE
        if all_epochs and all_values:
            auto_configure_ticks(ax, all_epochs, all_values)

    # Add a single legend with configurable font size, columns, and spacing
    if all_handles:
        fig.legend(all_handles, all_labels, loc="upper center",
                   bbox_to_anchor=(0.5, LEGEND_BBOX_TO_ANCHOR_Y), # Updated bbox_to_anchor y-coordinate
                   ncol=min(LEGEND_N_COLS, len(all_labels)), # Updated ncol
                   fontsize=LEGEND_FONT_SIZE, title="Classes", title_fontsize=LEGEND_FONT_SIZE) # Updated font sizes

    plt.suptitle("Class-wise Metrics Comparison Across All Classes",
                 fontsize=SUPTITLE_FONT_SIZE, fontweight="bold", y=SUPTITLE_Y) # Updated font size and y-position

    plt.tight_layout(rect=[0, 0, 1, TIGHT_LAYOUT_RECT_TOP]) # Updated with rect parameter

    # Modified save_path as requested
    save_path = os.path.join(OUTPUT_CLASSWISE_DIR, "Combined_Merged_All_Classes.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved {os.path.basename(save_path)}")


# ===================================================== MAIN =====================================================
if __name__ == "__main__":

    print("=" * 50)
    print("COMBINED METRICS PLOTTING (FID | KID | IS | LPIPS)")
    print("=" * 50)

    # Call the new improved function directly
    plot_combined_classwise_merged_improved()