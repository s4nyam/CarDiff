import pandas as pd
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_aug")

MODELS = ["Unet", "UnetPlusPlus", "FPN", "Linknet", "MAnet", "PAN", "PSPNet"]
AUG_LEVELS = ["aug1", "aug2", "aug3", "aug4", "aug5"]
CLASSES = ["SC", "MC", "DC", "MC_SC", "MC_DC", "SC_MC_DC"]

MODEL_DISPLAY = {
    "Unet": "U-Net",
    "UnetPlusPlus": "U-Net++",
    "FPN": "FPN",
    "Linknet": "LinkNet",
    "MAnet": "MA-Net",
    "PAN": "PAN",
    "PSPNet": "PSPNet",
}

AUG_DISPLAY = {
    "aug1": "Aug1",
    "aug2": "Aug2",
    "aug3": "Aug3",
    "aug4": "Aug4",
    "aug5": "Aug5",
}


def load_best_epoch(model, aug, metric="dice"):
    """Load the best epoch based on mean dice/iou across all classes."""
    fname = f"{model}_{aug}_test_detailed_metrics.csv"
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        return None

    df = pd.read_csv(fpath)
    metric_cols = [f"{c}_{metric}" for c in CLASSES]
    df["mean_metric"] = df[metric_cols].mean(axis=1)
    best_idx = df["mean_metric"].idxmax()
    best_row = df.loc[best_idx]

    result = {}
    for c in CLASSES:
        result[c] = best_row[f"{c}_{metric}"]
    result["mean"] = best_row["mean_metric"]
    result["epoch"] = int(best_row["epoch"])
    return result


def fmt(val, is_best=False):
    """Format a value, marking best with ** for bold."""
    s = f"{val:.3f}"
    return f"**{s}**" if is_best else s


def print_paper_table(metric="dice", model_groups=None):
    """
    Paper-style table (like the reference image):
      Rows: Class (SC...SC_MC_DC) + Average
      Columns: Grouped by Model → sub-columns for each Aug level
    Split across model groups so the table fits on a page.

    Prints in a clean fixed-width format ready for paper conversion.
    """
    if model_groups is None:
        # Split 7 models into two groups: 4 + 3
        model_groups = [
            ["Unet", "UnetPlusPlus", "FPN", "Linknet"],
            ["MAnet", "PAN", "PSPNet"],
        ]

    metric_label = "Dice" if metric == "dice" else "IoU"

    # Load ALL results once
    all_results = {}
    for model in MODELS:
        for aug in AUG_LEVELS:
            all_results[(model, aug)] = load_best_epoch(model, aug, metric)

    # Find global best per class (across ALL models, not just current group)
    global_best = {}
    for cls in CLASSES + ["mean"]:
        best_val = -1
        for model in MODELS:
            for aug in AUG_LEVELS:
                r = all_results[(model, aug)]
                if r is not None and r[cls] > best_val:
                    best_val = r[cls]
        global_best[cls] = best_val

    for g_idx, group in enumerate(model_groups):
        n_models = len(group)
        col_w = 7  # width per value column
        class_w = 10  # width for class name column

        print()
        print(f"{'=' * 100}")
        if len(model_groups) > 1:
            print(f"TABLE {g_idx+1}: Per-Class {metric_label} Scores (Best Epoch)")
        else:
            print(f"TABLE: Per-Class {metric_label} Scores (Best Epoch)")
        print(f"'Aug1'–'Aug5' = augmentation configurations. Bold (**) = best across ALL models for that class.")
        print(f"{'=' * 100}")

        # ---- Model header row ----
        line = " " * class_w
        for model in group:
            model_name = MODEL_DISPLAY[model]
            span = len(AUG_LEVELS) * col_w
            line += f"{model_name:^{span}}"
            line += " | "
        print(line.rstrip(" |"))

        # ---- Aug sub-header row ----
        line = f"{'Class':<{class_w}}"
        for model in group:
            for aug in AUG_LEVELS:
                line += f"{AUG_DISPLAY[aug]:>{col_w}}"
            line += " | "
        print(line.rstrip(" |"))

        # ---- Separator ----
        total_w = class_w + n_models * (len(AUG_LEVELS) * col_w + 3)
        print("-" * total_w)

        # ---- Data rows ----
        for cls_key in CLASSES + ["Average"]:
            cls_data = cls_key if cls_key != "Average" else "mean"
            line = f"{cls_key:<{class_w}}"
            for model in group:
                for aug in AUG_LEVELS:
                    r = all_results[(model, aug)]
                    if r is not None:
                        val = r[cls_data]
                        is_best = abs(val - global_best[cls_data]) < 1e-9
                        s = fmt(val, is_best)
                        line += f"{s:>{col_w}}"
                    else:
                        line += f"{'NA':>{col_w}}"
                line += " | "
            print(line.rstrip(" |"))

        print("-" * total_w)
        print()


def print_summary_table(metric="dice"):
    """
    Compact summary: Models (rows) × Aug levels (columns) → mean metric.
    Shows which augmentation is best per model at a glance.
    """
    metric_label = "Dice" if metric == "dice" else "IoU"
    col_w = 8
    model_w = 10

    print()
    print(f"{'=' * 60}")
    print(f"SUMMARY: Mean {metric_label} (Best Epoch) — Models × Augmentations")
    print(f"{'=' * 60}")

    # Header
    line = f"{'Model':<{model_w}}"
    for aug in AUG_LEVELS:
        line += f"{AUG_DISPLAY[aug]:>{col_w}}"
    print(line)
    print("-" * (model_w + len(AUG_LEVELS) * col_w))

    for model in MODELS:
        # Find best aug for this model
        vals = {}
        for aug in AUG_LEVELS:
            r = load_best_epoch(model, aug, metric)
            vals[aug] = r["mean"] if r else None

        best_val = max((v for v in vals.values() if v is not None), default=None)

        line = f"{MODEL_DISPLAY[model]:<{model_w}}"
        for aug in AUG_LEVELS:
            v = vals[aug]
            if v is not None:
                is_best = best_val is not None and abs(v - best_val) < 1e-9
                line += f"{fmt(v, is_best):>{col_w}}"
            else:
                line += f"{'NA':>{col_w}}"
        print(line)

    print("-" * (model_w + len(AUG_LEVELS) * col_w))
    print("** = best augmentation for that model")
    print()


def print_best_aug_per_model(metric="dice"):
    """
    Condensed table: For each model, show only the best augmentation's per-class scores.
    Single clean table with 7 model columns.
    """
    metric_label = "Dice" if metric == "dice" else "IoU"
    col_w = 9
    class_w = 10

    print()
    print(f"{'=' * 80}")
    print(f"TABLE: Per-Class {metric_label} — Best Augmentation per Model (Best Epoch)")
    print(f"{'=' * 80}")

    # Find best aug per model
    best_results = {}
    best_aug = {}
    for model in MODELS:
        best_mean = -1
        for aug in AUG_LEVELS:
            r = load_best_epoch(model, aug, metric)
            if r is not None and r["mean"] > best_mean:
                best_mean = r["mean"]
                best_results[model] = r
                best_aug[model] = aug

    # Find global best per class
    global_best = {}
    for cls in CLASSES + ["mean"]:
        best_val = max(
            (best_results[m][cls] for m in MODELS if m in best_results),
            default=-1,
        )
        global_best[cls] = best_val

    # Header
    line = f"{'Class':<{class_w}}"
    for model in MODELS:
        line += f"{MODEL_DISPLAY[model]:>{col_w}}"
    print(line)

    line = f"{'(best aug)':<{class_w}}"
    for model in MODELS:
        line += f"{'(' + AUG_DISPLAY[best_aug.get(model, 'aug1')] + ')':>{col_w}}"
    print(line)
    print("-" * (class_w + len(MODELS) * col_w))

    for cls_key in CLASSES + ["Average"]:
        cls_data = cls_key if cls_key != "Average" else "mean"
        line = f"{cls_key:<{class_w}}"
        for model in MODELS:
            if model in best_results:
                val = best_results[model][cls_data]
                is_best = abs(val - global_best[cls_data]) < 1e-9
                line += f"{fmt(val, is_best):>{col_w}}"
            else:
                line += f"{'NA':>{col_w}}"
        print(line)

    print("-" * (class_w + len(MODELS) * col_w))
    print("** = best across all models for that class")
    print()


if __name__ == "__main__":
    import sys

    output_path = os.path.join(os.path.dirname(__file__), "results_table.txt")
    with open(output_path, "w") as f:
        sys.stdout = f

        # ---- TABLE 1: Summary of mean Dice across models × aug levels ----
        print_summary_table("dice")

        # ---- TABLE 2 & 3: Full per-class Dice, split into two model groups ----
        print_paper_table("dice")

        # ---- TABLE 4: Condensed — best aug only, per-class Dice ----
        print_best_aug_per_model("dice")

        # ---- TABLE 5: Summary of mean IoU across models × aug levels ----
        print_summary_table("iou")

        # ---- TABLE 6 & 7: Full per-class IoU, split into two model groups ----
        print_paper_table("iou")

        # ---- TABLE 8: Condensed — best aug only, per-class IoU ----
        print_best_aug_per_model("iou")

        sys.stdout = sys.__stdout__

    print(f"Output saved to {output_path}")
