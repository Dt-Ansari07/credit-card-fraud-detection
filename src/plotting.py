"""Shared plotting utilities for the Credit Card Fraud Detection project.

Kept separate from the notebook so every chart uses the same visual style
(colors, fonts, layout) and so the plotting code isn't duplicated across cells.
None of these functions compute metrics — they only visualize values that were
already computed from the model (e.g. in `evaluation.py` or the notebook).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# Semantic, restrained color palette used across every chart in this project.
MODEL_COLORS = {
    "original": "#3B82F6",   # blue  — baseline Decision Tree
    "balanced": "#F97316",   # orange — class_weight='balanced' Decision Tree
}
CLASS_COLORS = {
    "not_fraud": "#64748B",  # neutral slate gray
    "fraud": "#E63946",      # red/orange — draws the eye to the rare class
}


def apply_style():
    """Consistent, professional matplotlib/seaborn config for every chart."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def plot_class_distribution(y, save_path, class_names=("Not Fraud", "Fraud")):
    """Log-scale bar chart so a rare positive class stays visible, with exact
    counts and percentages annotated (nothing is hidden by the log scale)."""
    apply_style()
    counts = y.value_counts().sort_index()
    values = counts.values
    total = values.sum()
    pct = values / total * 100
    colors = [CLASS_COLORS["not_fraud"], CLASS_COLORS["fraud"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(class_names, values, color=colors, width=0.55, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, values.max() * 4)
    ax.set_ylabel("Number of Transactions (log scale)")
    ax.set_title(f"Class Distribution — Severely Imbalanced ({pct[1]:.2f}% Fraud)")

    for bar, v, p in zip(bars, values, pct):
        ax.annotate(
            f"{v:,}\n({p:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def plot_pr_curves(y_test, proba_by_model, save_path,
                    title="Precision–Recall Performance — Fraud Detection",
                    annotation="Class-weighting produces a small improvement in PR-AUC"):
    """Plot the real precision_recall_curve() output for each model (no
    hand-placed points), plus a labeled no-skill baseline at the fraud rate."""
    apply_style()
    baseline = float(np.mean(y_test))
    style_by_model = {
        "Original": {"color": MODEL_COLORS["original"], "marker": "o"},
        "class_weight='balanced'": {"color": MODEL_COLORS["balanced"], "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in proba_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        style = style_by_model.get(name, {})
        step = max(len(recall) // 12, 1)
        ax.plot(
            recall, precision, linewidth=2.2,
            marker=style.get("marker"), markevery=step, markersize=5,
            color=style.get("color"), label=f"{name} (PR-AUC = {ap:.3f})",
        )

    ax.axhline(
        baseline, color="#999999", linestyle=":", linewidth=1.5,
        label=f"No-skill baseline (fraud rate = {baseline:.2%})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, pad=28)
    ax.text(
        0.5, 1.045, annotation, transform=ax.transAxes,
        ha="center", fontsize=9.5, style="italic", color="#555555",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def plot_confusion_matrices(y_test, preds_by_model, metrics_by_model, save_path,
                             title="Confusion Matrix Comparison — Fraud Class"):
    """Side-by-side confusion matrices on a shared color scale, labeled with
    TN/FP/FN/TP, with a compact precision/recall/F1 summary under each."""
    apply_style()
    cms = {name: confusion_matrix(y_test, preds) for name, preds in preds_by_model.items()}
    vmax = max(cm.max() for cm in cms.values())
    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 5.5))
    if len(cms) == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms.items()):
        annot = np.array([[f"{cell_labels[i, j]}\n{cm[i, j]:,}" for j in range(2)] for i in range(2)])
        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=vmax,
            xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"],
            linewidths=1.2, linecolor="white",
            annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax,
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        m = metrics_by_model[name]
        ax.text(
            0.5, -0.22,
            f"Precision {m['Precision (fraud)']:.3f}  ·  Recall {m['Recall (fraud)']:.3f}  ·  F1 {m['F1 (fraud)']:.3f}",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#444444",
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison(comparison_df, save_path,
                           title="Model Performance Comparison — Fraud Class Metrics"):
    """Grouped bar chart of Precision/Recall/F1/ROC-AUC/PR-AUC (accuracy is
    intentionally excluded — it's misleading on data this imbalanced)."""
    apply_style()
    metrics = ["Precision (fraud)", "Recall (fraud)", "F1 (fraud)", "ROC-AUC", "PR-AUC"]
    labels_short = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

    by_model = comparison_df.set_index("Model")
    original_name = [n for n in by_model.index if "Original" in n][0]
    balanced_name = [n for n in by_model.index if "balanced" in n][0]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, by_model.loc[original_name, metrics], width,
                    label="Original", color=MODEL_COLORS["original"])
    bars2 = ax.bar(x + width / 2, by_model.loc[balanced_name, metrics], width,
                    label="class_weight='balanced'", color=MODEL_COLORS["balanced"])

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path)
    return fig
