"""Reusable evaluation helpers, mirrored from the analysis notebook.

Kept separate from the notebook so the scoring logic can be reused
(e.g. in a future script or API) without re-running the full analysis.
"""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fraud_metrics(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (fraud)": precision_score(y_true, y_pred, pos_label=1),
        "Recall (fraud)": recall_score(y_true, y_pred, pos_label=1),
        "F1 (fraud)": f1_score(y_true, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }
