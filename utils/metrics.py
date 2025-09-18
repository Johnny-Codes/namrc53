from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
import numpy as np


def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, loss=0.0):
    """Calculate comprehensive metrics"""
    # Return default metrics if no predictions
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "loss": loss,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "precision_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_micro": 0.0,
            "recall_weighted": 0.0,
            "cohen_kappa": 0.0,
            "matthews_corrcoef": 0.0,
        }

    try:
        metrics = {
            "loss": loss,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        }
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        # Return default metrics if calculation fails
        metrics = {
            "loss": loss,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "precision_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_micro": 0.0,
            "recall_weighted": 0.0,
            "cohen_kappa": 0.0,
            "matthews_corrcoef": 0.0,
        }

    return metrics


def calculate_per_class_metrics(y_true, y_pred, class_names):
    """Calculate per-class precision, recall, F1"""
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": precision[i] if i < len(precision) else 0.0,
                "recall": recall[i] if i < len(recall) else 0.0,
                "f1_score": f1[i] if i < len(f1) else 0.0,
                "support": support[i] if i < len(support) else 0,
            }
    except Exception as e:
        print(f"Warning: Error calculating per-class metrics: {e}")
        per_class_metrics = {}
        for class_name in class_names:
            per_class_metrics[class_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "support": 0,
            }

    return per_class_metrics
