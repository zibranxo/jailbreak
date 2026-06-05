"""Evaluate a trained safety classifier on a test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.detectors.ml_classifier import MLClassifier, LABELS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -- load model ----------------------------------------------------------
    clf = MLClassifier(str(args.model_path))

    # -- load data -----------------------------------------------------------
    df = pd.read_csv(args.test_data)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()

    # -- predict -------------------------------------------------------------
    results = clf.predict_batch(texts, batch_size=args.batch_size)
    y_pred = [LABELS.index(r["label"]) for r in results]
    confidences = [r["confidence"] for r in results]

    # -- metrics -------------------------------------------------------------
    report = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "per_class": {},
    }

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    for label in LABELS:
        cls_metrics = report[label]
        summary["per_class"][label] = {
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1-score": cls_metrics["f1-score"],
            "support": cls_metrics["support"],
        }

    # -- calibration check ---------------------------------------------------
    # Bin predictions by confidence and compare to actual accuracy
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    calibration = []
    for b in bins:
        mask = [c >= b for c in confidences]
        if any(mask):
            correct = sum(
                1 for yt, yp, m in zip(y_true, y_pred, mask) if m and yt == yp
            )
            total = sum(mask)
            calibration.append({
                "confidence_threshold": b,
                "accuracy_in_bin": correct / total,
                "count": total,
            })
    summary["calibration"] = calibration

    # -- save ---------------------------------------------------------------
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # -- confusion matrix plot -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=LABELS
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
