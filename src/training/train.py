"""Fine-tune DistilBERT on safety classification data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


LABELS = ["safe", "toxic", "jailbreak"]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _load_csv_as_dataset(path: Path) -> Dataset:
    """Read a CSV with columns ``text`` and ``label``."""
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df.reset_index(drop=True))


def _tokenize_fn(
    examples: dict, tokenizer: AutoTokenizer, max_length: int = 512
) -> dict:
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=max_length
    )


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_precision": precision_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train safety classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/safety_classifier"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Compute class weights for imbalanced data",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding layer during training",
    )
    args = parser.parse_args()

    # -- load data -----------------------------------------------------------
    train_ds = _load_csv_as_dataset(args.data_dir / "train.csv")
    val_ds = _load_csv_as_dataset(args.data_dir / "val.csv")
    test_ds = _load_csv_as_dataset(args.data_dir / "test.csv")

    # -- tokenizer & model ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
    )

    if args.freeze_embeddings:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False

    # -- tokenize ------------------------------------------------------------
    train_ds = train_ds.map(
        lambda x: _tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
    )
    val_ds = val_ds.map(
        lambda x: _tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
    )
    test_ds = test_ds.map(
        lambda x: _tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
    )

    # -- class weights -------------------------------------------------------
    if args.class_weights:
        label_counts = train_ds.to_pandas()["label"].value_counts()
        total = len(train_ds)
        # Inverse frequency, normalized
        weights = {
            int(k): round(total / (len(label_counts) * v), 4)
            for k, v in label_counts.items()
        }
        print(f"Class weights: {weights}")
        # TODO: apply weights via custom loss if needed
        # Trainer doesn't natively support class weights for sequence classification
        # Can be done with a custom Trainer subclass

    # -- training args -------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    # -- trainer -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # -- save ----------------------------------------------------------------
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # -- final eval ----------------------------------------------------------
    test_results = trainer.evaluate(test_ds)
    print("\n=== Test Set Results ===")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()