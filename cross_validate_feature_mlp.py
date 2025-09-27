"""Run k-fold cross-validation for the engineered feature MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from train_feature_network import (
    build_mlp,
    compute_class_weight,
    compute_metrics,
    load_feature_dataset,
    normalize_features,
)
from train_models import compile_model, log_gpu_info


def stratified_kfold(indices: np.ndarray, labels: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if k < 2:
        raise ValueError("k must be at least 2 for k-fold cross-validation")

    rng = np.random.default_rng(seed)
    fold_bins: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(labels):
        cls_indices = indices[labels == cls]
        shuffled = cls_indices.copy()
        rng.shuffle(shuffled)
        splits = np.array_split(shuffled, k)
        for fold_id, split in enumerate(splits):
            fold_bins[fold_id].extend(int(idx) for idx in split)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(k):
        test_idx = np.array(fold_bins[fold_idx], dtype=np.int32)
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        folds.append((train_idx, test_idx))
    return folds


def stratified_train_val_split(train_idx: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    remaining_indices: List[int] = []
    for cls in np.unique(labels[train_idx]):
        cls_mask = labels[train_idx] == cls
        cls_indices = train_idx[cls_mask]
        cls_indices = cls_indices.copy()
        rng.shuffle(cls_indices)
        n_val = int(round(cls_indices.size * val_ratio))
        if 0 < cls_indices.size <= 2:
            n_val = 1 if cls_indices.size > 1 else 0
        n_val = max(0, min(n_val, cls_indices.size))
        val_indices.extend(cls_indices[:n_val])
        remaining_indices.extend(cls_indices[n_val:])

    if not val_indices:
        raise ValueError("Validation set is empty; consider increasing val_ratio or decreasing k")

    rng.shuffle(val_indices)
    rng.shuffle(remaining_indices)
    return np.asarray(remaining_indices, dtype=np.int32), np.asarray(val_indices, dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/cw_extended_features.csv"), help="Path to the feature CSV dataset.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of the training split reserved for validation.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs).")
    parser.add_argument("--plateau-patience", type=int, default=6, help="ReduceLROnPlateau patience.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for hidden layers.")
    parser.add_argument("--l2-weight", type=float, default=1e-4, help="L2 regularisation strength.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256, 128], help="Hidden layer widths (best model configuration).")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Binary cross-entropy label smoothing.")
    parser.add_argument("--threshold", type=float, default=0.16, help="Decision threshold for sigmoid outputs.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau.")
    parser.add_argument("--output", type=Path, help="Optional path to save the cross-validation report (JSON).")
    parser.add_argument("--quiet", action="store_true", help="Suppress fold-level training logs.")
    return parser.parse_args()


def build_callbacks(args: argparse.Namespace) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        ),
    ]
    return callbacks


def evaluate_split(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    metrics = compute_metrics(preds, labels)
    metrics["prob_mean"] = float(probs.mean())
    metrics["prob_std"] = float(probs.std())
    metrics["count"] = int(labels.size)
    return metrics


def run_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object], np.ndarray, np.ndarray]:
    tf.keras.utils.set_random_seed(args.seed + fold_idx)

    train_feat = features[train_idx]
    val_feat = features[val_idx]
    test_feat = features[test_idx]
    train_feat, val_feat, test_feat = normalize_features(train_feat, val_feat, test_feat)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    model = build_mlp(train_feat.shape[1], args.hidden_sizes, args.dropout, args.l2_weight)
    compile_model(model, lr=args.learning_rate, label_smoothing=args.label_smoothing)

    class_weight = compute_class_weight(y_train)
    callbacks = build_callbacks(args)
    verbose = 0 if args.quiet else 1
    history = model.fit(
        train_feat,
        y_train,
        validation_data=(val_feat, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )

    history_length = len(history.history.get("loss", []))

    train_probs = model.predict(train_feat, verbose=0).flatten()
    val_probs = model.predict(val_feat, verbose=0).flatten()
    test_probs = model.predict(test_feat, verbose=0).flatten()

    fold_metrics = {
        "train": evaluate_split(train_probs, y_train, args.threshold),
        "val": evaluate_split(val_probs, y_val, args.threshold),
        "test": evaluate_split(test_probs, y_test, args.threshold),
    }
    aux: Dict[str, object] = {
        "epochs_ran": history_length,
        "class_weight": {str(k): float(v) for k, v in class_weight.items()},
    }
    return fold_metrics, aux, test_probs, y_test


def main() -> None:
    args = parse_args()
    log_gpu_info()

    features, labels, sample_indices, label_names, feature_columns = load_feature_dataset(args.dataset)
    indices = np.arange(labels.shape[0], dtype=np.int32)
    folds = stratified_kfold(indices, labels, args.k, args.seed)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_test_probs: List[np.ndarray] = []
    all_test_labels: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_seed = args.seed + fold_idx
        train_split, val_split = stratified_train_val_split(train_idx, labels, args.val_ratio, fold_seed)
        metrics, aux, test_probs, test_labels = run_fold(
            fold_idx,
            train_split,
            val_split,
            test_idx,
            features,
            labels,
            args,
        )
        fold_name = f"fold_{fold_idx}"
        all_results[fold_name] = metrics
        all_results[fold_name]["meta"] = {
            "train_size": int(train_split.size),
            "val_size": int(val_split.size),
            "test_size": int(test_idx.size),
            "epochs_ran": aux["epochs_ran"],
            "class_weight": aux["class_weight"],
        }
        all_test_probs.append(test_probs)
        all_test_labels.append(test_labels)

    concatenated_probs = np.concatenate(all_test_probs, axis=0)
    concatenated_labels = np.concatenate(all_test_labels, axis=0)
    overall_metrics = evaluate_split(concatenated_probs, concatenated_labels, args.threshold)

    summary = {
        "dataset": str(args.dataset),
        "k": args.k,
        "threshold": args.threshold,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "l2_weight": args.l2_weight,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "folds": all_results,
        "overall": overall_metrics,
        "label_names": label_names,
        "feature_columns": feature_columns,
    }

    payload = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
