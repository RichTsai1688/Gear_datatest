import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from train_models import compile_model, log_gpu_info, stratified_split, train_and_evaluate

DEFAULT_DATASET = Path("datasets/cw_top_features.csv")
DEFAULT_SPLITS = Path("datasets/feature_splits.json")


def compute_class_weight(labels: np.ndarray) -> Dict[int, float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    n_classes = unique.shape[0]
    weights = {}
    for cls, count in zip(unique, counts):
        if count == 0:
            continue
        weights[int(cls)] = total / (n_classes * float(count))
    return weights


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = preds.astype(np.int32)
    labels = labels.astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = labels.size
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    balanced_accuracy = (recall + specificity) / 2 if (tp + fn) and (tn + fp) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def find_best_threshold(scores: np.ndarray, labels: np.ndarray, metric: str) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metric = -1.0
    best_metrics: Dict[str, float] = {}

    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (scores >= threshold).astype(np.int32)
        metrics = compute_metrics(preds, labels)
        current = metrics[metric]
        if current > best_metric:
            best_metric = current
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


META_COLUMNS = {"sample_index", "source", "label_idx", "label"}


def load_feature_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[str]]:
    """Load the engineered feature dataset from CSV."""
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("Feature dataset CSV is empty or missing a header row")

        feature_columns = [name for name in reader.fieldnames if name not in META_COLUMNS]
        required_columns = set(feature_columns) | {"sample_index"}
        missing_columns = [col for col in required_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Feature dataset missing columns: {missing_columns}")

        features: List[List[float]] = []
        labels: List[int] = []
        sample_indices: List[int] = []
        idx_to_label: Dict[int, str] = {}
        label_to_idx: Dict[str, int] = {}

        for row in reader:
            features.append([float(row[name]) for name in feature_columns])
            sample_indices.append(int(row["sample_index"]))

            label_name = row.get("label", "")
            label_idx_str = row.get("label_idx", "")

            if label_idx_str:
                label_idx = int(label_idx_str)
                if label_name:
                    idx_to_label.setdefault(label_idx, label_name)
                elif label_idx not in idx_to_label:
                    idx_to_label[label_idx] = f"class_{label_idx}"
            else:
                if not label_name:
                    raise ValueError("Feature dataset is missing both label and label_idx columns")
                label_idx = label_to_idx.setdefault(label_name, len(label_to_idx))
                idx_to_label.setdefault(label_idx, label_name)

            labels.append(label_idx)

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)

    if idx_to_label:
        label_names = [idx_to_label[idx] for idx in sorted(idx_to_label)]
    else:
        label_names = [f"class_{idx}" for idx in sorted(set(labels))]

    return feature_array, label_array, sample_indices, label_names, feature_columns


def save_index_splits(
    splits: Dict[str, np.ndarray],
    path: Path,
    sample_indices: Sequence[int],
    labels: np.ndarray,
) -> None:
    payload = {}
    for key, value in splits.items():
        payload[key] = [
            {
                "row_index": int(idx),
                "sample_index": int(sample_indices[idx]),
                "label_idx": int(labels[idx]),
            }
            for idx in value
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_index_splits(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    result: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if value and isinstance(value[0], dict):
            result[key] = np.asarray([item["row_index"] for item in value], dtype=np.int32)
        else:
            result[key] = np.asarray([int(idx) for idx in value], dtype=np.int32)
    return result


def normalize_features(train_feat: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = train_feat.mean(axis=0, keepdims=True)
    std = train_feat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    normalized = [(train_feat - mean) / std]
    for feat in others:
        normalized.append((feat - mean) / std)
    return tuple(arr.astype(np.float32) for arr in normalized)


def build_mlp(input_dim: int, hidden_layers: Sequence[int], dropout: float, l2_weight: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2_weight)
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=l2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)(x)
    return tf.keras.Model(inputs, outputs, name="feature_mlp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural network on engineered gear features.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the feature CSV dataset.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="Path to persist the train/val/test splits.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64, 32], help="Hidden layer sizes for the MLP.")
    parser.add_argument("--auto-class-weight", action="store_true", help="Balance classes using inverse-frequency weights during training.")
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy", "f1"], default="balanced_accuracy",
                        help="Validation metric to optimize when choosing the decision threshold.")
    parser.add_argument("--no-threshold-tuning", action="store_true", help="Skip post-training threshold search.")
    args = parser.parse_args()

    log_gpu_info()

    features, labels, sample_indices, label_names, feature_columns = load_feature_dataset(args.dataset)
    print(f"Loaded feature dataset with {features.shape[0]} samples and {features.shape[1]} features")
    print(f"Feature columns: {feature_columns}")

    if args.splits.exists():
        splits = load_index_splits(args.splits)
        print(f"Loaded feature splits from {args.splits}")
    else:
        splits = stratified_split(labels, args.train_ratio, args.val_ratio, args.seed)
        save_index_splits(splits, args.splits, sample_indices, labels)
        print(f"Saved new feature splits to {args.splits}")

    train_feat = features[splits["train"]]
    val_feat = features[splits["val"]]
    test_feat = features[splits["test"]]

    train_feat, val_feat, test_feat = normalize_features(train_feat, val_feat, test_feat)

    y_train_idx = labels[splits["train"]]
    y_val_idx = labels[splits["val"]]
    y_test_idx = labels[splits["test"]]

    y_train = y_train_idx.astype(np.float32)
    y_val = y_val_idx.astype(np.float32)
    y_test = y_test_idx.astype(np.float32)

    data_dict = {
        "train": train_feat,
        "val": val_feat,
        "test": test_feat,
    }
    label_dict = {
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }

    input_dim = train_feat.shape[1]
    model = build_mlp(input_dim, args.hidden_sizes, args.dropout, args.l2_weight)
    compile_model(model, args.learning_rate, args.label_smoothing)
    model.summary()

    class_weight = compute_class_weight(y_train_idx) if args.auto_class_weight else None

    train_and_evaluate(
        model=model,
        name="feature_mlp",
        data=data_dict,
        labels=label_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        plateau_patience=args.plateau_patience,
        class_weight=class_weight,
    )

    report_path = Path("reports") / "feature_mlp_report.json"
    threshold_payload = None

    if not args.no_threshold_tuning:
        val_scores = model.predict(val_feat, verbose=0).flatten()
        best_threshold, val_metrics = find_best_threshold(val_scores, y_val_idx, args.threshold_metric)
        test_scores = model.predict(test_feat, verbose=0).flatten()
        test_preds = (test_scores >= best_threshold).astype(np.int32)
        test_metrics = compute_metrics(test_preds, y_test_idx)
        threshold_payload = {
            "metric": args.threshold_metric,
            "threshold": best_threshold,
            "validation": val_metrics,
            "test": test_metrics,
        }
        print(f"Best threshold ({args.threshold_metric}) on val: {best_threshold:.3f}")
        print(f"Validation metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")

    if report_path.exists():
        payload = json.loads(report_path.read_text())
    else:
        payload = {}

    if label_names:
        payload["label_names"] = label_names

    if class_weight:
        payload["class_weight"] = {str(k): float(v) for k, v in class_weight.items()}

    if threshold_payload:
        payload["threshold_tuning"] = threshold_payload

    if payload:
        report_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
