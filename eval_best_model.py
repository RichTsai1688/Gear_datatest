"""Evaluate the saved feature MLP checkpoint against the stratified splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_feature_network import (
    build_mlp,
    compute_metrics,
    load_feature_dataset,
    load_index_splits,
    normalize_features,
)
from train_models import compile_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/cw_extended_features.csv"),
        help="Feature CSV used for evaluation.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("datasets/feature_splits.json"),
        help="JSON file describing the stratified splits.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("checkpoints/feature_mlp_best.weights.h5"),
        help="Checkpoint weights to load.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.16,
        help="Decision threshold applied to sigmoid outputs.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer sizes (must match the trained model).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used by the MLP (match training configuration).",
    )
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=1e-4,
        help="L2 regularisation coefficient (match training configuration).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for optimiser initialisation (affects compile only).",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    features, labels, _, _, _ = load_feature_dataset(args.dataset)
    splits = load_index_splits(args.splits)

    train_feat = features[splits["train"]]
    val_feat = features[splits["val"]]
    test_feat = features[splits["test"]]
    train_feat, val_feat, test_feat = normalize_features(train_feat, val_feat, test_feat)

    y_train = labels[splits["train"]]
    y_val = labels[splits["val"]]
    y_test = labels[splits["test"]]

    input_dim = train_feat.shape[1]
    model = build_mlp(input_dim, args.hidden_sizes, args.dropout, args.l2_weight)
    compile_model(model, lr=args.learning_rate, label_smoothing=0.0)
    model.load_weights(args.weights)

    results: dict[str, dict[str, float]] = {}
    for name, (feat, lab) in {
        "train": (train_feat, y_train),
        "val": (val_feat, y_val),
        "test": (test_feat, y_test),
    }.items():
        probs = model.predict(feat, verbose=0).flatten()
        preds = (probs >= args.threshold).astype(np.int32)
        metrics = compute_metrics(preds, lab)
        metrics["prob_mean"] = float(probs.mean())
        metrics["prob_std"] = float(probs.std())
        metrics["count"] = int(lab.size)
        results[name] = metrics

    all_feat = np.concatenate([train_feat, val_feat, test_feat], axis=0)
    all_labels = np.concatenate([y_train, y_val, y_test], axis=0)
    all_probs = model.predict(all_feat, verbose=0).flatten()
    all_preds = (all_probs >= args.threshold).astype(np.int32)
    overall = compute_metrics(all_preds, all_labels)
    overall["prob_mean"] = float(all_probs.mean())
    overall["prob_std"] = float(all_probs.std())
    overall["count"] = int(all_labels.size)
    results["overall"] = overall

    return results


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
