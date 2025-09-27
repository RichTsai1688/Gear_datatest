import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from train_models import (
    RAW_DATASETS,
    SPLIT_PATH,
    CHECKPOINT_DIR,
    REPORT_DIR,
    add_channel_dim,
    compute_fft_features,
    compile_model,
    DatasetBundle,
    load_combined_dataset,
    load_splits,
    log_gpu_info,
    normalize_features,
    stratified_split,
    save_splits,
    build_cnn_model,
    build_lstm_model,
)

DEFAULT_MODELS = {
    'cnn': CHECKPOINT_DIR / 'cnn_best.weights.h5',
    'lstm': CHECKPOINT_DIR / 'lstm_best.weights.h5',
}


def ensure_splits(bundle_labels: np.ndarray,
                  bundle: DatasetBundle,
                  train_ratio: float,
                  val_ratio: float,
                  seed: int) -> Dict[str, np.ndarray]:
    if SPLIT_PATH.exists():
        print(f'Loaded existing splits from {SPLIT_PATH}')
        return load_splits(SPLIT_PATH)
    print('Splits not found; creating new stratified splits to match training assumptions.')
    splits = stratified_split(bundle_labels, train_ratio, val_ratio, seed)
    save_splits(splits, bundle, SPLIT_PATH)
    return splits


def build_model(name: str, input_shape: Tuple[int, int], l2_weight: float, dropout: float) -> tf.keras.Model:
    if name == 'cnn':
        return build_cnn_model(input_shape, l2_weight, dropout)
    if name == 'lstm':
        return build_lstm_model(input_shape, l2_weight, dropout)
    raise ValueError(f'Unknown model name: {name}')


def evaluate_model(name: str,
                   weights_path: Path,
                   features: Dict[str, np.ndarray],
                   labels: Dict[str, np.ndarray],
                   l2_weight: float,
                   dropout: float,
                   learning_rate: float,
                   label_smoothing: float) -> Dict[str, object]:
    input_shape = features['train'].shape[1:]
    model = build_model(name, input_shape, l2_weight, dropout)
    compile_model(model, learning_rate, label_smoothing)

    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file not found for {name}: {weights_path}')
    model.load_weights(weights_path)
    print(f'Loaded weights for {name} from {weights_path}')

    eval_result = model.evaluate(features['test'], labels['test'], verbose=0, return_dict=True)
    preds = model.predict(features['test'], verbose=0).flatten()
    pred_labels = (preds >= 0.5).astype(np.int32)
    target_labels = labels['test'].astype(np.int32)
    cm = tf.math.confusion_matrix(target_labels, pred_labels, num_classes=2).numpy()

    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report = {
        'evaluation': {k: float(v) for k, v in eval_result.items()},
        'confusion_matrix': cm.astype(int).tolist(),
        'predictions': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        },
    }

    print(f"{name.upper()} metrics")
    print('  accuracy:', accuracy)
    print('  precision:', precision)
    print('  recall:', recall)
    print('  f1:', f1)
    print('  confusion matrix:', cm.astype(int).tolist())

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate trained gear classifiers on the held-out test split.')
    parser.add_argument('--models', nargs='+', choices=['cnn', 'lstm'], default=['cnn', 'lstm'], help='Which models to evaluate')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Split ratios (only used if splits need to be recreated)')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--l2-weight', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Loss smoothing used when compiling')
    parser.add_argument('--save-report', action='store_true', help='Store evaluation JSON next to training reports')
    args = parser.parse_args()

    log_gpu_info()

    bundle = load_combined_dataset(RAW_DATASETS)
    splits = ensure_splits(bundle.labels, bundle, args.train_ratio, args.val_ratio, args.seed)

    features = {}
    features['train'] = compute_fft_features(bundle.signals, bundle.lengths, splits['train'])
    features['val'] = compute_fft_features(bundle.signals, bundle.lengths, splits['val'])
    features['test'] = compute_fft_features(bundle.signals, bundle.lengths, splits['test'])

    features['train'], features['val'], features['test'] = normalize_features(features['train'], features['val'], features['test'])
    features['train'], features['val'], features['test'] = add_channel_dim(features['train'], features['val'], features['test'])

    labels = {
        'train': bundle.labels[splits['train']].astype(np.float32),
        'val': bundle.labels[splits['val']].astype(np.float32),
        'test': bundle.labels[splits['test']].astype(np.float32),
    }

    reports = {}
    for model_name in args.models:
        weights_path = DEFAULT_MODELS[model_name]
        report = evaluate_model(
            model_name,
            weights_path,
            features,
            labels,
            args.l2_weight,
            args.dropout,
            args.learning_rate,
            args.label_smoothing,
        )
        reports[model_name] = report

    if args.save_report:
        REPORT_DIR.mkdir(exist_ok=True)
        out_path = REPORT_DIR / 'evaluation_summary.json'
        out_path.write_text(json.dumps(reports, indent=2))
        print(f'Saved aggregate evaluation to {out_path}')


if __name__ == '__main__':
    main()
