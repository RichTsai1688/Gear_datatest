import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

RAW_DATASETS = (Path('datasets') / 'ccw_dataset.npz', Path('datasets') / 'cw_dataset.npz')
SPLIT_PATH = Path('datasets') / 'splits.json'
CHECKPOINT_DIR = Path('checkpoints')
REPORT_DIR = Path('reports')


@dataclass
class DatasetBundle:
    signals: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    sources: np.ndarray
    directions: np.ndarray


def log_gpu_info() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPU detected. Training will run on CPU.')
        return
    print(f'Detected {len(gpus)} GPU(s):')
    for gpu in gpus:
        print(f'  - {gpu.name}')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - defensive path
            print(f'Warning: could not set memory growth for {gpu.name}: {exc}')


def load_single_dataset(path: Path) -> DatasetBundle:
    data = np.load(path, allow_pickle=False)
    signals = data['signals']
    lengths = data['lengths']
    labels = data['labels']
    sources = data['sources']
    direction_scalar = data['direction']
    if direction_scalar.ndim == 0:
        direction = str(direction_scalar.item())
    else:
        direction = str(direction_scalar[()]) if () in direction_scalar else str(direction_scalar.flatten()[0])
    directions = np.array([direction] * labels.shape[0])
    return DatasetBundle(signals=signals, lengths=lengths, labels=labels, sources=sources, directions=directions)


def load_combined_dataset(paths: Iterable[Path]) -> DatasetBundle:
    bundles = [load_single_dataset(path) for path in paths]
    signals = np.concatenate([b.signals for b in bundles], axis=0)
    lengths = np.concatenate([b.lengths for b in bundles], axis=0)
    labels = np.concatenate([b.labels for b in bundles], axis=0)
    sources = np.concatenate([b.sources for b in bundles], axis=0)
    directions = np.concatenate([b.directions for b in bundles], axis=0)
    return DatasetBundle(signals=signals, lengths=lengths, labels=labels, sources=sources, directions=directions)


def stratified_split(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for cls in np.unique(labels):
        cls_indices = indices[labels == cls]
        rng.shuffle(cls_indices)
        n_total = cls_indices.shape[0]
        n_train = int(math.floor(n_total * train_ratio))
        n_val = int(math.floor(n_total * val_ratio))
        remaining = n_total - n_train - n_val
        n_test = max(remaining, 0)
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:n_train + n_val + n_test])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return {
        'train': np.array(train_idx, dtype=np.int32),
        'val': np.array(val_idx, dtype=np.int32),
        'test': np.array(test_idx, dtype=np.int32),
    }


def save_splits(splits: Dict[str, np.ndarray], bundle: DatasetBundle, path: Path) -> None:
    payload = {}
    for key, idxs in splits.items():
        payload[key] = [
            {
                'index': int(idx),
                'label': int(bundle.labels[idx]),
                'source': str(bundle.sources[idx]),
                'direction': str(bundle.directions[idx]),
            }
            for idx in idxs
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_splits(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    out = {}
    for key, items in payload.items():
        out[key] = np.array([item['index'] for item in items], dtype=np.int32)
    return out


def compute_fft_features(signals: np.ndarray, lengths: np.ndarray, indices: np.ndarray) -> np.ndarray:
    spectra: List[np.ndarray] = []
    for idx in indices:
        length = int(lengths[idx])
        waveform = signals[idx, :length, 0].astype(np.float32)
        waveform_centered = waveform - waveform.mean()
        spectrum = np.abs(np.fft.rfft(waveform_centered))
        spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
        spectra.append(spectrum_db.astype(np.float32))
    return np.stack(spectra, axis=0)


def normalize_features(train_feat: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0)
    std[std == 0] = 1.0
    normalized = [(train_feat - mean) / std]
    for feat in others:
        normalized.append((feat - mean) / std)
    return tuple(normalized)


def add_channel_dim(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(arr[..., np.newaxis].astype(np.float32) for arr in arrays)


def build_cnn_model(input_shape: Tuple[int, int], l2_weight: float, dropout: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2_weight)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', kernel_regularizer=l2)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2)(x)
    return tf.keras.Model(inputs, outputs, name='cnn_classifier')


def build_lstm_model(input_shape: Tuple[int, int], l2_weight: float, dropout: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(l2_weight)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=l2))(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2)(x)
    return tf.keras.Model(inputs, outputs, name='lstm_classifier')


def compile_model(model: tf.keras.Model, lr: float, label_smoothing: float) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_and_evaluate(model: tf.keras.Model,
                       name: str,
                       data: Dict[str, np.ndarray],
                       labels: Dict[str, np.ndarray],
                       epochs: int,
                       batch_size: int,
                       patience: int,
                       plateau_patience: int,
                       class_weight: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f'{name}_best.weights.h5'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateau_patience, min_lr=1e-6),
    ]

    history = model.fit(
        data['train'], labels['train'],
        validation_data=(data['val'], labels['val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    # Load best weights before evaluation
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    eval_result = model.evaluate(data['test'], labels['test'], verbose=0, return_dict=True)

    preds = model.predict(data['test'], verbose=0)
    pred_labels = (preds.flatten() >= 0.5).astype(np.int32)
    target_labels = labels['test'].astype(np.int32)
    cm = tf.math.confusion_matrix(target_labels, pred_labels, num_classes=2)

    report = {
        'history': {k: [float(x) for x in v] for k, v in history.history.items()},
        'evaluation': {k: float(v) for k, v in eval_result.items()},
        'confusion_matrix': cm.numpy().astype(int).tolist(),
    }

    report_path = REPORT_DIR / f'{name}_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f'Saved report to {report_path}')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Train CNN and LSTM classifiers for gear dataset.')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--l2-weight', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--plateau-patience', type=int, default=3)
    parser.add_argument('--skip-lstm', action='store_true')
    parser.add_argument('--skip-cnn', action='store_true')
    args = parser.parse_args()

    log_gpu_info()

    bundle = load_combined_dataset(RAW_DATASETS)

    if SPLIT_PATH.exists():
        splits = load_splits(SPLIT_PATH)
        print(f'Loaded existing splits from {SPLIT_PATH}')
    else:
        splits = stratified_split(bundle.labels, args.train_ratio, args.val_ratio, args.seed)
        save_splits(splits, bundle, SPLIT_PATH)
        print(f'Saved splits to {SPLIT_PATH}')

    train_feat = compute_fft_features(bundle.signals, bundle.lengths, splits['train'])
    val_feat = compute_fft_features(bundle.signals, bundle.lengths, splits['val'])
    test_feat = compute_fft_features(bundle.signals, bundle.lengths, splits['test'])

    train_feat, val_feat, test_feat = normalize_features(train_feat, val_feat, test_feat)
    train_feat, val_feat, test_feat = add_channel_dim(train_feat, val_feat, test_feat)

    y_train = bundle.labels[splits['train']].astype(np.float32)
    y_val = bundle.labels[splits['val']].astype(np.float32)
    y_test = bundle.labels[splits['test']].astype(np.float32)

    data_dict = {
        'train': train_feat,
        'val': val_feat,
        'test': test_feat,
    }
    label_dict = {
        'train': y_train,
        'val': y_val,
        'test': y_test,
    }

    input_shape = train_feat.shape[1:]

    if not args.skip_cnn:
        cnn_model = build_cnn_model(input_shape, args.l2_weight, args.dropout)
        compile_model(cnn_model, args.learning_rate, args.label_smoothing)
        cnn_model.summary()
        train_and_evaluate(cnn_model, 'cnn', data_dict, label_dict, args.epochs, args.batch_size, args.patience, args.plateau_patience)

    if not args.skip_lstm:
        lstm_model = build_lstm_model(input_shape, args.l2_weight, args.dropout)
        compile_model(lstm_model, args.learning_rate, args.label_smoothing)
        lstm_model.summary()
        train_and_evaluate(lstm_model, 'lstm', data_dict, label_dict, args.epochs, args.batch_size, args.patience, args.plateau_patience)


if __name__ == '__main__':
    main()
