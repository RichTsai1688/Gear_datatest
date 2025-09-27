import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

RAW_ROOT = Path('gear_raw_data')
DEFAULT_OUTPUT = Path('datasets')
LABEL_ENCODE = {'good': 0, 'bad': 1}


def find_label(path: Path) -> str:
    lower = path.name.lower()
    if 'good' in lower:
        return 'good'
    if 'bad' in lower:
        return 'bad'
    raise ValueError(f'Unable to infer label from directory name: {path}')


def iter_csv_paths(direction: str) -> Iterable[Tuple[Path, str]]:
    suffix = direction.lower()
    for label_dir in sorted(RAW_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue
        label = find_label(label_dir)
        for case_dir in sorted(label_dir.iterdir()):
            name = case_dir.name.lower()
            if not name.endswith(suffix):
                continue
            for csv_path in sorted(case_dir.glob('*.csv')):
                yield csv_path, label


def load_signal(csv_path: Path) -> np.ndarray:
    return np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(1, 2, 3), dtype=np.float32)


def build_dataset(direction: str) -> Dict[str, np.ndarray]:
    samples: List[np.ndarray] = []
    labels: List[int] = []
    lengths: List[int] = []
    sources: List[str] = []

    for csv_path, label_name in iter_csv_paths(direction):
        signal = load_signal(csv_path)
        samples.append(signal)
        labels.append(LABEL_ENCODE[label_name])
        lengths.append(signal.shape[0])
        sources.append(str(csv_path.relative_to(RAW_ROOT)))

    if not samples:
        raise RuntimeError(f'No samples found for direction {direction}')

    max_len = max(lengths)
    dataset = np.zeros((len(samples), max_len, 3), dtype=np.float32)
    for idx, signal in enumerate(samples):
        dataset[idx, : signal.shape[0], :] = signal

    return {
        'signals': dataset,
        'labels': np.asarray(labels, dtype=np.int64),
        'lengths': np.asarray(lengths, dtype=np.int32),
        'sources': np.asarray(sources),
        'direction': np.asarray(direction.upper()),
        'label_names': np.asarray(['good', 'bad']),
    }


def save_dataset(data: Dict[str, np.ndarray], out_path: Path) -> None:
    np.savez_compressed(out_path, **data)


def summarize(data: Dict[str, np.ndarray]) -> Dict[str, object]:
    signals = data['signals']
    labels = data['labels']
    lengths = data['lengths']
    return {
        'samples': int(signals.shape[0]),
        'timesteps': int(signals.shape[1]),
        'label_counts': {
            'good': int((labels == LABEL_ENCODE['good']).sum()),
            'bad': int((labels == LABEL_ENCODE['bad']).sum()),
        },
        'length_stats': {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build CCW and CW datasets from gear CSV files')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for direction in ('ccw', 'cw'):
        data = build_dataset(direction)
        out_path = args.output_dir / f'{direction}_dataset.npz'
        save_dataset(data, out_path)
        summaries[direction.upper()] = summarize(data)

    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
