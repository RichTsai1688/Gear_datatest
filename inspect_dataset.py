import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import heapq

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    plt = None
    MATPLOTLIB_ERROR = exc
else:
    MATPLOTLIB_ERROR = None

RAW_ROOT = Path('gear_raw_data')
AXIS_KEYS = ('x', 'y', 'z')
AXIS_LABELS = ('X', 'Y', 'Z')


def load_dataset(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def build_summary(data: dict) -> Dict[str, object]:
    signals = data['signals']
    labels = data['labels']
    lengths = data['lengths']
    label_names = data['label_names']

    axis_time_stats = compute_dataset_axis_metrics(signals, lengths)
    axis_correlations = compute_dataset_axis_correlations(signals, lengths)

    return {
        'direction': str(data['direction']),
        'samples': int(signals.shape[0]),
        'timesteps_padded': int(signals.shape[1]),
        'label_counts': {
            str(label_names[idx]): int((labels == idx).sum())
            for idx in range(label_names.shape[0])
        },
        'length_stats': {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
        },
        'axis_time_stats': axis_time_stats,
        'axis_correlations': axis_correlations,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print(f"direction: {summary['direction']}\n")
    print('samples:', summary['samples'])
    print('timesteps (padded):', summary['timesteps_padded'])
    print('label counts:')
    for name, count in summary['label_counts'].items():
        print(f'  {name}: {count}')
    print('length stats:')
    print('  min:', summary['length_stats']['min'])
    print('  max:', summary['length_stats']['max'])
    print('axis time-domain stats (global):')
    for axis_label, axis_key in zip(AXIS_LABELS, AXIS_KEYS):
        stats = summary['axis_time_stats'][axis_key]
        print(
            f"  {axis_label}: mean={stats['mean']:.4g}, std={stats['std']:.4g}, "
            f"rms={stats['rms']:.4g}, crest={stats['crest_factor']:.4g}, "
            f"kurtosis={stats['kurtosis']:.4g}"
        )
    print('axis correlations (mean Pearson):')
    for pair, value in summary['axis_correlations'].items():
        print(f'  {pair.upper()}: {value:.4g}')


def resolve_indices(indices: Optional[Iterable[int]], total: int) -> List[int]:
    if not indices:
        return []
    resolved: List[int] = []
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(f'sample index {idx} out of bounds (0, {total - 1})')
        resolved.append(idx)
    return resolved


def load_time_vector(source: str, expected_length: int) -> np.ndarray:
    csv_path = RAW_ROOT / source
    time = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(0,), dtype=np.float32)
    if time.shape[0] != expected_length:
        raise ValueError(f'time column length mismatch for {source}')
    return time


def plot_sample(time: np.ndarray, signal: np.ndarray, title: str, save_prefix: Optional[Path]) -> None:
    if plt is None:
        raise SystemExit('matplotlib is required for plotting. Install it or use --json mode instead.')

    signal_centered = signal - signal.mean(axis=0, keepdims=True)
    dt = float(np.mean(np.diff(time))) if time.size > 1 else 1.0
    freqs = np.fft.rfftfreq(time.size, dt)
    axes_fft = np.fft.rfft(signal_centered, axis=0)
    magnitude = np.abs(axes_fft)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col')
    axis_labels = ('X', 'Y', 'Z')
    for row in range(3):
        axes[row, 0].plot(time, signal[:, row])
        axes[row, 0].set_ylabel(axis_labels[row])
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].plot(freqs, magnitude_db[:, row])
        axes[row, 1].grid(True, alpha=0.3)
    axes[0, 0].set_title('Time Domain')
    axes[0, 1].set_title('Frequency Domain (|FFT| dB)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    fig.suptitle(title)
    fig.tight_layout()

    if save_prefix is None:
        plt.show()
    else:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        path = prefix.with_suffix('.png')
        fig.savefig(path, dpi=150)
        print(f'saved plot to {path}')
    plt.close(fig)


def decimate(array: np.ndarray, max_points: Optional[int] = None, stride: int = 1) -> np.ndarray:
    if stride > 1:
        array = array[::stride]
    if max_points is not None and array.shape[0] > max_points:
        idx = np.linspace(0, array.shape[0] - 1, num=max_points)
        idx = np.floor(idx).astype(int)
        array = array[idx]
    return array


def decimate_2d(array: np.ndarray, max_points: Optional[int] = None, stride: int = 1) -> np.ndarray:
    if stride > 1:
        array = array[::stride, :]
    if max_points is not None and array.shape[0] > max_points:
        idx = np.linspace(0, array.shape[0] - 1, num=max_points)
        idx = np.floor(idx).astype(int)
        array = array[idx, :]
    return array


def _safe_moment(data: np.ndarray, moment: int, mean: float, std: float) -> float:
    if std <= 0.0:
        return float('nan')
    normalized = (data - mean) / std
    return float(np.mean(normalized ** moment))


def compute_time_domain_metrics(signal: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for axis, axis_key in enumerate(AXIS_KEYS):
        axis_data = signal[:, axis]
        mean = float(axis_data.mean())
        std = float(axis_data.std())
        rms = float(np.sqrt(np.mean(axis_data ** 2)))
        abs_max = float(np.max(np.abs(axis_data))) if axis_data.size else 0.0
        crest_factor = abs_max / rms if rms > 0.0 else float('nan')
        peak_to_peak = float(axis_data.max() - axis_data.min()) if axis_data.size else float('nan')
        skewness = _safe_moment(axis_data, 3, mean, std)
        kurtosis = _safe_moment(axis_data, 4, mean, std)
        energy = float(np.sum(axis_data ** 2))
        metrics[axis_key] = {
            'mean': mean,
            'std': std,
            'rms': rms,
            'crest_factor': crest_factor,
            'peak_to_peak': peak_to_peak,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'energy': energy,
        }
    return metrics


def compute_frequency_metrics(freqs: np.ndarray,
                              magnitude: np.ndarray,
                              magnitude_db: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for axis, axis_key in enumerate(AXIS_KEYS):
        axis_mag = magnitude[:, axis]
        axis_mag_db = magnitude_db[:, axis]
        if axis_mag.size == 0:
            metrics[axis_key] = {
                'peak_frequency_hz': float('nan'),
                'peak_magnitude_db': float('nan'),
                'spectral_centroid_hz': float('nan'),
                'bandwidth_hz': float('nan'),
            }
            continue
        peak_idx = int(np.argmax(axis_mag))
        power = axis_mag ** 2
        total_power = float(np.sum(power))
        if total_power <= 0.0:
            centroid = float('nan')
            bandwidth = float('nan')
        else:
            centroid = float(np.sum(freqs * power) / total_power)
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
        metrics[axis_key] = {
            'peak_frequency_hz': float(freqs[peak_idx]),
            'peak_magnitude_db': float(axis_mag_db[peak_idx]),
            'spectral_centroid_hz': centroid,
            'bandwidth_hz': bandwidth,
        }
    return metrics


def compute_dataset_axis_metrics(signals: np.ndarray, lengths: np.ndarray) -> Dict[str, Dict[str, float]]:
    concatenated: Dict[str, List[np.ndarray]] = {axis: [] for axis in AXIS_KEYS}
    for idx, length in enumerate(lengths.astype(int)):
        if length <= 0:
            continue
        sample = signals[idx, :length, :]
        for axis, axis_key in enumerate(AXIS_KEYS):
            concatenated[axis_key].append(sample[:, axis])

    summary: Dict[str, Dict[str, float]] = {}
    for axis_key, values in concatenated.items():
        if not values:
            summary[axis_key] = {
                'mean': float('nan'),
                'std': float('nan'),
                'rms': float('nan'),
                'crest_factor': float('nan'),
                'peak_to_peak': float('nan'),
                'skewness': float('nan'),
                'kurtosis': float('nan'),
                'energy_total': float('nan'),
            }
            continue
        axis_data = np.concatenate(values)
        mean = float(axis_data.mean())
        std = float(axis_data.std())
        rms = float(np.sqrt(np.mean(axis_data ** 2)))
        abs_max = float(np.max(np.abs(axis_data))) if axis_data.size else 0.0
        crest_factor = abs_max / rms if rms > 0.0 else float('nan')
        peak_to_peak = float(axis_data.max() - axis_data.min()) if axis_data.size else float('nan')
        skewness = _safe_moment(axis_data, 3, mean, std)
        kurtosis = _safe_moment(axis_data, 4, mean, std)
        energy_total = float(np.sum(axis_data ** 2))
        summary[axis_key] = {
            'mean': mean,
            'std': std,
            'rms': rms,
            'crest_factor': crest_factor,
            'peak_to_peak': peak_to_peak,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'energy_total': energy_total,
        }
    return summary


def compute_axis_correlations(signal: np.ndarray) -> Dict[str, float]:
    if signal.shape[0] < 2:
        return {'xy': float('nan'), 'xz': float('nan'), 'yz': float('nan')}
    corr_matrix = np.corrcoef(signal, rowvar=False)
    return {
        'xy': float(corr_matrix[0, 1]),
        'xz': float(corr_matrix[0, 2]),
        'yz': float(corr_matrix[1, 2]),
    }


def compute_dataset_axis_correlations(signals: np.ndarray, lengths: np.ndarray) -> Dict[str, float]:
    accum: Dict[str, List[float]] = {'xy': [], 'xz': [], 'yz': []}
    for idx, length in enumerate(lengths.astype(int)):
        if length < 2:
            continue
        sample = signals[idx, :length, :]
        corr = compute_axis_correlations(sample)
        for key, value in corr.items():
            if not np.isnan(value):
                accum[key].append(value)

    summary: Dict[str, float] = {}
    for key, values in accum.items():
        summary[key] = float(np.mean(values)) if values else float('nan')
    return summary


def collect_sample_json(idx: int,
                        data: dict,
                        length: int,
                        source: str,
                        label_name: str,
                        max_points: Optional[int],
                        stride: int,
                        max_freqs: Optional[int]) -> Dict[str, object]:
    time = load_time_vector(source, length)
    signal = data['signals'][idx, :length, :]
    signal_centered = signal - signal.mean(axis=0, keepdims=True)
    dt = float(np.mean(np.diff(time))) if time.size > 1 else 1.0
    freqs = np.fft.rfftfreq(time.size, dt)
    magnitude = np.abs(np.fft.rfft(signal_centered, axis=0))
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    freqs_full = freqs
    magnitude_full = magnitude
    magnitude_db_full = magnitude_db

    time_dec = decimate(time, max_points, stride)
    signal_dec = decimate_2d(signal, max_points, stride)

    if max_freqs is not None and freqs.shape[0] > max_freqs:
        freq_idx = np.linspace(0, freqs.shape[0] - 1, num=max_freqs)
        freq_idx = np.floor(freq_idx).astype(int)
        freqs = freqs[freq_idx]
        magnitude = magnitude[freq_idx, :]
        magnitude_db = magnitude_db[freq_idx, :]

    time_metrics = compute_time_domain_metrics(signal)
    freq_metrics = compute_frequency_metrics(freqs_full, magnitude_full, magnitude_db_full)
    correlations = compute_axis_correlations(signal)

    return {
        'index': int(idx),
        'label': label_name,
        'source': source,
        'length': length,
        'time': time_dec.astype(float).tolist(),
        'signals': {
            'x': signal_dec[:, 0].astype(float).tolist(),
            'y': signal_dec[:, 1].astype(float).tolist(),
            'z': signal_dec[:, 2].astype(float).tolist(),
        },
        'frequency': {
            'freqs': freqs.astype(float).tolist(),
            'magnitude_linear': {
                'x': magnitude[:, 0].astype(float).tolist(),
                'y': magnitude[:, 1].astype(float).tolist(),
                'z': magnitude[:, 2].astype(float).tolist(),
            },
            'magnitude_db': {
                'x': magnitude_db[:, 0].astype(float).tolist(),
                'y': magnitude_db[:, 1].astype(float).tolist(),
                'z': magnitude_db[:, 2].astype(float).tolist(),
            },
        },
        'statistics': {
            'time_domain': time_metrics,
            'frequency_domain': freq_metrics,
            'axis_correlations': correlations,
        },
    }


def extract_significant_peaks(freqs: np.ndarray, magnitude_db: np.ndarray, top_n: int = 20):
    """
    Extract the top N significant peaks from the frequency spectrum.

    Args:
        freqs (np.ndarray): Array of frequencies.
        magnitude_db (np.ndarray): Array of magnitudes in dB.
        top_n (int): Number of top peaks to extract.

    Returns:
        List[Tuple[float, float]]: List of tuples containing (frequency, magnitude_db).
    """
    peaks = [(magnitude_db[i], freqs[i]) for i in range(len(freqs))]
    top_peaks = heapq.nlargest(top_n, peaks, key=lambda x: x[0])
    return [(freq, mag) for mag, freq in top_peaks]


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect gear datasets with optional plots')
    parser.add_argument('dataset', type=Path, help='Path to dataset npz file (e.g. datasets/ccw_dataset.npz)')
    parser.add_argument('--indices', type=int, nargs='*', help='Sample indices to visualize')
    parser.add_argument('--save-prefix', type=Path, help='Optional path prefix to save plots instead of showing')
    parser.add_argument('--no-summary', action='store_true', help='Skip dataset summary output')
    parser.add_argument('--json', action='store_true', help='Emit JSON payload instead of plotting')
    parser.add_argument('--max-points', type=int, default=None, help='Maximum time-domain points in JSON output (after stride)')
    parser.add_argument('--stride', type=int, default=1, help='Stride applied before limiting time-domain points in JSON output')
    parser.add_argument('--max-freqs', type=int, default=None, help='Maximum frequency bins in JSON output')
    args = parser.parse_args()

    if args.json:
        # JSON output does not require matplotlib
        pass
    elif plt is None:
        raise SystemExit('matplotlib is required for plotting. Install it or use --json mode.')

    data = load_dataset(args.dataset)
    summary = build_summary(data)

    if not args.no_summary and not args.json:
        print_summary(summary)

    total = data['signals'].shape[0]
    indices = resolve_indices(args.indices, total)

    if args.json:
        payload: Dict[str, object] = {
            'dataset': args.dataset.name,
            'summary': summary,
            'samples': [],
        }
        if not indices:
            raise SystemExit('JSON mode requires at least one --indices value.')
        for idx in indices:
            length = int(data['lengths'][idx])
            source = str(data['sources'][idx])
            label_idx = int(data['labels'][idx])
            label_name = str(data['label_names'][label_idx])
            sample = collect_sample_json(
                idx,
                data,
                length,
                source,
                label_name,
                args.max_points,
                args.stride,
                args.max_freqs,
            )
            payload['samples'].append(sample)
        print(json.dumps(payload))
        return

    for idx in indices:
        length = int(data['lengths'][idx])
        source = str(data['sources'][idx])
        label_idx = int(data['labels'][idx])
        label_name = str(data['label_names'][label_idx])

        time = load_time_vector(source, length)
        signal = data['signals'][idx, :length, :]
        title = f'{args.dataset.name} | idx={idx} | label={label_name} | source={source}'

        # Compute FFT and extract peaks
        signal_centered = signal - signal.mean(axis=0, keepdims=True)
        dt = float(np.mean(np.diff(time))) if time.size > 1 else 1.0
        freqs = np.fft.rfftfreq(time.size, dt)
        axes_fft = np.fft.rfft(signal_centered, axis=0)
        magnitude = np.abs(axes_fft)
        magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))

        time_metrics = compute_time_domain_metrics(signal)
        freq_metrics = compute_frequency_metrics(freqs, magnitude, magnitude_db)
        correlations = compute_axis_correlations(signal)

        print('Time-domain metrics:')
        for axis_key, axis_label in zip(AXIS_KEYS, AXIS_LABELS):
            stats = time_metrics[axis_key]
            print(
                f"  {axis_label}: mean={stats['mean']:.4g}, std={stats['std']:.4g}, "
                f"rms={stats['rms']:.4g}, crest={stats['crest_factor']:.4g}, "
                f"kurtosis={stats['kurtosis']:.4g}"
            )

        print('Frequency-domain metrics:')
        for axis_key, axis_label in zip(AXIS_KEYS, AXIS_LABELS):
            stats = freq_metrics[axis_key]
            print(
                f"  {axis_label}: peak={stats['peak_frequency_hz']:.2f} Hz ({stats['peak_magnitude_db']:.2f} dB), "
                f"centroid={stats['spectral_centroid_hz']:.2f} Hz, bandwidth={stats['bandwidth_hz']:.2f} Hz"
            )

        print('Axis correlations:')
        print(
            f"  XY: {correlations['xy']:.4g}, XZ: {correlations['xz']:.4g}, "
            f"YZ: {correlations['yz']:.4g}"
        )

        for axis, axis_label in enumerate(AXIS_LABELS):
            print(f"Top {args.max_freqs or 20} peaks for axis {axis_label}:")
            peaks = extract_significant_peaks(freqs, magnitude_db[:, axis], top_n=args.max_freqs or 20)
            for freq, mag in peaks:
                print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f} dB")

        plot_sample(time, signal, title, args.save_prefix)


if __name__ == '__main__':
    main()
