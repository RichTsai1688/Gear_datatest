"""Extract discriminative gear features for model training.

This script reproduces the analysis pipeline used in `analysis.md` by:
1. Computing time-domain, frequency-domain, and cross-axis correlation metrics
   via helpers defined in `inspect_dataset.py`.
2. Selecting the most discriminative statistics between good and bad gears.
3. Exporting a tabular dataset suitable for downstream ML training.
4. Emitting a JSON summary comparing class-wise feature means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import inspect_dataset as inspector

# Features selected from the manual analysis as the most discriminative signals.
FEATURE_NAMES = (
    "x_crest_factor",
    "x_peak_to_peak",
    "x_kurtosis",
    "y_rms",
    "y_spectral_centroid_hz",
    "y_bandwidth_hz",
    "z_peak_frequency_hz",
    "z_bandwidth_hz",
    "xy_corr",
    "xz_corr",
)

EXTENDED_FEATURE_NAMES = (
    "x_mean",
    "x_std",
    "x_rms",
    "x_crest_factor",
    "x_peak_to_peak",
    "x_skewness",
    "x_kurtosis",
    "x_energy",
    "y_mean",
    "y_std",
    "y_rms",
    "y_crest_factor",
    "y_peak_to_peak",
    "y_skewness",
    "y_kurtosis",
    "y_energy",
    "z_mean",
    "z_std",
    "z_rms",
    "z_crest_factor",
    "z_peak_to_peak",
    "z_skewness",
    "z_kurtosis",
    "z_energy",
    "x_peak_frequency_hz",
    "x_peak_magnitude_db",
    "x_spectral_centroid_hz",
    "x_bandwidth_hz",
    "y_peak_frequency_hz",
    "y_peak_magnitude_db",
    "y_spectral_centroid_hz",
    "y_bandwidth_hz",
    "z_peak_frequency_hz",
    "z_peak_magnitude_db",
    "z_spectral_centroid_hz",
    "z_bandwidth_hz",
    "xy_corr",
    "xz_corr",
    "yz_corr",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        default=Path("datasets/cw_dataset.npz"),
        nargs="?",
        help="Path to the gear dataset (NPZ). Default: datasets/cw_dataset.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/cw_top_features.csv"),
        help="Output CSV path for the feature dataset.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/cw_feature_summary.json"),
        help="Path to save the feature statistics summary JSON.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("top10", "extended"),
        default="top10",
        help="Select the feature subset to export.",
    )
    return parser


def compute_sample_features(signal: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """Compute discriminative features for a single sample."""
    time_stats = inspector.compute_time_domain_metrics(signal)
    centered = signal - signal.mean(axis=0, keepdims=True)
    magnitude = np.abs(np.fft.rfft(centered, axis=0))
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    freq_stats = inspector.compute_frequency_metrics(freqs, magnitude, magnitude_db)
    corr_stats = inspector.compute_axis_correlations(signal)

    features: Dict[str, float] = {}

    for axis in ("x", "y", "z"):
        axis_stats = time_stats[axis]
        features[f"{axis}_mean"] = float(axis_stats["mean"])
        features[f"{axis}_std"] = float(axis_stats["std"])
        features[f"{axis}_rms"] = float(axis_stats["rms"])
        features[f"{axis}_crest_factor"] = float(axis_stats["crest_factor"])
        features[f"{axis}_peak_to_peak"] = float(axis_stats["peak_to_peak"])
        features[f"{axis}_skewness"] = float(axis_stats["skewness"])
        features[f"{axis}_kurtosis"] = float(axis_stats["kurtosis"])
        features[f"{axis}_energy"] = float(axis_stats["energy"])

        axis_freq = freq_stats[axis]
        features[f"{axis}_peak_frequency_hz"] = float(axis_freq["peak_frequency_hz"])
        features[f"{axis}_peak_magnitude_db"] = float(axis_freq["peak_magnitude_db"])
        features[f"{axis}_spectral_centroid_hz"] = float(axis_freq["spectral_centroid_hz"])
        features[f"{axis}_bandwidth_hz"] = float(axis_freq["bandwidth_hz"])

    features["xy_corr"] = float(corr_stats["xy"])
    features["xz_corr"] = float(corr_stats["xz"])
    features["yz_corr"] = float(corr_stats["yz"])

    return features


def ensure_parent(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def export_csv(rows: List[Dict[str, float]], header: List[str], path: Path) -> None:
    import csv

    ensure_parent(path)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    data = inspector.load_dataset(args.dataset)
    signals = data["signals"]
    labels = data["labels"].astype(int)
    lengths = data["lengths"].astype(int)
    sources = data["sources"]
    label_names = data["label_names"]

    label_map = {idx: str(name) for idx, name in enumerate(label_names)}

    feature_names = FEATURE_NAMES if args.feature_set == "top10" else EXTENDED_FEATURE_NAMES

    # Compute sampling frequency using the first sample's time vector.
    first_length = lengths[0]
    time_vector = inspector.load_time_vector(str(sources[0]), first_length)
    dt = float(np.mean(np.diff(time_vector))) if time_vector.size > 1 else 1.0
    freqs = np.fft.rfftfreq(first_length, dt)

    rows: List[Dict[str, float]] = []
    feature_accumulator: Dict[str, Dict[str, List[float]]] = {
        label_name: {feature: [] for feature in feature_names}
        for label_name in label_map.values()
    }

    for idx in range(signals.shape[0]):
        length = lengths[idx]
        sample = signals[idx, :length, :]
        features = compute_sample_features(sample, freqs)
        label_idx = labels[idx]
        label_name = label_map[label_idx]

        # Record per-sample features for downstream ML.
        row = {name: features[name] for name in feature_names}
        row["label"] = label_name
        row["label_idx"] = label_idx
        row["sample_index"] = idx
        row["source"] = str(sources[idx])
        rows.append(row)

        # Accumulate for summary statistics.
        for feature_name in feature_names:
            feature_accumulator[label_name][feature_name].append(features[feature_name])

    # Export feature dataset.
    export_header = ["sample_index", "source", "label_idx", "label", *feature_names]
    export_csv(rows, export_header, args.output)

    # Build summary statistics (mean/std per label + differences).
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for feature_name in feature_names:
        summary[feature_name] = {}
        for label_name in label_map.values():
            values = np.asarray(feature_accumulator[label_name][feature_name], dtype=float)
            summary[feature_name][label_name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

        good_mean = summary[feature_name].get("good", {}).get("mean", np.nan)
        bad_mean = summary[feature_name].get("bad", {}).get("mean", np.nan)
        summary[feature_name]["difference"] = {
            "bad_minus_good": float(bad_mean - good_mean),
            "relative_change": float((bad_mean - good_mean) / good_mean) if good_mean else float("nan"),
        }

    # Save summary JSON.
    ensure_parent(args.summary_json)
    with args.summary_json.open("w") as fp:
        json.dump(summary, fp, indent=2)

    # Also print a concise table to the console for quick inspection.
    print("Feature comparison (bad - good):")
    for feature_name in feature_names:
        diff = summary[feature_name]["difference"]["bad_minus_good"]
        rel = summary[feature_name]["difference"]["relative_change"]
        print(f"  {feature_name:>22}: diff={diff:10.4f}, rel={rel:7.2%}")

    print(f"\nExported feature dataset to {args.output}")
    print(f"Saved summary statistics to {args.summary_json}")


if __name__ == "__main__":
    main()
