# Gear Feature Extraction Methodology

## 1. Raw Signal Context
- Source data: three-axis vibration recordings saved in `gear_raw_data/*` (each CSV contains time and X/Y/Z amplitude columns).
- Pre-bundled dataset: `datasets/cw_dataset.npz` (clockwise rotation) with fields `signals`, `lengths`, `labels`, `sources`, and `label_names` ("good", "bad").
- Sampling: inferred per sample from the time vector (`inspect_dataset.load_time_vector`) to compute frequency bins via `np.fft.rfftfreq`.

## 2. Feature Families
Feature extraction reuses the analysis helpers in `inspect_dataset.py` and is orchestrated through `build_feature_dataset.py`.

### 2.1 Time-Domain Metrics (per axis X/Y/Z)
- Mean & standard deviation
- Root-mean-square (RMS)
- Crest factor (|max| / RMS)
- Peak-to-peak amplitude
- Skewness and kurtosis (using `_safe_moment`)
- Total energy (`np.sum(axis_data ** 2)`)

### 2.2 Frequency-Domain Metrics (per axis X/Y/Z)
- Dominant frequency (`np.argmax(|FFT|)`)
- Peak magnitude (dB scale)
- Spectral centroid (power-weighted mean frequency)
- Spectral bandwidth (power-weighted standard deviation)

### 2.3 Cross-Axis Correlations
- Pearson correlations between axis pairs: XY, XZ, YZ (computed via `np.corrcoef`).

### 2.4 Feature Sets
1. **Top 10 subset (`FEATURE_NAMES`)** – manual selection of the most discriminative statistics observed during exploratory analysis: crest factor/peak-to-peak/kurtosis (X), RMS/spectral centroid/bandwidth (Y), peak frequency/bandwidth (Z), and XY/XZ correlations. Output defaults to `datasets/cw_top_features.csv`.
2. **Extended set (`EXTENDED_FEATURE_NAMES`)** – full suite of 39 metrics enumerated above, saved to `datasets/cw_extended_features.csv` when running with `--feature-set extended`.

## 3. Export Workflow
Command example:
```bash
python build_feature_dataset.py datasets/cw_dataset.npz \
  --feature-set extended \
  --output datasets/cw_extended_features.csv \
  --summary-json reports/cw_extended_feature_summary.json
```
- For each sample: compute time/frequency stats and correlations (see `compute_sample_features`).
- Write per-sample rows: `sample_index`, `source`, `label_idx`, `label`, and selected features.
- Persist summary JSON with per-class means/stds and (bad − good) differences for every feature.

## 4. Class Comparisons (Extended Set)
Derived from `reports/cw_extended_feature_summary.json`; key contrasts between "good" and "bad" gears:
- **X-axis extremes**: crest factor +8.07 (bad > good), peak-to-peak +0.50, kurtosis +794.50 → defective gears show spikier X-axis shocks.
- **Y-axis energy**: RMS −0.0014 and bandwidth −17.86 (bad < good) but spectral centroid −107.67 Hz → energy shifts toward lower frequencies when bad.
- **Z-axis frequency spread**: peak frequency +35.20 Hz, bandwidth +100.50 Hz → broader high-frequency activity for bad gears.
- **Axis correlations**: XY correlation decreases (−0.0282), suggesting degraded coupling; XZ/YZ shifts are smaller but negative.
- **Higher-order moments**: bad gears exhibit larger X skewness (+2.42) and Y skewness (−1.34, flipping sign), highlighting asymmetry changes.

## 5. Usage in Modeling
- `train_feature_network.py` auto-detects feature columns, normalises them (z-score from the train split), and feeds them into the MLP.
- The viewer (`/api/model/summary`) shows both the top-10 list and the fuller 39-column set so downstream consumers can understand the preprocessing contract.

## 6. Reproducibility Notes
- Splits recorded in `datasets/feature_splits.json` ensure consistent train/val/test partitions with metadata (row index, sample index, label).
- Feature CSVs include the original sample index and source file for traceability back to raw signals.
- Running `build_feature_dataset.py` with the same arguments is deterministic (no RNG usage).

