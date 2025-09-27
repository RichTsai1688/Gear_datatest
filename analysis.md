# CW Dataset Gear Condition Analysis

## Dataset Overview
- Source: `datasets/cw_dataset.npz`
- Total samples: 822 (good: 192, bad: 630)
- Sampling rate: ≈16.384 kHz, duration per sample: ≈5 s
- Axes: X (tangential), Y (radial), Z (axial)

## Key Observations
1. **X-axis impulsiveness increases in bad gears**
   - Crest factor rises from 56.86 → 64.93 on average.
   - Peak-to-peak amplitude increases from 3.37 → 3.87.
   - Kurtosis increases from 5.35e3 → 6.14e3, reflecting heavier tails.

2. **Y-axis energy shifts toward lower frequency for bad gears**
   - Spectral centroid drops from 948 Hz → 841 Hz.
   - Slight reduction in bandwidth (1317 Hz → 1299 Hz).
   - RMS energy also dips (0.082 → 0.081) despite occasional spikes.

3. **Z-axis spectrum broadens for bad gears**
   - Peak frequency shifts upward (336 Hz → 371 Hz).
   - Bandwidth expands from 1689 Hz → 1790 Hz.

4. **Axis correlations change**
   - XY mean correlation moves from +0.026 → -0.002.
   - XZ correlation becomes more negative (-0.339 → -0.369).
   - Indicates vibration coupling differences across axes.

## Notes & Considerations
- High-order statistics (crest, kurtosis, skewness) show large variance; investigate outliers before modeling.
- Class imbalance ratio ≈3.3:1 (bad:good); account for this in training.
- Derived features originate from `inspect_dataset.py`:
  - `compute_time_domain_metrics`
  - `compute_frequency_metrics`
  - `compute_axis_correlations`

## Suggested Next Steps
1. Visualize outlier samples with extreme crest factor/kurtosis to confirm validity.
2. Use the extracted feature set to train a classifier (e.g., gradient boosting, random forest) with imbalance mitigation.
3. Extend feature extraction to include additional statistics (e.g., band energy ratios) if needed.
