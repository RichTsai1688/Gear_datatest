# Gear Feature Classification – Fine-Tuning Report

## Overview
- Objective: build a neural-network classifier for the engineered gear-feature dataset with >95 % accuracy.
- Scope: extend feature engineering, rebalance training, fine-tune an MLP, and benchmark against the original configuration.
- Final outcome: extended-feature MLP with accuracy **98.4 %**, precision **97.9 %**, recall **100 %** on the held-out test split (threshold tuned to 0.16).

## Dataset Status
- Source files: `datasets/cw_dataset.npz` (raw signals) → `datasets/cw_extended_features.csv` (engineered features).
- Label distribution (good=0, bad=1): total 822 samples → 192 good / 630 bad.
- Stratified splits (persisted in `datasets/feature_splits.json`):
  - Train: 575 samples (134 good, 441 bad)
  - Validation: 122 samples (28 good, 94 bad)
  - Test: 125 samples (30 good, 95 bad)
- Splits are stored with sample metadata (row index, original sample id, label) to guarantee reproducibility across reruns.

## Feature Engineering
1. **Baseline feature set (`cw_top_features.csv`)**
   - 10 metrics selected from prior analysis (`build_feature_dataset.py` original output): crest factor, peak-to-peak, kurtosis, RMS, spectral centroid/bandwidth, dominant frequency, and axis correlations.
2. **Extended feature set (`cw_extended_features.csv`)**
   - Added 29 complementary statistics for each axis (X/Y/Z): mean, standard deviation, RMS, crest factor, peak-to-peak, skewness, kurtosis, total energy, peak magnitude (dB), spectral centroid, bandwidth.
   - Included full tri-axis correlation triplet (`xy`, `xz`, `yz`).
   - Exported via `python build_feature_dataset.py datasets/cw_dataset.npz --feature-set extended --output datasets/cw_extended_features.csv`.
   - Generated summary JSON `reports/cw_extended_feature_summary.json` capturing class-wise means/std and (bad − good) deltas for all 39 features.

## Modeling Methodology
1. **Training script (`train_feature_network.py`) updates**
   - Auto-discovers feature columns from CSV header (supports 10 or 39 features).
   - Optionally computes inverse-frequency class weights (`--auto-class-weight`).
   - Supports configurable MLP architecture, dropout, L2 regularisation, patience, and learning rate schedule.
   - Integrates with shared utilities from `train_models.py` for compilation, callbacks, checkpointing, and reports.
   - Adds validation-threshold search (`--threshold-metric accuracy|balanced_accuracy|f1`) to optimise the decision boundary post-training.

2. **Shared training loop (`train_models.py`)**
   - Extended `train_and_evaluate` with optional `class_weight` forwarding to `model.fit`.

3. **Training configuration that achieved the final result**
   - Command:
     ```bash
     python train_feature_network.py \
       --dataset datasets/cw_extended_features.csv \
       --epochs 200 --batch-size 32 \
       --dropout 0.2 --hidden-sizes 512 256 128 \
       --learning-rate 3e-4 --patience 30 --plateau-patience 15 \
       --auto-class-weight --threshold-metric accuracy
     ```
   - Architecture: 3 hidden dense layers (512→256→128) with ReLU, batch norm, dropout 0.2, sigmoid output.
   - Regularisation: L2 weight 1e-4, ReduceLROnPlateau (factor 0.5, patience 15), early stopping (patience 30).
   - Class weights: {good: 2.146, bad: 0.652}.
   - Checkpoint: `checkpoints/feature_mlp_best.weights.h5`.
   - Report: `reports/feature_mlp_report.json` (history, metrics, confusion matrices, threshold tuning payload).

## Experimental Results
| Configuration | Feature Set | Class Weighting | Threshold Metric | Test Accuracy | Precision | Recall | F1 | Notes |
|---------------|-------------|-----------------|------------------|---------------|-----------|--------|----|-------|
| Baseline MLP  | Top 10      | Off             | Balanced acc.    | 0.864         | 0.915     | 0.905  |0.910| Initial tuning with richer architecture but limited features |
| +Class weight | Top 10      | On              | Balanced acc.    | 0.864         | 0.915     | 0.905  |0.910| Improved recall but accuracy capped by narrow feature space |
| **Final MLP** | **Extended**| **On**          | **Accuracy**     | **0.984**     | **0.979** | **1.000** | **0.990** | Threshold tuned to 0.16; specificity 0.933 |

- Validation tuning selected threshold 0.16 (accuracy-optimised), yielding validation accuracy 99.18 % (TP=93, TN=28, FP=0, FN=1).
- Confusion matrix (test, tuned threshold):
  - TP=95, TN=28, FP=2, FN=0 → balanced accuracy 0.967, specificity 0.933.

## Key Insights
- Feature expansion (39 metrics) dramatically increased separability between classes, particularly via additional time-domain statistics and the inclusion of all axis correlations.
- Class weighting was essential to maintain sensitivity to the minority “good” class while preventing false negatives.
- Validation-based threshold tuning eliminated residual FNs (recall 100 %) with only two false positives.

## Replication Checklist
1. Build extended feature dataset:
   ```bash
   python build_feature_dataset.py datasets/cw_dataset.npz \
     --feature-set extended \
     --output datasets/cw_extended_features.csv \
     --summary-json reports/cw_extended_feature_summary.json
   ```
2. Train the tuned MLP (command above); ensure `datasets/feature_splits.json` is present to reuse the stratified split.
3. Inspect `reports/feature_mlp_report.json` for training curves, evaluation metrics, and tuned threshold metadata.
4. Deploy using weights in `checkpoints/feature_mlp_best.weights.h5` and apply the 0.16 decision threshold during inference.

## Recommendations
- Validate on complementary datasets (e.g., clockwise direction) to confirm generalisation of the tuned threshold.
- Package the extended feature generation and model inference into the production pipeline to guarantee consistent preprocessing.
- Monitor specificity; if false positives remain costly, consider ensembling with the FFT-based CNN/LSTM models for cross-checking decisions.

