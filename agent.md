# Gear Audio Classifier Agent Plan

This agent orchestrates dataset preparation, model training, and evaluation for classifying gear health from vibration signals.

## 1. Dataset Splits
- Load `datasets/ccw_dataset.npz` and `datasets/cw_dataset.npz`; keep direction metadata so we can train direction-specific or combined models.
- Concatenate CCW and CW samples (optionally add a `direction` one-hot feature later) and shuffle with a fixed seed.
- Split indexes into train/validation/test, e.g. 70/15/15, stratified by `labels` to maintain the good/bad ratio.
- Persist split indices (e.g. JSON with sample ids) to ensure reproducibility for future runs.

## 2. Preprocessing Pipeline
- Extract the X-axis waveform from each sample (`signals[:, :, 0]`) using the stored `lengths` to trim padding.
- Compute magnitude spectrum via real FFT: `fft = np.abs(np.fft.rfft(x_waveform))`. Keep consistent frequency resolution (pad/truncate to a common length if needed).
- Stack or concatenate frequency-domain features with time-domain stats if beneficial.
- Normalize features per split:
  - Fit `StandardScaler` (or min-max) on the training spectra.
  - Apply the same scaler to validation and test sets.
- Cache processed tensors to disk (`.npz` or TFRecords) for faster reuse.

## 3. Model Architectures
Prepare two TensorFlow/Keras models sharing preprocessing + training utilities.

### 3.1 CNN Model
- Input: normalized spectrum (shape `[freq_bins, 1]`).
- Use 1D convolutions with kernel sizes that capture local frequency patterns (e.g. Conv1D → BatchNorm → ReLU → Dropout).
- Add global average pooling and dense layers with L2 regularization (`kernel_regularizer=tf.keras.regularizers.l2(l2_weight)`).
- Output layer: `Dense(1, activation='sigmoid', kernel_regularizer=l2, bias_regularizer=l2)`. Enable label smoothing via `BinaryCrossentropy(label_smoothing=0.1)` when overfitting appears.

### 3.2 LSTM Model
- Optionally feed either time-domain trimmed sequences or spectrogram patches.
- Example stack: Bidirectional LSTM → Dropout → Dense (L2) → Sigmoid output.
- Keep sequence length manageable (downsample or segment if necessary).

## 4. Training Strategy
- Optimizer: `Adam(learning_rate=initial_lr)`.
- Compile with `metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]` to track class performance.
- Callbacks configuration:
  - `EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)`.
  - `ModelCheckpoint(filepath='checkpoints/best_{model}.keras', monitor='val_loss', save_best_only=True)`.
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateau_patience, min_lr=1e-6)`.
- Log training via TensorBoard or CSVLogger for later review.

## 5. Evaluation
- Load best weights for each model and evaluate on the held-out test dataset.
- Generate predictions and build a confusion matrix (`tf.math.confusion_matrix` or `sklearn.metrics.confusion_matrix`).
- Compute accuracy, precision, recall, F1 for test results; highlight good/bad classification balance.
- Compare CNN vs LSTM: summarize metrics side-by-side and inspect misclassified samples.
- Save confusion matrix plots/images and a final report (`reports/test_eval.json`) documenting performance and settings.

## Future Extensions
- Experiment with combining CCW/CW models or multi-task training that predicts both condition and direction.
- Explore data augmentation (noise injection, random windowing) if datasets show imbalance.
- Automate hyperparameter sweeps with KerasTuner or Optuna.
