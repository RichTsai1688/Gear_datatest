# Gear Report Viewer

Simple Express-based viewer for the JSON reports produced by the training pipeline.

## Usage

```bash
cd viewer
npm install
npm start
```

The server listens on `http://localhost:3000`, serving the UI from `viewer/public`. It expects:

- Evaluation JSON files inside `../reports` (e.g. `cnn_report.json`).
- Dataset archives inside `../datasets` (e.g. `ccw_dataset.npz`, `cw_dataset.npz`).

## Endpoints

- `GET /api/reports` &mdash; list available JSON files.
- `GET /api/reports/:name` &mdash; fetch a specific report.

The front-end automatically loads the first report and displays metrics, prediction summaries, confusion matrix, and training history when available.

## Dataset Explorer

The "Dataset Explorer" panel allows browsing individual samples directly from the `.npz` datasets. It shells out to `inspect_dataset.py` with `--json` mode to fetch the selected sample's time-domain traces and FFT magnitudes (downsampled for practicality) and plots them with Chart.js.

- Choose a dataset and provide the sample index.
- Click **Load Sample** to visualize the X/Y/Z waveforms and frequency spectra.
- Change datasets from the dropdown to switch between CCW/CW archives.
