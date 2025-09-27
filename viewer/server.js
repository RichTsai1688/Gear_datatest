import express from 'express';
import path from 'path';
import fs from 'fs/promises';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPORT_DIR = path.resolve(__dirname, '..', 'reports');
const DATASET_DIR = path.resolve(__dirname, '..', 'datasets');
const INSPECT_SCRIPT = path.resolve(__dirname, '..', 'inspect_dataset.py');
const PYTHON_BIN = process.env.PYTHON_BIN || 'python';
const app = express();
const PORT = process.env.PORT || 3000;

const FEATURE_DATASET = path.join(DATASET_DIR, 'cw_extended_features.csv');
const FEATURE_SPLITS = path.join(DATASET_DIR, 'feature_splits.json');
const FEATURE_REPORT = path.join(REPORT_DIR, 'feature_mlp_report.json');
const TOP_FEATURES = [
  'x_crest_factor',
  'x_peak_to_peak',
  'x_kurtosis',
  'y_rms',
  'y_spectral_centroid_hz',
  'y_bandwidth_hz',
  'z_peak_frequency_hz',
  'z_bandwidth_hz',
  'xy_corr',
  'xz_corr',
];

app.use(express.static(path.join(__dirname, 'public')));

async function listReports() {
  const entries = await fs.readdir(REPORT_DIR);
  return entries.filter((name) => name.endsWith('.json'));
}

async function readReport(name) {
  const target = path.join(REPORT_DIR, name);
  const data = await fs.readFile(target, 'utf-8');
  return JSON.parse(data);
}

app.get('/api/reports', async (req, res) => {
  try {
    const reports = await listReports();
    res.json({ reports });
  } catch (error) {
    console.error('Failed to list reports:', error);
    res.status(500).json({ error: 'Unable to list reports' });
  }
});

app.get('/api/reports/:name', async (req, res) => {
  try {
    const { name } = req.params;
    if (!name.endsWith('.json')) {
      return res.status(400).json({ error: 'Report name must end with .json' });
    }
    const available = await listReports();
    if (!available.includes(name)) {
      return res.status(404).json({ error: 'Report not found' });
    }
    const report = await readReport(name);
    res.json(report);
  } catch (error) {
    console.error('Failed to read report:', error);
    res.status(500).json({ error: 'Unable to read report' });
  }
});

async function listDatasets() {
  const entries = await fs.readdir(DATASET_DIR);
  return entries.filter((name) => name.endsWith('.npz'));
}

async function readJsonIfExists(filePath) {
  try {
    const payload = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(payload);
  } catch (error) {
    if (error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

function parseCsv(content) {
  const lines = content.trim().split(/\r?\n/);
  if (!lines.length) {
    return { header: [], rows: [] };
  }
  const header = lines[0].split(',');
  const rows = lines.slice(1).map((line) => line.split(','));
  return { header, rows };
}

async function buildFeatureSummary() {
  try {
    const csvRaw = await fs.readFile(FEATURE_DATASET, 'utf-8');
    const { header, rows } = parseCsv(csvRaw);
    const labelIdxCol = header.indexOf('label_idx');
    const labelCol = header.indexOf('label');
    if (labelIdxCol === -1 || labelCol === -1) {
      throw new Error('Feature CSV missing label/label_idx columns');
    }

    const featureColumns = header.filter((name) => !['sample_index', 'source', 'label_idx', 'label'].includes(name));

    const totals = {};
    const labelMap = new Map();
    rows.forEach((columns) => {
      if (!columns.length || columns.length < header.length) {
        return;
      }
      const idx = Number.parseInt(columns[labelIdxCol], 10);
      const labelName = columns[labelCol];
      if (!Number.isNaN(idx)) {
        labelMap.set(idx, labelName);
        const key = labelName || `class_${idx}`;
        totals[key] = (totals[key] || 0) + 1;
      }
    });

    const splitPayload = await readJsonIfExists(FEATURE_SPLITS);
    const splits = {};
    if (splitPayload) {
      for (const [splitName, items] of Object.entries(splitPayload)) {
        const indices = Array.isArray(items) && items.length && typeof items[0] === 'object'
          ? items.map((entry) => Number(entry.row_index))
          : items.map((value) => Number(value));
        const summary = { total: indices.length };
        indices.forEach((idx) => {
          if (!Number.isInteger(idx) || idx < 0 || idx >= rows.length) {
            return;
          }
          const row = rows[idx];
          if (!row) {
            return;
          }
          const labelName = row[labelCol];
          summary[labelName] = (summary[labelName] || 0) + 1;
        });
        splits[splitName] = summary;
      }
    }

    const report = await readJsonIfExists(FEATURE_REPORT);

    return {
      dataset: path.basename(FEATURE_DATASET),
      samples: rows.length,
      featureCount: featureColumns.length,
      featureColumns,
      topFeatureSubset: TOP_FEATURES,
      totals,
      splits,
      labelNames: report?.label_names || Array.from(labelMap.entries()).sort((a, b) => a[0] - b[0]).map(([, name]) => name),
      classWeight: report?.class_weight || null,
      evaluation: report?.evaluation || null,
      thresholdTuning: report?.threshold_tuning || null,
    };
  } catch (error) {
    console.error('Failed to build feature summary:', error);
    return null;
  }
}

function runInspect(datasetName, index, options) {
  return new Promise((resolve, reject) => {
    const datasetPath = path.join(DATASET_DIR, datasetName);
    const args = [
      INSPECT_SCRIPT,
      datasetPath,
      '--indices',
      String(index),
      '--json',
    ];

    if (options.maxPoints) {
      args.push('--max-points', String(options.maxPoints));
    }
    if (options.maxFreqs) {
      args.push('--max-freqs', String(options.maxFreqs));
    }
    if (options.stride && options.stride !== 1) {
      args.push('--stride', String(options.stride));
    }

    const child = spawn(PYTHON_BIN, args, {
      cwd: path.resolve(__dirname, '..'),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(stderr || `inspect_dataset exited with code ${code}`));
      }
      try {
        const payload = JSON.parse(stdout);
        resolve(payload);
      } catch (error) {
        reject(new Error(`Failed to parse inspect_dataset output: ${error.message}`));
      }
    });
  });
}

app.get('/api/datasets', async (req, res) => {
  try {
    const datasets = await listDatasets();
    res.json({ datasets });
  } catch (error) {
    console.error('Failed to list datasets:', error);
    res.status(500).json({ error: 'Unable to list datasets' });
  }
});

app.get('/api/datasets/:dataset/samples', async (req, res) => {
  try {
    const rawName = req.params.dataset;
    const datasetName = path.basename(rawName);
    const datasets = await listDatasets();
    const effectiveName = datasetName.endsWith('.npz') ? datasetName : `${datasetName}.npz`;
    if (!datasets.includes(effectiveName)) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    const indexParam = req.query.index;
    if (indexParam === undefined) {
      return res.status(400).json({ error: 'Query parameter "index" is required' });
    }
    const index = Number(indexParam);
    if (!Number.isInteger(index) || index < 0) {
      return res.status(400).json({ error: 'Query parameter "index" must be a non-negative integer' });
    }

    const maxPoints = req.query.maxPoints ? Number(req.query.maxPoints) : 2000;
    const maxFreqs = req.query.maxFreqs ? Number(req.query.maxFreqs) : 4096;
    const stride = req.query.stride ? Number(req.query.stride) : 40;

    const payload = await runInspect(effectiveName, index, {
      maxPoints: Number.isInteger(maxPoints) && maxPoints > 0 ? maxPoints : 2000,
      maxFreqs: Number.isInteger(maxFreqs) && maxFreqs > 0 ? maxFreqs : 4096,
      stride: Number.isInteger(stride) && stride > 0 ? stride : 40,
    });

    res.json(payload);
  } catch (error) {
    console.error('Failed to extract dataset sample:', error);
    res.status(500).json({ error: 'Unable to extract dataset sample' });
  }
});

app.get('/api/model/summary', async (req, res) => {
  try {
    const summary = await buildFeatureSummary();
    if (!summary) {
      return res.status(404).json({ error: 'Feature summary unavailable' });
    }
    res.json(summary);
  } catch (error) {
    console.error('Failed to provide model summary:', error);
    res.status(500).json({ error: 'Unable to provide model summary' });
  }
});

app.listen(PORT, () => {
  console.log(`Report viewer running on http://localhost:${PORT}`);
});
