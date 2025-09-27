const listEl = document.getElementById('report-list');
const metaEl = document.getElementById('report-meta');
const detailsEl = document.getElementById('report-details');
const modelSummaryEl = document.getElementById('model-summary');
const modelSummaryContent = document.getElementById('model-summary-content');

const datasetSelect = document.getElementById('dataset-select');
const sampleInput = document.getElementById('sample-index');
const loadSampleBtn = document.getElementById('load-sample');
const sampleSummary = document.getElementById('sample-summary');
const datasetViewer = document.getElementById('dataset-viewer');
const timeCanvas = document.getElementById('time-chart');
const freqCanvas = document.getElementById('freq-chart');
const datasetSummarySection = document.getElementById('dataset-summary');
const datasetSummaryContent = document.getElementById('dataset-summary-content');
const sampleStatsSection = document.getElementById('sample-statistics');
const sampleStatsContent = document.getElementById('sample-statistics-content');

const AXIS_LABELS = { x: 'X', y: 'Y', z: 'Z' };

let timeChart;
let freqChart;

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function clearActive() {
  listEl.querySelectorAll('li').forEach((item) => item.classList.remove('active'));
}

function renderMetricsTable(metrics) {
  const rows = Object.entries(metrics)
    .map(([key, value]) => `<tr><th>${key}</th><td>${Number(value).toFixed(4)}</td></tr>`)
    .join('');
  return `<table class="metrics-table"><tbody>${rows}</tbody></table>`;
}

function formatScalar(value) {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return '—';
    }
    if (Number.isInteger(value) && Math.abs(value) < 1000) {
      return value.toString();
    }
    const absVal = Math.abs(value);
    if (absVal >= 1e6) {
      return value.toExponential(2);
    }
    if ((absVal >= 1000 && absVal < 1e6) || absVal === 0) {
      return value.toFixed(2);
    }
    if (absVal < 0.001 && absVal !== 0) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  }
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value);
}

function renderPairsTable(pairs) {
  const rows = pairs
    .map(([label, value]) => `<tr><th>${label}</th><td>${formatScalar(value)}</td></tr>`)
    .join('');
  return `<table class="metrics-table"><tbody>${rows}</tbody></table>`;
}

function renderList(items) {
  if (!items || !items.length) {
    return '<p>—</p>';
  }
  return `<ul class="inline-list">${items.map((item) => `<li>${item}</li>`).join('')}</ul>`;
}

function renderAxisCards(metrics, valueLabels) {
  const axisEntries = Object.entries(metrics || {});
  if (!axisEntries.length) {
    return '';
  }
  return `
    <div class="axis-grid">
      ${axisEntries
        .map(([axis, stats]) => {
          const rows = valueLabels
            .map(([key, label]) => [label, stats ? stats[key] : undefined]);
          return `
            <div class="axis-card">
              <h5>${AXIS_LABELS[axis] || axis.toUpperCase()}</h5>
              ${renderPairsTable(rows)}
            </div>
          `;
        })
        .join('')}
    </div>
  `;
}

function renderDatasetSummary(summary) {
  if (!summary) {
    datasetSummarySection.classList.add('hidden');
    datasetSummaryContent.innerHTML = '';
    return;
  }

  const generalPairs = [
    ['Direction', summary.direction],
    ['Samples', summary.samples],
    ['Timesteps (padded)', summary.timesteps_padded],
    ['Min length', summary.length_stats?.min],
    ['Max length', summary.length_stats?.max],
  ];

  const labelPairs = Object.entries(summary.label_counts || {}).map(([label, count]) => [label, count]);
  const generalCard = `
    <section class="metrics-subcard">
      <h4>Global Totals</h4>
      ${renderPairsTable(generalPairs)}
    </section>
  `;

  const labelsCard = labelPairs.length
    ? `
      <section class="metrics-subcard">
        <h4>Label Counts</h4>
        ${renderPairsTable(labelPairs)}
      </section>
    `
    : '';

  const axisMetrics = renderAxisCards(summary.axis_time_stats, [
    ['mean', 'Mean'],
    ['std', 'Std Dev'],
    ['rms', 'RMS'],
    ['crest_factor', 'Crest Factor'],
    ['peak_to_peak', 'Peak-to-Peak'],
    ['skewness', 'Skewness'],
    ['kurtosis', 'Kurtosis'],
    ['energy_total', 'Energy'],
  ]);
  const axisCard = axisMetrics
    ? `
      <section class="metrics-subcard">
        <h4>Axis Time Metrics</h4>
        ${axisMetrics}
      </section>
    `
    : '';

  const correlationPairs = Object.entries(summary.axis_correlations || {}).map(([pair, value]) => [pair.toUpperCase(), value]);
  const correlationsCard = correlationPairs.length
    ? `
      <section class="metrics-subcard">
        <h4>Axis Correlations</h4>
        ${renderPairsTable(correlationPairs)}
      </section>
    `
    : '';

  datasetSummaryContent.innerHTML = [generalCard, labelsCard, axisCard, correlationsCard].join('');
  datasetSummarySection.classList.remove('hidden');
}

function renderSampleStatistics(statistics) {
  if (!statistics) {
    sampleStatsSection.classList.add('hidden');
    sampleStatsContent.innerHTML = '';
    return;
  }

  const timeMetrics = statistics.time_domain
    ? renderAxisCards(statistics.time_domain, [
        ['mean', 'Mean'],
        ['std', 'Std Dev'],
        ['rms', 'RMS'],
        ['crest_factor', 'Crest Factor'],
        ['peak_to_peak', 'Peak-to-Peak'],
        ['skewness', 'Skewness'],
        ['kurtosis', 'Kurtosis'],
        ['energy', 'Energy'],
      ])
    : '';
  const timeDomainCard = timeMetrics
    ? `
      <section class="metrics-subcard">
        <h4>Time Domain</h4>
        ${timeMetrics}
      </section>
    `
    : '';

  const freqMetrics = statistics.frequency_domain
    ? renderAxisCards(statistics.frequency_domain, [
        ['peak_frequency_hz', 'Peak Freq (Hz)'],
        ['peak_magnitude_db', 'Peak Mag (dB)'],
        ['spectral_centroid_hz', 'Centroid (Hz)'],
        ['bandwidth_hz', 'Bandwidth (Hz)'],
      ])
    : '';
  const freqDomainCard = freqMetrics
    ? `
      <section class="metrics-subcard">
        <h4>Frequency Domain</h4>
        ${freqMetrics}
      </section>
    `
    : '';

  const correlationPairs = Object.entries(statistics.axis_correlations || {}).map(([pair, value]) => [pair.toUpperCase(), value]);
  const correlationCard = correlationPairs.length
    ? `
      <section class="metrics-subcard">
        <h4>Axis Correlations</h4>
        ${renderPairsTable(correlationPairs)}
      </section>
    `
    : '';

  const combined = [timeDomainCard, freqDomainCard, correlationCard].filter(Boolean).join('');
  if (!combined) {
    sampleStatsSection.classList.add('hidden');
    sampleStatsContent.innerHTML = '';
    return;
  }

  sampleStatsContent.innerHTML = combined;
  sampleStatsSection.classList.remove('hidden');
}

function renderConfusionMatrix(matrix) {
  const header = '<tr><th></th><th>Predicted Good</th><th>Predicted Bad</th></tr>';
  const [row0, row1] = matrix;
  const rows = [
    `<tr><th>Actual Good</th><td>${row0[0]}</td><td>${row0[1]}</td></tr>`,
    `<tr><th>Actual Bad</th><td>${row1[0]}</td><td>${row1[1]}</td></tr>`
  ].join('');
  return `<table class="matrix-table"><tbody>${header}${rows}</tbody></table>`;
}

function renderStandardReport(name, report) {
  const metaHtml = `<h3>${name}</h3>`;
  metaEl.innerHTML = metaHtml;
  metaEl.classList.remove('hidden');

  let content = '';
  if (report.evaluation) {
    content += '<h3>Evaluation Metrics</h3>';
    content += renderMetricsTable(report.evaluation);
  }

  if (report.predictions) {
    content += '<h3>Prediction Summary</h3>';
    content += renderMetricsTable(report.predictions);
  }

  if (report.confusion_matrix) {
    content += '<h3>Confusion Matrix</h3>';
    content += renderConfusionMatrix(report.confusion_matrix);
  }

  if (report.history) {
    content += '<h3>Training History</h3>';
    const historyRows = Object.entries(report.history)
      .map(([metric, values]) => `<tr><th>${metric}</th><td>${values.map((v) => Number(v).toFixed(4)).join(', ')}</td></tr>`)
      .join('');
    content += `<table class="history-table"><tbody>${historyRows}</tbody></table>`;
  }

  if (!content) {
    content = `<pre>${JSON.stringify(report, null, 2)}</pre>`;
  }

  detailsEl.innerHTML = content;
  detailsEl.classList.remove('hidden');
}

function renderAggregateReport(name, report) {
  const metaHtml = `<h3>${name}</h3><p>Aggregate summary for multiple models.</p>`;
  metaEl.innerHTML = metaHtml;
  metaEl.classList.remove('hidden');

  const sections = Object.entries(report)
    .map(([modelName, modelReport]) => {
      let section = `<section class="sub-card"><h4>${modelName.toUpperCase()}</h4>`;
      if (modelReport.evaluation) {
        section += '<h5>Evaluation</h5>' + renderMetricsTable(modelReport.evaluation);
      }
      if (modelReport.predictions) {
        section += '<h5>Prediction Summary</h5>' + renderMetricsTable(modelReport.predictions);
      }
      if (modelReport.confusion_matrix) {
        section += '<h5>Confusion Matrix</h5>' + renderConfusionMatrix(modelReport.confusion_matrix);
      }
      if (modelReport.history) {
        const historyRows = Object.entries(modelReport.history)
          .map(([metric, values]) => `<tr><th>${metric}</th><td>${values.map((v) => Number(v).toFixed(4)).join(', ')}</td></tr>`)
          .join('');
        section += `<h5>History</h5><table class="history-table"><tbody>${historyRows}</tbody></table>`;
      }
      section += '</section>';
      return section;
    })
    .join('');

  detailsEl.innerHTML = sections || `<pre>${JSON.stringify(report, null, 2)}</pre>`;
  detailsEl.classList.remove('hidden');
}

function renderReport(name, report) {
  metaEl.classList.add('hidden');
  detailsEl.classList.add('hidden');

  if (report && typeof report === 'object' && !Array.isArray(report)) {
    const keys = Object.keys(report);
    const looksAggregate = keys.every((key) => typeof report[key] === 'object' && !Array.isArray(report[key]));
    if (looksAggregate && !('evaluation' in report)) {
      renderAggregateReport(name, report);
      return;
    }
  }
  renderStandardReport(name, report);
}

function ensureCharts() {
  if (!timeChart) {
    timeChart = new Chart(timeCanvas.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { title: { display: true, text: 'Time (s)' } },
          y: { title: { display: true, text: 'Acceleration' } },
        },
      },
    });
  }
  if (!freqChart) {
    freqChart = new Chart(freqCanvas.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { title: { display: true, text: 'Frequency (Hz)' } },
          y: { title: { display: true, text: '|FFT|' } },
        },
      },
    });
  }
}

function updateCharts(sample) {
  ensureCharts();
  const colors = {
    x: '#0d47a1',
    y: '#2e7d32',
    z: '#c62828',
  };

  timeChart.data.labels = sample.time;
  timeChart.data.datasets = ['x', 'y', 'z'].map((axis) => ({
    label: axis.toUpperCase(),
    data: sample.signals[axis],
    borderColor: colors[axis],
    tension: 0.1,
    pointRadius: 0,
  }));
  timeChart.update();

  const freqMagnitude = sample.frequency.magnitude_db || sample.frequency.magnitude || sample.frequency.magnitude_linear;
  const freqYAxis = sample.frequency.magnitude_db ? 'Magnitude (dB)' : '|FFT|';

  freqChart.options.scales.y.title.text = freqYAxis;
  freqChart.data.labels = sample.frequency.freqs;
  freqChart.data.datasets = ['x', 'y', 'z'].map((axis) => ({
    label: axis.toUpperCase(),
    data: freqMagnitude[axis],
    borderColor: colors[axis],
    tension: 0.1,
    pointRadius: 0,
  }));
  freqChart.update();
}

async function loadSample(datasetName, index) {
  try {
    const params = new URLSearchParams({
      index: String(index),
      maxPoints: '2000',
      maxFreqs: '4096',
      stride: '40',
    });
    const payload = await fetchJSON(`/api/datasets/${encodeURIComponent(datasetName)}/samples?${params.toString()}`);
    renderDatasetSummary(payload.summary);
    if (!payload.samples.length) {
      sampleSummary.innerHTML = '<p>No sample data returned.</p>';
      renderSampleStatistics(null);
      return;
    }
    const sample = payload.samples[0];
    sampleSummary.innerHTML = `
      <p><strong>Dataset:</strong> ${payload.dataset}
      &nbsp;|&nbsp;<strong>Direction:</strong> ${payload.summary.direction}
      &nbsp;|&nbsp;<strong>Label:</strong> ${sample.label}
      &nbsp;|&nbsp;<strong>Length:</strong> ${sample.length}
      &nbsp;|&nbsp;<strong>Source:</strong> ${sample.source}</p>
    `;
    renderSampleStatistics(sample.statistics);
    updateCharts(sample);
  } catch (error) {
    sampleSummary.innerHTML = `<p class="error">Failed to load sample: ${error.message}</p>`;
    renderDatasetSummary(null);
    renderSampleStatistics(null);
  }
}

async function loadDatasets() {
  try {
    const { datasets } = await fetchJSON('/api/datasets');
    if (!datasets.length) {
      datasetViewer.classList.add('hidden');
      return;
    }
    datasetViewer.classList.remove('hidden');
    datasetSelect.innerHTML = '';
    datasets.forEach((name, idx) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      if (idx === 0) {
        option.selected = true;
      }
      datasetSelect.appendChild(option);
    });

    const initialDataset = datasetSelect.value;
    await loadSample(initialDataset, Number(sampleInput.value) || 0);
  } catch (error) {
    datasetViewer.classList.remove('hidden');
    sampleSummary.innerHTML = `<p class="error">Unable to list datasets: ${error.message}</p>`;
    renderDatasetSummary(null);
    renderSampleStatistics(null);
  }
}

loadSampleBtn.addEventListener('click', () => {
  const datasetName = datasetSelect.value;
  const index = Number(sampleInput.value) || 0;
  loadSample(datasetName, index);
});

datasetSelect.addEventListener('change', () => {
  sampleInput.value = '0';
  loadSample(datasetSelect.value, 0);
});

async function loadReports() {
  try {
    const { reports } = await fetchJSON('/api/reports');
    if (!reports.length) {
      listEl.innerHTML = '<li>No reports found.</li>';
      return;
    }
    listEl.innerHTML = '';
    reports.forEach((name, idx) => {
      const item = document.createElement('li');
      item.textContent = name;
      item.addEventListener('click', async () => {
        clearActive();
        item.classList.add('active');
        try {
          const data = await fetchJSON(`/api/reports/${encodeURIComponent(name)}`);
          renderReport(name, data);
        } catch (error) {
          detailsEl.classList.remove('hidden');
          detailsEl.innerHTML = `<p class="error">Failed to load report: ${error.message}</p>`;
        }
      });
      if (idx === 0) {
        item.classList.add('active');
        fetchJSON(`/api/reports/${encodeURIComponent(name)}`)
          .then((data) => renderReport(name, data))
          .catch((error) => {
            detailsEl.classList.remove('hidden');
            detailsEl.innerHTML = `<p class="error">Failed to load report: ${error.message}</p>`;
          });
      }
      listEl.appendChild(item);
    });
  } catch (error) {
    listEl.innerHTML = `<li class="error">Unable to list reports: ${error.message}</li>`;
  }
}

loadReports();
loadDatasets();

async function loadModelSummary() {
  try {
    const summary = await fetchJSON('/api/model/summary');
    modelSummaryEl.classList.remove('hidden');

    const totalPairs = [
      ['Dataset', summary.dataset],
      ['Total Samples', summary.samples],
      ['Feature Count', summary.featureCount],
    ];

    const thresholdPairs = [];
    if (summary.thresholdTuning?.threshold !== undefined) {
      thresholdPairs.push(['Validation Threshold', formatScalar(summary.thresholdTuning.threshold)]);
      thresholdPairs.push(['Threshold Metric', summary.thresholdTuning.metric || 'accuracy']);
    }

    if (summary.classWeight) {
      thresholdPairs.push(['Class Weight', JSON.stringify(summary.classWeight)]);
    }

    const totalsCard = `
      <section class="metrics-subcard">
        <h4>Dataset Totals</h4>
        ${renderPairsTable(totalPairs)}
        <h5>Class Distribution</h5>
        ${renderPairsTable(Object.entries(summary.totals || {}))}
      </section>
    `;

    const splitsCard = summary.splits
      ? `
        <section class="metrics-subcard">
          <h4>Stratified Splits</h4>
          ${Object.entries(summary.splits)
            .map(([split, data]) => {
              const rows = Object.entries(data);
              return `<div><h5>${split.toUpperCase()}</h5>${renderPairsTable(rows)}</div>`;
            })
            .join('')}
        </section>
      `
      : '';

    const metricsCard = summary.thresholdTuning
      ? `
        <section class="metrics-subcard">
          <h4>Validation-Tuned Metrics</h4>
          ${renderPairsTable(thresholdPairs)}
          <h5>Validation</h5>
          ${renderPairsTable(Object.entries(summary.thresholdTuning.validation || {}))}
          <h5>Test</h5>
          ${renderPairsTable(Object.entries(summary.thresholdTuning.test || {}))}
        </section>
      `
      : '';

    const featuresCard = `
      <section class="metrics-subcard">
        <h4>Feature Sets</h4>
        <p><strong>Top 10 (manual subset):</strong></p>
        ${renderList(summary.topFeatureSubset)}
        <p><strong>Extended Feature Columns (${summary.featureColumns.length}):</strong></p>
        ${renderList(summary.featureColumns)}
      </section>
    `;

    modelSummaryContent.innerHTML = `${totalsCard}${splitsCard}${metricsCard}${featuresCard}`;
  } catch (error) {
    modelSummaryEl.classList.remove('hidden');
    modelSummaryContent.innerHTML = `<p class="error">Unable to load model summary: ${error.message}</p>`;
  }
}

loadModelSummary();
