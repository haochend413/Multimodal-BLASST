const SCHEMA = 'blasst-jl-region-audit-v1';
const POLICIES = [
  ['original_shared', 'Original · shared max', 'var(--muted)'],
  ['original_reset', 'Original · reset max', 'var(--purple)'],
  ['packed_shared', 'Packed · shared max', 'var(--blue)'],
  ['packed_reset', 'Packed · reset max', 'var(--teal)'],
];
const GROUPS = ['all', 'T->T', 'T->V', 'V->T', 'V->V', 'mixed'];
const COUNT_KEYS = ['tiles', 'full_skips', 'projected_skips', 'matched_skips', 'false_skips', 'missed_skips'];
const WEIGHT_KEYS = ['dense_weight', 'projected_skipped_dense_weight', 'false_skip_dense_weight'];
const $ = id => document.getElementById(id);
const count = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, {style: 'percent', maximumFractionDigits: 3});
const ratio = (n, d) => d ? n / d : null;

export const isRegionAudit = data => data?.schema === SCHEMA;

function validatePhases(phases) {
  const entries = Object.entries(phases || {});
  if (!entries.length) throw new Error('Region audit contains no phases.');
  for (const [phase, result] of entries) {
    if (phase !== 'prefill' && phase !== 'decode') throw new Error('Unknown region audit phase.');
    for (const [policy] of POLICIES) {
      if (policy === 'original_reset' && result?.[policy] === undefined) continue;
      for (const group of GROUPS) {
        const row = result?.[policy]?.[group];
        if (!row || COUNT_KEYS.some(key => !Number.isSafeInteger(row[key]) || row[key] < 0) ||
            WEIGHT_KEYS.some(key => !Number.isFinite(row[key]) || row[key] < 0) ||
            row.full_skips + row.false_skips > row.tiles ||
            row.matched_skips + row.false_skips !== row.projected_skips ||
            row.matched_skips + row.missed_skips !== row.full_skips) {
          throw new Error('Invalid region audit measurements.');
        }
      }
    }
  }
}

export function validateRegionAudit(data) {
  if (!isRegionAudit(data) || typeof data.model !== 'string' || typeof data.synthetic !== 'boolean' ||
      !Number.isSafeInteger(data.pairs_completed) || data.pairs_completed < 1 ||
      !Number.isInteger(data.dimension) || data.dimension <= 0 ||
      !Number.isInteger(data.head_width) || data.head_width <= data.dimension ||
      !Number.isFinite(data.lambda) || data.lambda <= 0 || data.lambda > 1) {
    throw new Error('Invalid region audit file.');
  }
  validatePhases(data.phases);
  if (data.cases !== undefined) {
    if (!Array.isArray(data.cases) || data.cases.length !== data.pairs_completed) {
      throw new Error('Region audit case count does not match the completed pairs.');
    }
    for (const item of data.cases) {
      if (!item || !Number.isSafeInteger(item.sample_index) || item.sample_index < 0 ||
          (item.sample_id !== null && typeof item.sample_id !== 'string')) {
        throw new Error('Invalid region audit case.');
      }
      validatePhases(item.phases);
    }
  }
  return data;
}

function selectOptions(el, entries, initial = el.value) {
  el.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  if (entries.some(([value]) => String(value) === String(initial))) el.value = initial;
}

function barChart(title, rows, getValue, format) {
  const chart = document.createElement('figure');
  chart.className = 'region-audit-chart';
  const caption = document.createElement('figcaption');
  caption.textContent = title;
  chart.append(caption);
  const values = rows.map(row => getValue(row.data));
  const max = Math.max(...values, 0) || 1;
  rows.forEach((row, index) => {
    const line = document.createElement('div');
    line.className = 'region-audit-bar';
    const label = document.createElement('span');
    label.textContent = row.label;
    const track = document.createElement('span');
    track.className = 'region-audit-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('i');
    fill.style.width = `${100 * (values[index] ?? 0) / max}%`;
    fill.style.setProperty('--bar-color', row.color);
    track.append(fill);
    const value = document.createElement('strong');
    value.textContent = values[index] === null ? '—' : format(values[index]);
    line.append(label, track, value);
    chart.append(line);
  });
  return chart;
}

function rate(n, d) {
  return d ? `${count.format(n)} / ${count.format(d)} · ${percent.format(n / d)}` : '—';
}

function drawTable(rows) {
  $('region-table').replaceChildren(...rows.map(row => {
    const d = row.data;
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.scope = 'row';
    th.textContent = row.label;
    tr.append(th);
    for (const value of [count.format(d.full_skips), count.format(d.projected_skips),
      rate(d.false_skips, d.tiles - d.full_skips), rate(d.missed_skips, d.full_skips)]) {
      tr.insertCell().textContent = value;
    }
    return tr;
  }));
}

export function setupRegionAudit(show) {
  const datasets = [];
  const dataset = $('region-dataset'), casePicker = $('region-case');
  const phase = $('region-phase'), group = $('region-group');
  const selected = () => datasets[Number(dataset.value)]?.data;
  const selectedPhases = data => data.cases?.[Number(casePicker.value)]?.phases ?? data.phases;
  function draw() {
    const data = selected();
    if (!data) return;
    const phaseData = selectedPhases(data)[phase.value];
    const rows = POLICIES.filter(([key]) => phaseData[key]).map(([key, label, color]) =>
      ({label, color, data: phaseData[key][group.value]}));
    $('region-charts').replaceChildren(
      barChart('Tile decisions', rows, d => d.tiles, n => count.format(n)),
      barChart('JL skip rate', rows, d => ratio(d.projected_skips, d.tiles), n => percent.format(n)),
      barChart('Dense attention weight dropped', rows, d => ratio(d.projected_skipped_dense_weight, d.dense_weight),
        n => percent.format(n)),
      barChart('Dense weight on false skips', rows, d => ratio(d.false_skip_dense_weight, d.dense_weight),
        n => percent.format(n)),
    );
    drawTable(rows);
    const all = group.value === 'all';
    const original = phaseData.original_shared[group.value].tiles;
    const packed = phaseData.packed_shared[group.value].tiles;
    $('region-status').textContent = all && original
      ? `Packing adds ${count.format(packed - original)} tile decisions (${percent.format((packed - original) / original)}). Resetting uses the same tiles within each layout.`
      : 'Each bar chart compares the three policies within the selected tile group.';
    $('region-content').hidden = false;
  }
  function selectCase() {
    selectOptions(phase, Object.keys(selectedPhases(selected())).map(value =>
      [value, value === 'prefill' ? 'Prefill' : 'Decode']));
    draw();
  }
  function selectDataset() {
    const data = selected();
    $('region-source').textContent = `${data.synthetic ? 'Synthetic demo' : 'Recorded data'} · ${data.model} · ${data.pairs_completed} pairs · ${data.dimension} dimensions · λ ${data.lambda}`;
    const cases = (data.cases ?? []).map((item, index) =>
      [index, `Case ${item.sample_index + 1}${item.sample_id ? ` · ${item.sample_id}` : ''}`]);
    selectOptions(casePicker, [[-1, 'All pairs'], ...cases], -1);
    casePicker.disabled = !cases.length;
    selectCase();
  }
  dataset.addEventListener('change', selectDataset);
  casePicker.addEventListener('change', selectCase);
  phase.addEventListener('change', draw);
  group.addEventListener('change', draw);
  return {
    add(name, input, activate = true) {
      datasets.push({name, data: validateRegionAudit(input)});
      selectOptions(dataset, datasets.map((item, index) => [index, item.name]));
      if (activate || datasets.length === 1) {
        dataset.value = datasets.length - 1;
        selectDataset();
      }
      if (activate) show('region');
    },
  };
}
