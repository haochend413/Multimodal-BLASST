const schema = 'blasst-jl-sweep-v1';
const regionNames = {
  'T->T': 'Text → text', 'T->V': 'Text → visual',
  'V->T': 'Visual → text', 'V->V': 'Visual → visual',
};
const regions = Object.keys(regionNames);
const groups = ['all', ...regions, 'mixed'];
const fields = ['tiles', 'full_skips', 'projected_skips', 'matched_skips',
  'false_skips', 'missed_skips', 'pairs_observed', 'pairs_with_false_skips'];
const $ = id => document.getElementById(id);
const count = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, {style: 'percent', maximumFractionDigits: 1});
const total = bins => bins.reduce((a, b) => a + b, 0);
const ratio = (n, d) => d ? n / d : null;
const validCount = n => Number.isSafeInteger(n) && n >= 0;

export function isJLSweep(data) {
  return data?.schema === schema;
}

export function validateJLSweep(data) {
  if (!isJLSweep(data) || typeof data.model !== 'string' || typeof data.synthetic !== 'boolean' ||
      !validCount(data.pairs_completed) || !data.pairs_completed ||
      (data.pairs_requested !== undefined &&
       (!validCount(data.pairs_requested) || data.pairs_requested < data.pairs_completed)) ||
      !Number.isFinite(data.lambda) || data.lambda <= 0 || data.lambda > 1 ||
      !Number.isInteger(data.head_width) || !Array.isArray(data.dimensions) || !data.dimensions.length ||
      data.dimensions.some((d, i) => !Number.isInteger(d) || d <= 0 || d >= data.head_width ||
        (i > 0 && d <= data.dimensions[i - 1])) ||
      data.histogram?.bin_width !== 1 || data.histogram?.overflow_from !== 80) {
    throw new Error('Invalid JL sweep report.');
  }
  validatePhases(data.phases, data.dimensions, data.pairs_completed);
  if (data.cases !== undefined) {
    if (!Array.isArray(data.cases) || data.cases.length !== data.pairs_completed) {
      throw new Error('JL sweep case count does not match the completed pairs.');
    }
    const indices = new Set();
    for (const item of data.cases) {
      if (!item || !validCount(item.sample_index) || indices.has(item.sample_index) ||
          (item.sample_id !== null && typeof item.sample_id !== 'string')) {
        throw new Error('Invalid JL sweep case.');
      }
      indices.add(item.sample_index);
      validatePhases(item.phases, data.dimensions, 1);
    }
  }
  return data;
}

function validatePhases(results, dimensions, pairs) {
  const phases = Object.entries(results || {});
  if (!phases.length) throw new Error('JL sweep contains no phases.');
  for (const [phase, result] of phases) {
    if (!['prefill', 'decode'].includes(phase)) throw new Error('Unknown JL sweep phase.');
    for (const region of regions) {
      const full = result.full_gaps?.[region];
      if (!Array.isArray(full) || full.length !== 81 || !full.every(validCount)) {
        throw new Error('Invalid full-score gap histogram.');
      }
      for (const dim of dimensions) {
        const gaps = result.projections?.[dim]?.gaps?.[region];
        if (!Array.isArray(gaps) || gaps.length !== 81 || !gaps.every(validCount) || total(gaps) !== total(full)) {
          throw new Error('Invalid projected-score gap histogram.');
        }
      }
    }
    for (const dim of dimensions) {
      for (const group of groups) {
        const value = result.projections[dim].decisions?.[group];
        if (!value || !fields.every(field => validCount(value[field])) ||
            value.matched_skips + value.false_skips !== value.projected_skips ||
            value.matched_skips + value.missed_skips !== value.full_skips ||
            value.full_skips + value.false_skips > value.tiles ||
            value.pairs_with_false_skips > value.pairs_observed ||
            value.pairs_observed > pairs) {
          throw new Error('Invalid JL skip counts.');
        }
      }
    }
  }
}

function svgElement(parent, tag, attrs, content) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (content !== undefined) node.textContent = content;
  parent.append(node);
  return node;
}

function graph(width, height, label, tickValues, max, xTitle) {
  const svg = svgElement(document.createDocumentFragment(), 'svg', {
    viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': label,
  });
  const plot = {left: 46, right: width - 13, top: 13, bottom: height - 38};
  const y = value => plot.bottom - (plot.bottom - plot.top) * value / max;
  for (const value of tickValues) {
    svgElement(svg, 'line', {x1: plot.left, x2: plot.right, y1: y(value), y2: y(value), stroke: 'var(--border)'});
    svgElement(svg, 'text', {x: plot.left - 6, y: y(value) + 4, 'text-anchor': 'end',
      fill: 'currentColor', 'font-size': 12}, percent.format(value));
  }
  svgElement(svg, 'text', {x: (plot.left + plot.right) / 2, y: height - 4,
    'text-anchor': 'middle', fill: 'currentColor', 'font-size': 12}, xTitle);
  return {svg, plot, y};
}

function tick(svg, plot, x, label) {
  svgElement(svg, 'text', {x, y: plot.bottom + 17, 'text-anchor': 'middle',
    fill: 'currentColor', 'font-size': 12}, label);
}

function series(svg, coordinates, color, dash) {
  svgElement(svg, 'polyline', {points: coordinates.map(p => p.join(',')).join(' '),
    fill: 'none', stroke: color, 'stroke-width': 2.4, 'stroke-dasharray': dash});
}

function cell(row, label, header = false) {
  const node = document.createElement(header ? 'th' : 'td');
  if (header) node.scope = 'row';
  node.textContent = label;
  row.append(node);
}

function asRate(numerator, denominator) {
  return denominator ? `${count.format(numerator)} / ${count.format(denominator)} · ${percent.format(numerator / denominator)}` : '—';
}

function drawRates(data, phase, group) {
  const rows = data.dimensions.map(dim => {
    const d = phase.projections[dim].decisions[group];
    return {dim, d, keeps: d.tiles - d.full_skips, false: ratio(d.false_skips, d.tiles - d.full_skips),
      missed: ratio(d.missed_skips, d.full_skips)};
  });
  const max = Math.max(.1, Math.ceil(Math.max(...rows.flatMap(row => [row.false ?? 0, row.missed ?? 0])) * 10) / 10);
  const {svg, plot, y} = graph(680, 196, 'False and missed skip rates by JL dimension',
    [0, max / 2, max], max, 'JL dimensions per head');
  const x = dim => plot.left + (plot.right - plot.left) *
    (rows.length === 1 ? .5 : (dim - rows[0].dim) / (rows.at(-1).dim - rows[0].dim));
  for (const row of rows) tick(svg, plot, x(row.dim), row.dim);
  for (const [key, label, color, dash] of [['false', 'False skips', 'var(--blue)', ''],
    ['missed', 'Missed skips', 'var(--teal)', '5 3']]) {
    let points = [];
    for (const row of rows) {
      if (row[key] === null) {
        if (points.length) series(svg, points, color, dash);
        points = [];
      } else points.push([x(row.dim), y(row[key])]);
    }
    if (points.length) series(svg, points, color, dash);
    for (const row of rows) {
      if (row[key] === null) continue;
      const point = svgElement(svg, 'circle', {cx: x(row.dim), cy: y(row[key]), r: 4, fill: color});
      svgElement(point, 'title', {}, `${row.dim} dimensions · ${label} ${percent.format(row[key])}`);
    }
  }
  $('jl-rate-chart').replaceChildren(svg);
  $('jl-rate-table').replaceChildren(...rows.map(({dim, d, keeps}) => {
    const row = document.createElement('tr');
    cell(row, count.format(dim), true);
    cell(row, asRate(d.false_skips, keeps));
    cell(row, asRate(d.missed_skips, d.full_skips));
    cell(row, d.pairs_observed ? `${d.pairs_with_false_skips} / ${d.pairs_observed}` : '—');
    return row;
  }));
}

function drawGaps(data, phase, dimension, limit) {
  const histogram = regions.map(region => {
    const full = phase.full_gaps[region], projected = phase.projections[dimension].gaps[region];
    const observations = total(full);
    const show = bins => bins.slice(0, limit === 80 ? 81 : limit).map(n => observations ? n / observations : 0);
    return {region, observations, full: show(full), projected: show(projected),
      fullOutside: total(full.slice(limit)), projectedOutside: total(projected.slice(limit))};
  });
  const max = Math.max(.02, Math.ceil(Math.max(...histogram.flatMap(h => [...h.full, ...h.projected])) * 20) / 20);
  $('jl-gap-charts').replaceChildren(...histogram.map(h => {
    const figure = document.createElement('figure');
    const title = document.createElement('h4');
    title.textContent = regionNames[h.region];
    figure.append(title);
    const caption = document.createElement('figcaption');
    if (!h.observations) {
      caption.textContent = 'No valid observations in this phase.';
      figure.append(caption);
      return figure;
    }
    const {svg, plot, y} = graph(470, 176, `${regionNames[h.region]} full and projected gap distributions`,
      [0, max / 2, max], max, 'Gap in logits');
    const x = value => plot.left + (plot.right - plot.left) * value / (limit === 80 ? 81 : limit);
    for (let i = 0; i <= 4; i++) {
      const value = limit * i / 4;
      tick(svg, plot, x(value), limit === 80 && i === 4 ? '≥80' : value);
    }
    const cutoff = -Math.log(data.lambda);
    if (cutoff <= limit) svgElement(svg, 'line', {x1: x(cutoff), x2: x(cutoff), y1: plot.top, y2: plot.bottom,
      stroke: 'var(--orange)', 'stroke-width': 1.5, 'stroke-dasharray': '2 3'});
    for (const [values, color, dash] of [[h.full, 'var(--blue)', ''], [h.projected, 'var(--teal)', '5 3']]) {
      const points = values.flatMap((value, i) => [[x(i), y(value)], [x(i + 1), y(value)]]);
      series(svg, points, color, dash);
    }
    h.full.forEach((fraction, i) => {
      const cell = svgElement(svg, 'rect', {x: x(i), width: x(i + 1) - x(i), y: plot.top,
        height: plot.bottom - plot.top, fill: 'transparent'});
      svgElement(cell, 'title', {}, `${i === 80 ? '≥80' : `[${i}, ${i + 1})`} logits · full ${percent.format(fraction)} · JL ${percent.format(h.projected[i])}`);
    });
    caption.textContent = `${count.format(h.observations)} observations · ≥${limit} logits: full ${percent.format(h.fullOutside / h.observations)}, JL ${percent.format(h.projectedOutside / h.observations)}`;
    figure.append(svg, caption);
    return figure;
  }));
}

function options(select, values, label = String) {
  const previous = select.value;
  select.replaceChildren(...values.map(value => new Option(label(value), value)));
  if (values.some(value => String(value) === previous)) select.value = previous;
}

export function setupJLStudy() {
  const datasets = [];
  const dataset = $('jl-dataset'), phase = $('jl-phase'), dimension = $('jl-dimension');
  const casePicker = $('jl-case'), previousCase = $('jl-case-prev'), nextCase = $('jl-case-next');
  function selectedPhases(data) {
    return data.cases?.[Number(casePicker.value)]?.phases ?? data.phases;
  }
  function show(active) {
    $('jl-workspace').hidden = !active;
    $('tile-workspace').hidden = active;
    $('show-jl').setAttribute('aria-pressed', active);
    $('show-tiles').setAttribute('aria-pressed', !active);
    if (!active) window.dispatchEvent(new Event('resize'));
  }
  function draw() {
    const selected = datasets[Number(dataset.value)];
    if (!selected) return;
    const data = selected.data;
    const requested = data.pairs_requested ?? data.pairs_completed;
    $('jl-source').textContent = `${data.synthetic ? 'Synthetic demo' : 'Recorded data'} · ${data.model} · ${data.pairs_completed}/${requested} pairs · width ${data.head_width} · λ ${data.lambda} · seed ${data.seed}`;
    const result = selectedPhases(data)[phase.value];
    drawRates(data, result, $('jl-group').value);
    drawGaps(data, result, dimension.value, +$('jl-window').value);
    $('jl-content').hidden = false;
  }
  function selectCase() {
    const {data} = datasets[Number(dataset.value)];
    const phases = selectedPhases(data);
    options(phase, Object.keys(phases), value => value === 'prefill' ? 'Prefill' : 'Decode');
    previousCase.disabled = casePicker.selectedIndex <= 0;
    nextCase.disabled = casePicker.selectedIndex >= casePicker.options.length - 1;
    draw();
  }
  function selectDataset() {
    const {data} = datasets[Number(dataset.value)];
    const cases = data.cases ?? [];
    options(casePicker, [-1, ...cases.keys()], index => {
      if (index === -1) return `All ${data.pairs_completed} pairs`;
      const item = cases[index];
      return `Case ${item.sample_index + 1}${item.sample_id ? ` · ${item.sample_id}` : ''}`;
    });
    casePicker.value = '-1';
    casePicker.disabled = cases.length === 0;
    $('jl-case-note').hidden = cases.length > 0;
    options(dimension, data.dimensions);
    if (data.dimensions.includes(32) && !dimension.dataset.initialized) dimension.value = '32';
    dimension.dataset.initialized = 'true';
    selectCase();
  }
  $('show-tiles').addEventListener('click', () => show(false));
  $('show-jl').addEventListener('click', () => show(true));
  dataset.addEventListener('change', selectDataset);
  casePicker.addEventListener('change', selectCase);
  previousCase.addEventListener('click', () => { casePicker.selectedIndex -= 1; selectCase(); });
  nextCase.addEventListener('click', () => { casePicker.selectedIndex += 1; selectCase(); });
  for (const control of [phase, dimension, $('jl-group'), $('jl-window')]) control.addEventListener('change', draw);
  return {
    add(name, input, activate = true) {
      const data = validateJLSweep(input);
      datasets.push({name, data});
      options(dataset, datasets.map((_, index) => index), index => datasets[index].name);
      if (activate || datasets.length === 1) {
        dataset.value = datasets.length - 1;
        selectDataset();
      }
      $('jl-error').hidden = true;
      if (activate) show(true);
    },
    demoFailed() {
      if (datasets.length) return;
      $('jl-source').textContent = 'Load a JL sweep JSON to view the charts.';
      $('jl-error').textContent = 'The synthetic demo could not be loaded.';
      $('jl-error').hidden = false;
    },
  };
}
