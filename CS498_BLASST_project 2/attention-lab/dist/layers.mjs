const groups = ['T->T', 'T->V', 'V->T', 'V->V', 'mixed'];
const labels = {'T->T':'T → T', 'T->V':'T → V', 'V->T':'V → T', 'V->V':'V → V', mixed:'Mixed'};
const colors = ['#2459c4', '#08776c', '#a05b22', '#803aac', '#64748b'];
const percent = value => value == null ? '—' : new Intl.NumberFormat(undefined, {
  style:'percent', maximumFractionDigits:1,
}).format(value);
const unique = values => [...new Set(values)].sort((a,b) => a-b);
const decisionMetrics = {
  skip_persistence: 'Earlier skips still skipped', agreement: 'Same decision',
  later_skips_recovered: 'Later skips covered',
};
const indexMetrics = {
  same_all_indices: 'Same query and both keys', same_query: 'Same decisive query',
  same_tile_pair: 'Same query and tile key', same_running_pair: 'Same query and running key',
  false_skip_rate: 'First-layer replay · false skip rate',
  missed_skip_rate: 'First-layer replay · missed skip rate',
};
const rowMetrics = {
  'row.key_match_rate': 'Per-row key match · previous layer',
  'row.max_score_hit_rate': 'Per-row max score recovered · previous layer',
  'row.false_skip_rate': 'Row-key replay · false skip rate',
  'row.missed_skip_rate': 'Row-key replay · missed skip rate',
};
const isRowMetric = metric => metric.startsWith('row.');
const metricValue = (point, metric) => isRowMetric(metric)
  ? point.row_key_replay?.[metric.slice(4)]
  : point[metric] ?? point.anchor_replay?.[metric];
const logit = value => value == null ? '—' : value.toFixed(3);

export const isLayerStudy = data => data?.schema === 'blasst-tile-layer-study-v1';

function options(select, values, label = String) {
  const previous = select.value;
  select.replaceChildren(...values.map(value => new Option(label(value), value)));
  if (values.some(value => String(value) === previous)) select.value = previous;
}

function row(section, values, tag = 'td') {
  const tr = section.insertRow();
  for (const value of values) {
    const cell = document.createElement(tag);
    cell.textContent = value;
    tr.append(cell);
  }
  return tr;
}

function similarityChart(svg, data, metric) {
  svg.replaceChildren();
  function draw(name, attrs, content, parent = svg) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
    if (content !== undefined) node.textContent = content;
    parent.append(node);
    return node;
  }
  const pointsFor = group => isRowMetric(metric)
    ? (data[group]?.by_layer_pair ?? []).filter(point => point.row_key_replay)
      .map(point => ({...point, lag:point.later}))
    : data[group]?.by_lag ?? [];
  const lags = unique(pointsFor('all').map(point => point.lag));
  if (!lags.length) {
    draw('text', {x:52, y:46, 'font-size':14}, 'No adjacent layers were recorded.');
    return;
  }
  const x = lag => 52 + 674 * (lags.length === 1 ? .5 : (lag - lags[0]) / (lags.at(-1) - lags[0]));
  const y = val => 166 - 140 * val;
  for (const val of [0, .25, .5, .75, 1]) {
    draw('line', {x1:52, x2:726, y1:y(val), y2:y(val), stroke:'#d7e0e9'});
    draw('text', {x:45, y:y(val)+4, 'text-anchor':'end', 'font-size':12}, percent(val));
  }
  for (const lag of lags) {
    draw('text', {x:x(lag), y:185, 'text-anchor':'middle', 'font-size':11}, lag);
  }
  draw('text', {x:389, y:205, 'text-anchor':'middle', 'font-size':12},
    isRowMetric(metric) ? 'Target layer · reuse from previous layer' : 'Layer distance');
  groups.forEach((group, index) => {
    const points = pointsFor(group);
    if (!points.length) return;
    const color = colors[index];
    draw('text', {x:52+112*index, y:225, fill:color, 'font-size':13}, labels[group]);
    let previous;
    for (const point of points) {
      const value = metricValue(point, metric);
      if (value == null) {
        previous = undefined;
        continue;
      }
      if (previous?.lag === point.lag - 1) {
        draw('line', {x1:x(previous.lag), y1:y(metricValue(previous, metric)), x2:x(point.lag),
          y2:y(value), stroke:color, 'stroke-width':2});
      }
      const dot = draw('circle', {cx:x(point.lag), cy:y(value), r:4, fill:color});
      draw('title', {},
        `${labels[group]} · ${isRowMetric(metric) ? 'target layer' : 'distance'} ${point.lag} · ${percent(value)}`, dot);
      previous = point;
    }
  });
}

export function setupLayerStudy(showView) {
  const $ = id => document.getElementById(`layers-${id}`);
  const datasets = [];
  const current = () => datasets[Number($('dataset').value)]?.data;
  const phase = () => current()?.phases[$('phase').value];
  const threshold = () => current().metadata.calibration?.thresholds[$('phase').value] ?? current().metadata;
  const keepLabel = counts => `${percent(counts.keep_rate)} keep (${counts.kept}/${counts.tiles} tiles)`;

  function renderTile() {
    const data = current();
    const sample = data.cases[Number($('case').value)];
    const tiles = sample.tiles.filter(tile => tile[0] === $('phase').value && tile[7] === $('group').value);
    options($('head'), unique(tiles.map(tile => tile[2])));
    const headTiles = tiles.filter(tile => tile[2] === Number($('head').value));
    options($('step'), unique(headTiles.map(tile => tile[1])), value => value + 1);
    $('step-label').hidden = $('phase').value !== 'decode';
    const candidates = headTiles.filter(tile => tile[1] === Number($('step').value));
    const coordinate = tile => `${tile[3]}:${tile[4]}`;
    const choices = new Map(candidates.map(tile => [coordinate(tile), tile]));
    options($('tile'), [...choices.keys()], key => {
      const [, , , q, k, rows, cols] = choices.get(key);
      return `Q ${q}–${q+rows-1} / K ${k}–${k+cols-1}`;
    });
    const tile = choices.get($('tile').value);
    const table = $('timeline');
    table.replaceChildren();
    const cutoff = threshold().cutoff;
    const caseKeep = sample.keep_rates?.[$('phase').value]?.[$('group').value];
    $('case-keep').hidden = !caseKeep;
    $('case-keep').textContent = caseKeep ? `This case and tile group · ${keepLabel(caseKeep)}` : '';
    $('query-label').hidden = !tile || !data.metadata.row_key_replay;
    $('tile-status').textContent = tile
      ? `Tile skips when margin < ${cutoff.toFixed(3)}. Token positions and layer numbers start at zero.`
      : 'This case has no tiles for the selected phase and group.';
    if (!tile) return;
    row(table.createTHead(), ['Layer', ...sample.layers], 'th');
    const body = table.createTBody();
    const margins = tile[8];
    const decisions = margins.map(value => value === null ? '—' : value < cutoff ? 'Skip' : 'Keep');
    const decisionRow = row(body, ['Decision', ...decisions]);
    decisions.forEach((decision, index) => decisionRow.cells[index+1].dataset.decision = decision);
    row(body, ['Margin', ...margins.map(logit)]);
    const indices = tile[9];
    if (indices) {
      row(body, ['Decisive query', ...indices.map(entry => entry?.[0] ?? '—')]);
      row(body, ['Tile-max key', ...indices.map(entry => entry?.[1] ?? '—')]);
      row(body, ['Running-max key', ...indices.map(entry => entry?.[2] ?? '—')]);
      row(body, ['Tile-max score', ...indices.map(entry => logit(entry?.[3]))]);
      row(body, ['Running-max score', ...indices.map(entry => logit(entry?.[4]))]);
      row(body, [`Layer ${sample.layers[0]} replay margin`, ...indices.map(entry => logit(entry?.[5]))]);
      row(body, ['Replay decision', ...indices.map(entry =>
        entry == null ? '—' : entry[5] < cutoff ? 'Skip' : 'Keep')]);
    }
    if (data.metadata.row_key_replay && indices) {
      const qStart = tile[3];
      const kStart = tile[4];
      options($('query'), Array.from({length:tile[5]}, (_, offset) => qStart + offset));
      const offset = 2 * (Number($('query').value) - qStart);
      const keyFor = entry => {
        const key = Number.parseInt(entry?.[6]?.slice(offset, offset + 2), 16);
        return key < 255 ? kStart + key : '—';
      };
      row(body, ['Selected row · actual best key', ...indices.map(keyFor)]);
      row(body, ['Selected row · inherited key', ...indices.map((entry, index) =>
        entry?.[7] ? keyFor(indices[index - 1]) : '—')]);
      row(body, ['Rows with same key', ...indices.map(entry =>
        entry?.[7] ? `${entry[7][0]} / ${entry[7][2]}` : '—')]);
      row(body, ['Previous-layer row-key replay margin', ...indices.map(entry => logit(entry?.[7]?.[3]))]);
      row(body, ['Previous-layer row-key replay decision', ...indices.map(entry => {
        const margin = entry?.[7]?.[3];
        return margin == null ? '—' : margin < cutoff ? 'Skip' : 'Keep';
      })]);
    }
  }

  function renderPairs() {
    const group = phase()[$('group').value];
    const keep = current().keep_rates?.[$('phase').value]?.[$('group').value];
    $('group-keep').hidden = !keep;
    $('group-keep').textContent = keep ? `Selected tile group · ${keepLabel(keep)}` : '';
    const metric = $('metric').value;
    const rowMetric = isRowMetric(metric);
    const replay = rowMetric || metric === 'false_skip_rate' || metric === 'missed_skip_rate';
    $('lag').disabled = rowMetric;
    options($('lag'), rowMetric ? [1] : group.by_lag.map(point => point.lag));
    const rows = group.by_layer_pair.filter(point => point.later - point.earlier === Number($('lag').value));
    $('pairs').replaceChildren();
    const header = $('pair-head').cells;
    header[1].textContent = rowMetric ? 'Matching rows / valid rows' : replay ? 'Replay comparisons' : 'Both keep';
    header[2].textContent = replay ? 'Missed skips' : 'Keep → skip';
    header[3].textContent = replay ? 'False skips' : 'Skip → keep';
    header[4].textContent = replay ? 'Mean |margin error|' : 'Both skip';
    for (const point of rows) {
      const counts = rowMetric ? point.row_key_replay : point.anchor_replay;
      if (replay && !counts) continue;
      const cells = replay ? [rowMetric ? `${counts.matched_rows} / ${counts.row_observations}` : counts.observations,
        counts.missed_skips, counts.false_skips,
        logit(counts.mean_absolute_margin_error)] :
        [point.keep_keep, point.keep_skip, point.skip_keep, point.skip_skip];
      row($('pairs'), [`${point.earlier} → ${point.later}`, ...cells,
        percent(metricValue(point, metric)), percent(point.later_skip_rate)]);
    }
    renderTile();
  }

  function renderPhase() {
    const data = current();
    const {model, synthetic, calibration} = data.metadata;
    const cutoff = threshold();
    const lambda = calibration ? Number(cutoff.lambda.toPrecision(4)) : cutoff.lambda;
    const mask = data.metadata.attention_mask_policy === 'vv_prefix' ? ' · Bidirectional V→V' : '';
    $('source').textContent = `${synthetic ? 'Synthetic · ' : ''}${model} · ${data.pairs_completed}/${data.pairs_requested} cases · λ = ${lambda} · full scores on a dense trajectory${mask}`;
    const keep = data.keep_rates?.[$('phase').value]?.all;
    $('calibration').hidden = !calibration || !keep;
    if (calibration && keep) {
      const overlap = data.calibration_overlap_cases;
      const minimum = cutoff.target_below_minimum ? ' · Target below the attainable calibration keep rate' : '';
      $('calibration').textContent = `Target ${calibration.target_keep_percent}% keep · Calibration ${keepLabel(cutoff)} · Evaluation ${keepLabel(keep)}${minimum}${overlap ? ` · ${overlap} evaluation cases also used for calibration` : ''}`;
    }
    options($('group'), groups.filter(group => phase()[group]), group => labels[group]);
    similarityChart($('chart'), phase(), $('metric').value);
    renderPairs();
  }

  function renderDataset() {
    const data = current();
    $('content').hidden = false;
    $('index-note').hidden = !data.metadata.max_indices;
    $('row-note').hidden = !data.metadata.row_key_replay;
    const metrics = {...decisionMetrics, ...(data.metadata.max_indices ? indexMetrics : {}),
      ...(data.metadata.row_key_replay ? rowMetrics : {})};
    options($('metric'), Object.keys(metrics), id => metrics[id]);
    options($('phase'), Object.keys(data.phases).sort().reverse());
    options($('case'), data.cases.map((_, index) => index),
      index => data.cases[index].sample_id ?? `Case ${data.cases[index].sample_index + 1}`);
    renderPhase();
  }

  $('dataset').addEventListener('change', renderDataset);
  for (const id of ['phase', 'metric']) $(id).addEventListener('change', renderPhase);
  for (const id of ['group', 'lag']) $(id).addEventListener('change', renderPairs);
  for (const id of ['case', 'head', 'step', 'tile', 'query']) $(id).addEventListener('change', renderTile);

  return {
    add(name, data) {
      if (!isLayerStudy(data) || !data.cases?.length || !Object.keys(data.phases ?? {}).length) {
        throw new Error('Invalid cross-layer tile report.');
      }
      datasets.push({name, data});
      options($('dataset'), datasets.map((_, index) => index), index => datasets[index].name);
      $('dataset').value = datasets.length - 1;
      renderDataset();
      showView('layers');
    },
  };
}
