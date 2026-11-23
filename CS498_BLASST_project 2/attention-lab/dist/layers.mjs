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
const metricValue = (point, metric) => point[metric] ?? point.anchor_replay?.[metric];
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
  const lags = unique(data.all.by_lag.map(point => point.lag));
  const x = lag => 52 + 674 * (lags.length === 1 ? .5 : (lag - lags[0]) / (lags.at(-1) - lags[0]));
  const y = val => 166 - 140 * val;
  for (const val of [0, .25, .5, .75, 1]) {
    draw('line', {x1:52, x2:726, y1:y(val), y2:y(val), stroke:'#d7e0e9'});
    draw('text', {x:45, y:y(val)+4, 'text-anchor':'end', 'font-size':12}, percent(val));
  }
  for (const lag of lags) {
    draw('text', {x:x(lag), y:185, 'text-anchor':'middle', 'font-size':11}, lag);
  }
  draw('text', {x:389, y:205, 'text-anchor':'middle', 'font-size':12}, 'Layer distance');
  groups.forEach((group, index) => {
    const points = data[group]?.by_lag;
    if (!points) return;
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
      draw('title', {}, `${labels[group]} · distance ${point.lag} · ${percent(value)}`, dot);
      previous = point;
    }
  });
}

export function setupLayerStudy(showView) {
  const $ = id => document.getElementById(`layers-${id}`);
  const datasets = [];
  const current = () => datasets[Number($('dataset').value)]?.data;
  const phase = () => current()?.phases[$('phase').value];

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
    $('tile-status').textContent = tile
      ? `Tile skips when margin < ${data.metadata.cutoff.toFixed(3)}. Token positions and layer numbers start at zero.`
      : 'This case has no tiles for the selected phase and group.';
    if (!tile) return;
    row(table.createTHead(), ['Layer', ...sample.layers], 'th');
    const body = table.createTBody();
    const margins = tile[8];
    const decisions = margins.map(value => value === null ? '—' : value < data.metadata.cutoff ? 'Skip' : 'Keep');
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
        entry == null ? '—' : entry[5] < data.metadata.cutoff ? 'Skip' : 'Keep')]);
    }
  }

  function renderPairs() {
    const group = phase()[$('group').value];
    options($('lag'), group.by_lag.map(point => point.lag));
    const rows = group.by_layer_pair.filter(point => point.later - point.earlier === Number($('lag').value));
    $('pairs').replaceChildren();
    const metric = $('metric').value;
    const replay = metric === 'false_skip_rate' || metric === 'missed_skip_rate';
    const header = $('pair-head').cells;
    header[1].textContent = replay ? 'Replay comparisons' : 'Both keep';
    header[2].textContent = replay ? 'Missed skips' : 'Keep → skip';
    header[3].textContent = replay ? 'False skips' : 'Skip → keep';
    header[4].textContent = replay ? 'Mean |margin error|' : 'Both skip';
    for (const point of rows) {
      const counts = point.anchor_replay;
      if (replay && !counts) continue;
      const cells = replay ? [counts.observations, counts.missed_skips, counts.false_skips,
        logit(counts.mean_absolute_margin_error)] :
        [point.keep_keep, point.keep_skip, point.skip_keep, point.skip_skip];
      row($('pairs'), [`${point.earlier} → ${point.later}`, ...cells,
        percent(metricValue(point, metric)), percent(point.later_skip_rate)]);
    }
    renderTile();
  }

  function renderPhase() {
    options($('group'), groups.filter(group => phase()[group]), group => labels[group]);
    similarityChart($('chart'), phase(), $('metric').value);
    renderPairs();
  }

  function renderDataset() {
    const data = current();
    const {model, synthetic, lambda} = data.metadata;
    $('source').textContent = `${synthetic ? 'Synthetic · ' : ''}${model} · ${data.pairs_completed}/${data.pairs_requested} cases · λ = ${lambda} · full scores on a dense trajectory`;
    $('content').hidden = false;
    $('index-note').hidden = !data.metadata.max_indices;
    const metrics = data.metadata.max_indices ? {...decisionMetrics, ...indexMetrics} : decisionMetrics;
    options($('metric'), Object.keys(metrics), id => metrics[id]);
    options($('phase'), Object.keys(data.phases).sort().reverse());
    options($('case'), data.cases.map((_, index) => index),
      index => data.cases[index].sample_id ?? `Case ${data.cases[index].sample_index + 1}`);
    renderPhase();
  }

  $('dataset').addEventListener('change', renderDataset);
  for (const id of ['phase', 'metric']) $(id).addEventListener('change', renderPhase);
  for (const id of ['group', 'lag']) $(id).addEventListener('change', renderPairs);
  for (const id of ['case', 'head', 'step', 'tile']) $(id).addEventListener('change', renderTile);

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
