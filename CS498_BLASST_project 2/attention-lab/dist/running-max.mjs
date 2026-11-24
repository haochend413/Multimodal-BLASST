/** The same physical tile boundaries and contiguous key runs as original_reset. */
export function runningMaxTrace(scores, keyLength, visualSpans, tileSize) {
  const boundaries = [0, ...visualSpans.flat().filter(k => 0 < k && k < keyLength), keyLength];
  const runs = [];
  let shared = -Infinity;
  for (let index = 0; index < boundaries.length - 1; index++) {
    const start = boundaries[index], end = boundaries[index + 1];
    if (start === end) continue;
    const visual = visualSpans.some(([lo, hi]) => lo <= start && start < hi);
    const run = {start, end, kind: visual ? 'V' : 'T', points:[]};
    let maximum = -Infinity, maxKey;
    for (let lo = start; lo < Math.min(end, scores.length);) {
      const hi = Math.min(end, (Math.floor(lo / tileSize) + 1) * tileSize, scores.length);
      let segmentMax = -Infinity;
      for (let key = lo; key < hi; key++) {
        const score = scores[key];
        if (!Number.isFinite(score)) continue;
        if (score > maximum) {
          maximum = score;
          maxKey = key;
        }
        segmentMax = Math.max(segmentMax, score);
      }
      if (Number.isFinite(segmentMax)) {
        shared = Math.max(shared, segmentMax);
        run.points.push({start:lo, key:hi - 1, maximum, maxKey, shared});
      }
      lo = hi;
    }
    runs.push(run);
  }
  return runs;
}

export function drawRunningMax(svg, runs, tile) {
  svg.replaceChildren();
  const points = runs.flatMap(run => run.points);
  function draw(tag, attrs, label) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (label !== undefined) node.textContent = label;
    svg.append(node);
    return node;
  }
  svg.hidden = !points.length;
  if (!points.length) return;
  const start = 66, end = 752, top = 12, bottom = 130;
  const keyLimit = Math.max(tile.k + tile.cols - 1, points.at(-1).key, 1);
  const values = points.flatMap(point => [point.maximum, point.shared]);
  const min = Math.min(...values), max = Math.max(...values);
  const x = key => start + (end - start) * key / keyLimit;
  const y = score => max === min ? (top + bottom) / 2 :
    bottom - 8 - (bottom - top - 16) * (score - min) / (max - min);
  draw('rect', {x:x(tile.k), y:top, width:Math.max(2, x(tile.k + tile.cols - 1) - x(tile.k)),
    height:bottom-top, fill:'#eaf0f8'});
  for (const score of new Set([min, (min + max) / 2, max])) {
    draw('line',{x1:start, x2:end, y1:y(score), y2:y(score), stroke:'#d9e1eb'});
    draw('text',{x:start-8, y:y(score)+4, 'text-anchor':'end','font-size':12}, Number(score.toPrecision(4)));
  }
  function line(series, field, color, dash) {
    if (!series.length) return;
    let path = `M ${x(series[0].key)} ${y(series[0][field])}`;
    for (const point of series.slice(1)) path += ` H ${x(point.key)} V ${y(point[field])}`;
    draw('path', {d:path, fill:'none', stroke:color, 'stroke-dasharray':dash, 'stroke-width':2});
    for (const point of series) {
      const dot = draw('circle',{cx:x(point.key), cy:y(point[field]), r:3, fill:color});
      const tip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      tip.textContent = `Through key ${point.key} · ${point[field].toPrecision(5)}`;
      dot.append(tip);
    }
  }
  line(points, 'shared', '#64748b', '5 3');
  for (const run of runs) line(run.points, 'maximum', run.kind === 'V' ? '#08776c' : '#2459c4', 'none');
  draw('text',{x:start, y:150, 'font-size':12}, '0');
  draw('text',{x:end, y:150, 'text-anchor':'end','font-size':12}, keyLimit);
  draw('text',{x:(start+end)/2, y:150, 'text-anchor':'middle','font-size':12}, 'Key position');
}
