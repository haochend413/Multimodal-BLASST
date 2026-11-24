// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import {validate, summary, regions, colors, scoreFileId, readScoreChunk, wouldSkip} from './data.mjs?v=vv-prefix-skips';
import {isJLSweep, setupJLStudy} from './jl.mjs?v=region-audit';
import {isRegionAudit, setupRegionAudit} from './region.mjs?v=region-reset';
import {isLayerStudy, setupLayerStudy} from './layers.mjs?v=vv-prefix-skips';
import {runningMaxTrace, drawRunningMax} from './running-max.mjs';
import {isTVVVSweep, setupTVVVSweep} from './tv-vv.mjs?v=matched-budgets';
const $ = id => document.getElementById(id);
const workspaces = {tiles:'tile-workspace', jl:'jl-workspace', region:'region-workspace', layers:'layers-workspace', 'tv-vv':'tv-vv-workspace'};
function showView(view) {
  for (const [name, panel] of Object.entries(workspaces)) {
    $(panel).hidden = name !== view;
    $(`show-${name}`).setAttribute('aria-pressed', name === view);
  }
  if (view === 'tiles') window.dispatchEvent(new Event('resize'));
}
for (const view of Object.keys(workspaces)) $(`show-${view}`).addEventListener('click', () => showView(view));
const jlStudy = setupJLStudy(showView);
const regionAudit = setupRegionAudit(showView);
const layerStudy = setupLayerStudy(showView);
const tvVVSweep = setupTVVVSweep(showView);
const datasets = []; let data, current, region = 'All', hits = [];
const scoreFiles = new Map();
let demoScoreId, demoScores, selectedTile, selectedScores, inspection = 0;
const format = n => Number(n).toLocaleString(undefined,{maximumFractionDigits:3});
const notes = {abs:'Mean absolute scaled QK score. Large negative scores also have large magnitudes.',mean:'Signed mean scaled QK score, excluding masked future positions.',max:'Largest valid scaled QK score in each tile.',mass:'Dense softmax weight on the tile, averaged across its query rows. It is not the live sparse weight.',margin:'Maximum over query rows of (tile row max − running row max). Lower margins are easier to skip.'};
function options(id, values, label=x=>String(x)) { const el=$(id), old=el.value; el.replaceChildren(...values.map(x=>new Option(label(x),x))); if(values.map(String).includes(old)) el.value=old; }
function unique(values) { return [...new Set(values)].sort((a,b)=>a-b); }
function fail(error) { $('error').textContent=error.message; $('error').hidden=false; }
function add(name, input) { const parsed=validate(input); datasets.push({name,data:parsed}); options('dataset',datasets.map((_,i)=>i),i=>datasets[i].name); $('dataset').value=datasets.length-1; selectDataset(); }
function selectDataset() { closeTile(); data=datasets[Number($('dataset').value)].data; $('error').hidden=true; $('badge').textContent=data.metadata.synthetic?'SYNTHETIC DEMO':'RECORDED DATA'; $('model').textContent=data.metadata.model; $('trajectory').textContent=(data.metadata.recorded_trajectory==='dense'?'Dense trajectory · offline threshold exploration':'Live sparse trajectory · later inputs reflect earlier skips'); $('dataset-note').textContent=data.metadata.synthetic?'Generated from random Q/K tensors with a planted sink. These are not model results.':`${format(data.metadata.prompt_tokens)} prompt tokens · ${data.metadata.sampling}`; options('phase',[...new Set(data.views.map(v=>v.phase))],x=>x==='prefill'?'Prefill · prompt processing':'Decode · generated text query'); updateControls('phase'); }
function updateControls(changed) { let views=data.views.filter(v=>v.phase===$('phase').value); if(changed==='phase')options('layer',unique(views.map(v=>v.layer))); views=views.filter(v=>v.layer===Number($('layer').value)); if(changed==='phase'||changed==='layer')options('head',unique(views.map(v=>v.head))); views=views.filter(v=>v.head===Number($('head').value)); options('step',unique(views.map(v=>v.step)),x=>`Step ${x+1}`); $('step-wrap').hidden=$('phase').value!=='decode'; draw(); }
function view() { return data.views.find(v=>v.phase===$('phase').value && v.layer===Number($('layer').value) && v.head===Number($('head').value) && v.step===Number($('step').value)); }
function closeTile() {
  selectedTile = undefined;
  selectedScores = undefined;
  inspection += 1;
  $('tile-inspector').hidden = true;
  $('tile-score-table').replaceChildren();
}

function tokenKind(position) {
  return (data.visual_spans || []).some(([start, end]) => start <= position && position < end) ? 'V' : 'T';
}

function showScoreMatrix(tile, scores) {
  const keyKinds = scores[0].map((_, col) => tokenKind(tile.k + col));
  const table = $('tile-score-table');
  const head = document.createElement('thead').insertRow();
  const axis = document.createElement('th');
  axis.textContent = 'Query / Key';
  head.append(axis);
  function tokenHeader(position, kind, scope) {
    const label = document.createElement('th');
    label.scope = scope;
    label.dataset.kind = kind;
    label.textContent = `${position} · ${kind === 'V' ? 'Video' : 'Text'}`;
    return label;
  }
  keyKinds.forEach((kind, col) => head.append(tokenHeader(tile.k + col, kind, 'col')));
  const body = document.createElement('tbody');
  scores.forEach((values, row) => {
    const queryKind = tokenKind(tile.q + row);
    const line = body.insertRow();
    line.append(tokenHeader(tile.q + row, queryKind, 'row'));
    values.forEach((score, col) => {
      const cell = line.insertCell();
      const pair = document.createElement('small');
      pair.textContent = `${queryKind}→${keyKinds[col]}`;
      if (Number.isFinite(score)) {
        cell.textContent = Number(score.toPrecision(9));
        cell.title = String(score);
      } else {
        cell.textContent = 'Masked';
        cell.className = 'masked';
      }
      cell.append(pair);
    });
  });
  table.replaceChildren(head.parentElement, body);
  $('tile-score-scroll').hidden = false;
  $('tile-score-scroll').scrollTo(0, 0);
}

function showRunningMax() {
  const tile = selectedTile;
  const query = Number($('running-query').value);
  const runs = runningMaxTrace(selectedScores.row(query - tile.q), current.key_length,
    data.visual_spans ?? [], data.metadata.tile_size ?? 128);
  const through = tile.k + tile.cols - 1;
  const rows = document.createElement('tbody');
  const display = value => value === undefined ? '—' : format(value);
  for (const [index, run] of runs.entries()) {
    const before = run.points.findLast(point => point.key < tile.k);
    const after = run.points.findLast(point => point.key <= through);
    const cells = [`${tokenKind(query)} → ${run.kind} · run ${index + 1}`,
      `${run.start}–${run.end - 1}`, display(before?.maximum), display(after?.maximum),
      display(after?.maxKey), display(run.points.at(-1)?.maximum)];
    const row = rows.insertRow();
    for (const value of cells) row.insertCell().textContent = value;
  }
  const shared = runs.flatMap(run => run.points).findLast(point => point.key <= through);
  $('running-max-summary').textContent = `Query ${query} · Shared running max through selected tile ${display(shared?.shared)}`;
  $('running-max-table').tBodies[0].replaceWith(rows);
  drawRunningMax($('running-max-chart'), runs, tile);
}

$('running-query').addEventListener('change', showRunningMax);

async function inspectTile(tile) {
  selectedTile = tile;
  selectedScores = undefined;
  const request = ++inspection;
  const reference = data.score_file;
  const selectedView = current;
  const status = $('tile-detail-status');
  $('tile-inspector').hidden = false;
  $('tile-score-scroll').hidden = true;
  $('running-max-panel').hidden = true;
  $('tile-score-table').replaceChildren();
  $('tile-detail-title').textContent = `Q ${tile.q}–${tile.q + tile.rows - 1} · K ${tile.k}–${tile.k + tile.cols - 1}`;
  if (!reference) {
    status.textContent = 'This export contains block summaries. Collect with --score-data to inspect individual scores.';
    return;
  }
  if (!scoreFiles.has(reference.id) && reference.id !== demoScoreId) {
    status.textContent = 'Choose the matching .scores.bin companion using Load experiment.';
    return;
  }
  status.textContent = 'Loading token scores…';
  try {
    let file = scoreFiles.get(reference.id);
    if (!file) {
      demoScores ??= fetch(`demo.scores.bin?id=${demoScoreId}`).then(async response => {
        if (!response.ok) throw new Error('Demo token scores could not be loaded.');
        const blob = await response.blob();
        if (await scoreFileId(blob) !== demoScoreId) throw new Error('Demo token scores are out of date.');
        return blob;
      }).catch(error => { demoScores = undefined; throw error; });
      file = await demoScores;
    }
    const scores = await readScoreChunk(file, reference, selectedView, tile);
    if (request !== inspection) return;
    selectedScores = scores;
    showScoreMatrix(tile, scores.tile);
    const previousQuery = $('running-query').value;
    options('running-query', scores.tile.map((_, offset) => tile.q + offset),
      query => `${query} · ${tokenKind(query) === 'V' ? 'Visual' : 'Text'}`);
    if (![...$('running-query').options].some(option => option.value === previousQuery)) {
      const firstValid = scores.tile.findIndex(row => row.some(Number.isFinite));
      $('running-query').value = tile.q + Math.max(0, firstValid);
    }
    $('running-max-panel').hidden = false;
    showRunningMax();
    status.textContent = 'Scaled QK logits before softmax · T = Text, V = Video · query → key';
  } catch (error) {
    if (request === inspection) status.textContent = error.message;
  }
}

function tileAt(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  return hits.find(a => x >= a.x && x < a.x + a.w && y >= a.y && y < a.y + a.h)?.t;
}

function openTile(tile, reveal = false) {
  inspectTile(tile);
  $('tooltip').hidden = true;
  drawMap(10 ** Number($('lambda').value));
  if (reveal) $('tile-inspector').scrollIntoView({block: 'nearest'});
}
function active(t) { if(region==='All')return true; const [q,k]=region.split('→'); return (q==='V'?t.q_visual>0:t.q_visual<1)&&(k==='V'?t.k_visual>0:t.k_visual<1); }
function color(value,min,max) { const stops=[[232,240,253],[60,134,176],[18,109,123],[237,177,76]], x=Math.max(0,Math.min(1,(value-min)/(max-min||1)))*3, i=Math.min(2,Math.floor(x)),f=x-i; return `rgb(${stops[i].map((a,j)=>Math.round(a+(stops[i+1][j]-a)*f)).join(',')})`; }
function draw() { const next = view(); if (next !== current) closeTile(); current = next; if(!current)return; const lambda=10**Number($('lambda').value), stats=summary(current,lambda); $('lambda-value').textContent=lambda<.001?lambda.toExponential(1):lambda.toPrecision(3); $('tile-count').textContent=format(stats.tiles); $('skip-count').textContent=`${(100*stats.skipped/stats.tiles).toFixed(1)}%`; $('score-stat').textContent=format(stats.abs); $('key-count').textContent=format(current.key_length); $('metric-note').textContent=notes[$('metric').value]; $('map-title').textContent=`${current.phase==='decode'?'Decode':'Prefill'} · layer ${current.layer} / head ${current.head}`; $('map-note').textContent=`${current.phase==='decode'?'Each tile contains one generated-text query and up to 128 cached keys.':data.metadata.attention_mask_policy==='vv_prefix'?'Pure V→V tiles are bidirectional. Other tiles are causal. Every visible tile uses the usual BLASST skip rule.':'Each cell groups up to 128 query tokens × 128 key tokens; future-only tiles are blank.'} Click a block to expand its token scores. Hover for block measurements. Statistics below include all four regions; highlighting only dims the map.`; drawMap(lambda); drawHist(stats); }
function drawMap(lambda) { const canvas=$('heatmap'), rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1; canvas.width=rect.width*dpr; canvas.height=rect.height*dpr; const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr); const width=rect.width,height=rect.height,pad=28, x0=pad,y0=22,w=width-pad-10,h=height-48;
 const tiles=current.tiles, metric=$('metric').value, min=tiles.reduce((a,t)=>Math.min(a,t[metric]),Infinity),max=tiles.reduce((a,t)=>Math.max(a,t[metric]),-Infinity); $('scale-min').textContent=format(min); $('scale-max').textContent=format(max);
 const vv = tiles.filter(tile => tile.q_visual === 1 && tile.k_visual === 1);
 const vvCount = $('pure-vv-count');
 vvCount.hidden = !vv.length;
 vvCount.textContent = vv.length ? `Pure V→V at this λ · ${format(vv.filter(tile => wouldSkip(tile, lambda)).length)} of ${format(vv.length)} visible tiles would skip` : '';
 const qMin=tiles.reduce((a,t)=>Math.min(a,t.q),Infinity),qMax=tiles.reduce((a,t)=>Math.max(a,t.q+t.rows),-Infinity), keyLength=current.key_length; hits=[]; ctx.clearRect(0,0,width,height); ctx.fillStyle='#f2f5f8';ctx.fillRect(x0,y0,w,h);
 for(const t of tiles){const x=x0+t.k/keyLength*w,y=y0+(t.q-qMin)/(qMax-qMin)*h,cw=t.cols/keyLength*w,ch=t.rows/(qMax-qMin)*h;ctx.globalAlpha=active(t)?1:.12;ctx.fillStyle=color(t[metric],min,max);ctx.fillRect(x,y,cw,ch);ctx.strokeStyle='#ffffff60';ctx.lineWidth=.5;ctx.strokeRect(x,y,cw,ch);
 if($('overlay').checked&&wouldSkip(t,lambda)){ctx.save();ctx.beginPath();ctx.rect(x,y,cw,ch);ctx.clip();ctx.strokeStyle='#152a3c70';ctx.lineWidth=1;for(let d=-ch;d<cw;d+=9){ctx.beginPath();ctx.moveTo(x+d,y);ctx.lineTo(x+d+ch,y+ch);ctx.stroke();}ctx.restore();}hits.push({x,y,w:cw,h:ch,t});}ctx.globalAlpha=1;
 const selection = hits.find(hit => hit.t === selectedTile);
 if (selection) {
   ctx.lineWidth = 3;
   ctx.strokeStyle = '#152a3c';
   ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
 }
 // Modality strips retain exact token span boundaries, including mixed tiles.
 ctx.fillStyle=colors[0];ctx.fillRect(x0,y0-9,w,5);ctx.fillRect(x0-9,y0,5,h);ctx.fillStyle=colors[3]; for(const [start,end] of data.visual_spans||[]){ctx.fillRect(x0+start/keyLength*w,y0-9,(Math.min(end,keyLength)-start)/keyLength*w,5);const a=Math.max(start,qMin),b=Math.min(end,qMax);if(b>a)ctx.fillRect(x0-9,y0+(a-qMin)/(qMax-qMin)*h,5,(b-a)/(qMax-qMin)*h);}
 ctx.fillStyle='#607185';ctx.font='10px system-ui';ctx.textAlign='center';for(let i=0;i<=4;i++){const position=Math.round(keyLength*i/4);ctx.fillText(format(position),x0+w*i/4,y0+h+17);}ctx.textAlign='right';ctx.fillText(format(qMin),x0-12,y0+14);ctx.fillText(format(qMax-1),x0-12,y0+h); }
function drawHist(stats) { const svg=$('histogram'); svg.replaceChildren(); const NS='http://www.w3.org/2000/svg';function el(name,attrs,text){const e=document.createElementNS(NS,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
 const w=770,h=155,x=50,y=15, hist=data.histogram;let max=.001;for(const r of regions){const s=current.regions[r];for(const n of s.hist)max=Math.max(max,s.count?n/s.count:0);} for(let i=0;i<=3;i++){const yy=y+h-i*h/3;el('line',{x1:x,x2:x+w,y1:yy,y2:yy,stroke:'#e5eaf0'});el('text',{x:x-8,y:yy+4,'text-anchor':'end',fill:'#607185','font-size':11},`${(max*i/3*100).toFixed(0)}%`);}
 for(let i=0;i<regions.length;i++){const r=regions[i],s=current.regions[r];if(!s.count)continue;const points=s.hist.map((n,j)=>`${x+(j+.5)/80*w},${y+h-(n/s.count)/max*h}`).join(' ');el('polyline',{points,fill:'none',stroke:colors[i],'stroke-width':2.3,opacity:region==='All'||region===r?1:.15});}for(let i=0;i<=8;i++)el('text',{x:x+w*i/8,y:y+h+23,'text-anchor':'middle',fill:'#607185','font-size':11},String(-40+i*10));el('text',{x:x+w/2,y:212,'text-anchor':'middle',fill:'#607185','font-size':11},'SCALED QK SCORE · BEFORE SOFTMAX');
 let clipped=0;for(const s of Object.values(current.regions))clipped+=s.underflow+s.overflow; $('hist-note').textContent=`Histogram: 80 bins from −40 to 40. ${format(clipped)} valid scores outside this range are counted separately. Curves show the fraction of each region’s valid token pairs per bin.`;
 $('region-cards').replaceChildren(...regions.map((r,i)=>{const s=current.regions[r],card=document.createElement('div');card.className='region-card';card.style.setProperty('--region',colors[i]);const title=document.createElement('h3');title.textContent=r;const value=document.createElement('strong');value.textContent=s.count?format(s.abs_sum/s.count):'—';const label=document.createElement('small');label.textContent='mean |score|';const detail=document.createElement('p');detail.textContent=s.count?`${format(s.count)} valid pairs · mean ${format(s.sum/s.count)} · ${(100*s.mass/(stats.mass||1)).toFixed(1)}% of total dense weight`:'No causal token pairs in this view';card.append(title,value,label,detail);return card;})); }
for(const name of ['All',...regions]){const button=document.createElement('button');button.textContent=name;button.className=name==='All'?'active':'';button.addEventListener('click',()=>{region=name;for(const b of $('region-buttons').children)b.classList.toggle('active',b===button);draw();});$('region-buttons').append(button);}
for(const name of ['phase','layer','head'])$(name).addEventListener('change',()=>updateControls(name));for(const name of ['step','metric','overlay'])$(name).addEventListener('change',draw);$('lambda').addEventListener('input',draw);$('dataset').addEventListener('change',selectDataset);window.addEventListener('resize',()=>{if(current)draw();});
$('heatmap').addEventListener('mousemove',event=>{const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,hit=hits.find(a=>x>=a.x&&x<a.x+a.w&&y>=a.y&&y<a.y+a.h),tip=$('tooltip');tip.hidden=!hit;if(!hit)return;const t=hit.t;tip.textContent=`Q ${t.q}–${t.q+t.rows-1} · K ${t.k}–${t.k+t.cols-1}\nmean ${format(t.mean)} · |score| ${format(t.abs)}\nmax ${format(t.max)} · margin ${format(t.margin)}\ndense weight ${(t.mass*100).toFixed(3)}%\nvisual Q ${(t.q_visual*100).toFixed(0)}% / K ${(t.k_visual*100).toFixed(0)}%\n${wouldSkip(t,10**Number($('lambda').value))?'WOULD SKIP':'KEEP'} · recorded ${t.skipped?'skipped':'kept'}`;tip.style.whiteSpace='pre-line';tip.style.left=`${Math.max(0,Math.min(x+12,rect.width-260))}px`;tip.style.top=`${Math.min(y+12,rect.height-175)}px`;});$('heatmap').addEventListener('mouseleave',()=>$('tooltip').hidden=true);
$('heatmap').addEventListener('click', event => {
  const tile = tileAt(event);
  if (tile) openTile(tile, true);
});
$('heatmap').addEventListener('keydown', event => {
  if (!current || !['Enter', ' ', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
  event.preventDefault();
  const tile = selectedTile ?? current.tiles[0];
  let next = tile;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    const row = current.tiles.filter(t => t.q === tile.q);
    const delta = event.key === 'ArrowLeft' ? -1 : 1;
    next = row[row.indexOf(tile) + delta] ?? tile;
  }
  if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
    const rows = unique(current.tiles.map(t => t.q));
    const delta = event.key === 'ArrowUp' ? -1 : 1;
    const query = rows[rows.indexOf(tile.q) + delta];
    if (query !== undefined) {
      const row = current.tiles.filter(t => t.q === query);
      next = row.find(t => t.k === tile.k) ?? row.at(-1);
    }
  }
  openTile(next, event.key === 'Enter' || event.key === ' ');
});
$('close-tile').addEventListener('click', () => {
  closeTile();
  drawMap(10 ** Number($('lambda').value));
  $('heatmap').focus({preventScroll: true});
});
$('upload').addEventListener('change', async event => {
  const files = [...event.target.files];
  event.target.value = '';
  $('error').hidden = true;
  const errors = [];
  for (const file of files) {
    try {
      if (file.name.toLowerCase().endsWith('.bin')) {
        scoreFiles.set(await scoreFileId(file), file);
      } else {
        if (file.size > 100 * 1024 * 1024) throw new Error('JSON exceeds 100 MB; collect fewer layers or heads.');
        const input = JSON.parse(await file.text());
        if (isJLSweep(input)) jlStudy.add(file.name, input);
        else if (isRegionAudit(input)) regionAudit.add(file.name, input);
        else if (isLayerStudy(input)) layerStudy.add(file.name, input);
        else if (isTVVVSweep(input)) tvVVSweep.add(file.name, input);
        else {
          showView('tiles');
          add(file.name, input);
        }
      }
    } catch (error) {
      errors.push(error.message);
    }
  }
  if (selectedTile) inspectTile(selectedTile);
  if (errors.length) fail(new Error(errors.join(' ')));
});
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='attention-study.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
try{const response=await fetch('demo.json?v=token-scores');if(!response.ok)throw new Error('Demo dataset could not be loaded. Load an experiment JSON to continue.');const demo = await response.json(); demoScoreId = demo.score_file?.id; add('Synthetic demonstration', demo);}catch(error){fail(error);}
try {
  const response = await fetch('jl-demo.json?v=jl-cases');
  if (!response.ok) throw new Error('Missing JL demo');
  jlStudy.add('Synthetic demonstration', await response.json(), false);
} catch {
  jlStudy.demoFailed();
}
try {
  const response = await fetch('region-demo.json?v=region-reset');
  if (response.ok) regionAudit.add('Synthetic demonstration', await response.json(), false);
} catch {
  // A local report can still be loaded if the demonstration is unavailable.
}
