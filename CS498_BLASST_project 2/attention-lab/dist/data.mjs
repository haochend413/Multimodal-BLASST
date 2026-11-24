// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export const regions = ['T→T', 'V→T', 'T→V', 'V→V'];
export const colors = ['#225fd3', '#8752be', '#cb6d24', '#128779'];
export function validate(data) {
  if (data?.schema !== 'blasst-attention-v1' || !Array.isArray(data.views) || !data.views.length || !data.metadata || typeof data.metadata.model !== 'string') throw new Error('Choose a blasst-attention-v1 JSON file produced by attention_study.py.');
  if (data.histogram?.bins !== 80 || data.histogram.min !== -40 || data.histogram.max !== 40) throw new Error('Unsupported histogram format.');
  if (data.score_file !== undefined && (typeof data.score_file?.id !== 'string' ||
      !/^[a-f0-9]{32}$/.test(data.score_file.id) ||
      !Number.isSafeInteger(data.score_file.bytes) || data.score_file.bytes < 32)) {
    throw new Error('Invalid token score companion reference.');
  }
  for (const v of data.views) {
    if (!['prefill', 'decode'].includes(v.phase) || !Number.isInteger(v.layer) || !Number.isInteger(v.head) || !Number.isInteger(v.step) || !Number.isFinite(v.key_length) || v.key_length <= 0 || !Array.isArray(v.tiles) || !v.tiles.length) throw new Error('Invalid attention view.');
    for (const t of v.tiles) if (!['q','k','rows','cols','mean','abs','max','mass','margin','q_visual','k_visual'].every(k => Number.isFinite(t[k])) || t.rows <= 0 || t.cols <= 0 || t.q < 0 || t.k < 0 || t.mass < 0) throw new Error('Invalid tile measurements.');
    for (const r of regions) { const s = v.regions?.[r]; if (!s || !['count','sum','abs_sum','mass','underflow','overflow'].every(k => Number.isFinite(s[k])) || s.count < 0 || !Array.isArray(s.hist) || s.hist.length !== 80 || s.hist.some(x=>!Number.isFinite(x)||x<0)) throw new Error('Invalid region statistics.'); }
  }
  return data;
}
export function summary(view, lambda) {
  const sums = Object.values(view.regions);
  const count = sums.reduce((a,s)=>a+s.count,0);
  return {tiles:view.tiles.length, skipped:view.tiles.filter(t=>wouldSkip(t, lambda)).length,
    abs:count ? sums.reduce((a,s)=>a+s.abs_sum,0)/count : 0,
    mass:sums.reduce((a,s)=>a+s.mass,0)};
}
export const wouldSkip = (tile, lambda) => tile.margin < Math.log(lambda);

export async function scoreFileId(file) {
  const id = await file.slice(0, 32).text();
  if (!/^[a-f0-9]{32}$/.test(id)) throw new Error('Invalid token score companion header.');
  return id;
}

export async function readScoreChunk(file, reference, view, tile) {
  const offset = view.score_chunks?.[tile.q];
  const stride = view.score_widths?.[tile.q] ?? Math.min(view.key_length, tile.q + tile.rows);
  const {rows, cols} = tile;
  const end = offset + rows * stride * 4;
  if (file.size !== reference.bytes || ![offset, rows, cols, stride, end].every(Number.isSafeInteger) ||
      offset < 32 || offset % 4 || tile.k + cols > stride || stride > view.key_length || end > file.size) {
    throw new Error('Token score data does not match this block.');
  }
  const bytes = new DataView(await file.slice(offset, end).arrayBuffer());
  const score = (row, key) => bytes.getFloat32((row * stride + key) * 4, true);
  return {
    tile: Array.from({length: rows}, (_, row) =>
      Array.from({length: cols}, (_, col) => score(row, tile.k + col))),
    row: row => Array.from({length: stride}, (_, key) => score(row, key)),
  };
}

export async function tileScores(file, reference, view, tile) {
  return (await readScoreChunk(file, reference, view, tile)).tile;
}
