export const isTVVVSweep = data => data?.schema === 'blasst-tv-guided-vv-v1';

export function setupTVVVSweep(showView) {
  const $ = id => document.getElementById(`tv-vv-${id}`);
  const reports = [];
  const active = () => reports[Number($('report').value)];
  const variantLabel = value => active().data.variant_labels?.[value] ??
    (value === 'baseline' ? 'Standard BLASST' : `${Number(value) * 100}% relevance`);
  const fraction = (a, b) => b ? `${a} / ${b} (${(100 * a / b).toFixed(1)}%)` : '—';

  function options(id, values, label = String) {
    const element = $(id);
    const previous = element.value;
    element.replaceChildren(...values.map(value => new Option(label(value), value)));
    if (values.map(String).includes(previous)) element.value = previous;
  }

  function row(body, values) {
    const entry = body.insertRow();
    for (const value of values) entry.insertCell().textContent = value ?? '—';
  }

  function drawVotes() {
    const data = active().data;
    const run = data.cases[Number($('case').value)].results[$('variant').value];
    const layer = run.layers.find(item => item.layer === Number($('layer').value));
    $('votes').replaceChildren();
    if (!layer) return;
    const source = layer.vote_layer == null ? '' : ` · votes from layer ${layer.vote_layer}`;
    $('coverage').textContent = `V→V QK tiles bypassed in layer ${layer.layer} · ${fraction(layer.vv_tiles_bypassed, layer.pure_vv_tiles)}${source}`;
    const score = run.vqa_score == null ? '' : ` · VQA score ${run.vqa_score.toFixed(1)}%`;
    $('answer').textContent = run.answer == null ? '' : `Generated answer · ${run.answer || '(empty)'}${score}`;
    for (const vote of layer.votes) {
      const decision = vote.mask_fallback ? 'Mask fallback' : vote.protected_first_key
        ? 'Keep first key' : {skip:'Bypass', keep:'Keep', warmup:'Warm-up', baseline:'Baseline'}[vote.decision];
      row($('votes'), [vote.key_start,
        vote.previous_relevance == null ? '—' : `${(100 * vote.previous_relevance).toFixed(1)}%`,
        decision, vote.vv_tiles_bypassed, fraction(vote.kept, vote.eligible)]);
    }
  }

  function drawLayers() {
    const data = active().data;
    const run = data.cases[Number($('case').value)].results[$('variant').value];
    $('layers-detail').hidden = !run.layers?.length;
    options('layer', (run.layers ?? []).map(layer => layer.layer), layer => `Layer ${layer}`);
    if (!run.layers?.length) {
      const score = run.vqa_score == null ? '' : ` · VQA score ${run.vqa_score.toFixed(1)}%`;
      $('answer').textContent = `Generated answer · ${run.answer || '(empty)'}${score}`;
      return;
    }
    drawVotes();
  }

  function drawReport() {
    const {data} = active();
    const variants = Object.keys(data.variants);
    const official = data.answer_metric === 'VQA v2 consensus';
    const budgets = data.budget_calibration;
    $('budget-note').hidden = !budgets;
    if (budgets) {
      const targets = budgets.qk_skip_targets;
      $('budget-note').textContent = `Calibration targets from QK skip only · prefill ${(100 * targets.prefill).toFixed(1)}% skipped · decode ${(100 * targets.decode).toFixed(1)}% skipped. The table shows held-out skip rates.`;
    }
    const maxPhases = ["prefill", "decode"].filter(phase => data[`${phase}_max_policies`]);
    $('accuracy-header').textContent = budgets ? 'VQA v2 accuracy · Δ vs QK skip' :
      official ? 'VQA v2 accuracy' : 'Reference exact matches';
    for (const phase of ["prefill", "decode"]) {
      $(`${phase}-header`).hidden = !maxPhases.includes(phase);
    }
    $('status').textContent = `${data.synthetic ? 'Synthetic · ' : ''}${data.pairs_completed ?? data.cases.length}/${data.pairs_requested} cases · BLASST λ = ${data.blasst_lambda}`;
    $('summary').replaceChildren();
    for (const variant of variants) {
      const result = data.summary[variant];
      const difference = budgets ? result.vqa_delta_vs_qk_pp : result.vqa_delta_pp;
      const delta = difference == null ? '' :
        ` (${difference >= 0 ? '+' : ''}${difference.toFixed(1)} pp)`;
      const accuracy = official ? `${result.vqa_accuracy.toFixed(1)}%${delta}` :
        fraction(result.reference_exact_matches ?? 0, result.answers_scored);
      const cells = [variantLabel(variant), fraction(result.vv_tiles_bypassed, result.pure_vv_tiles)];
      for (const phase of maxPhases) {
        const policy = data[`${phase}_max_policies`][variant] === 'per_region' ? 'Per region' : 'Shared';
        const lam = budgets ? ` · λ ${Number(budgets.lambdas[variant][phase].toPrecision(4))}` : '';
        cells.push(`${policy}${lam} · ${fraction(result[`${phase}_tiles_skipped`], result[`${phase}_tiles`])}`);
      }
      row($('summary'), [...cells, accuracy, result.answers_changed]);
    }
    $('content').hidden = false;
    $('case-detail').hidden = !data.cases.length;
    if (!data.cases.length) return;
    options('case', data.cases.map((_, index) => index), index => data.cases[index].sample_id ?? `Case ${index + 1}`);
    options('variant', variants, variantLabel);
    drawLayers();
  }

  $('report').addEventListener('change', drawReport);
  $('layer').addEventListener('change', drawVotes);
  for (const id of ['case', 'variant']) $(id).addEventListener('change', drawLayers);
  return {
    add(name, data) {
      if (!isTVVVSweep(data) || (!data.cases?.length && !data.vqa_merged) || !data.summary) {
        throw new Error('Invalid T→V-guided V→V report.');
      }
      reports.push({name, data});
      options('report', reports.map((_, index) => index), index => reports[index].name);
      $('report').value = reports.length - 1;
      drawReport();
      showView('tv-vv');
    },
  };
}
