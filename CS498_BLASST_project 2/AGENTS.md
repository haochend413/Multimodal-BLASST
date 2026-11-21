# For AI agents: how to use this folder

A team member gave you this folder so you can get them up to speed on the project quickly. It is a snapshot
(Nov 3, 2026) of a CS 498 course project: BLASST attention sparsity, reproduced on Llama text and then studied
on a video-language model.

## Read in this order

1. `README.md`: the overview, glossary, and key numbers (about 10 minutes).
2. `reports/1_baseline_llama.md`: skim section 1 (what reproduced) and section 8 (caveats).
3. `reports/2_vlm_regions_step1.md` and `reports/3_vlm_regions_batch2.md`: the main study. Read in full.
4. `reports/4_future_directions.md`: what is planned (nothing there has been run).
5. Only when needed: `results/README.md` (raw files), `code/README.md` (scripts, re-running),
   `notes/README.md` (paper notes and related work).

## Rules for answering questions

- **The reports are the source of truth for numbers.** Every number in them was computed from files in
  `results/`. Quote the number with its file and section. If a number is not in the reports or the CSVs,
  say so; do not estimate or invent one.
- **Use the glossary words** in `README.md` (tile, skip rule, lambda, margin, running max / bar, % skipped,
  dense, calibrated, regions T->T / T->V / V->T / V->V, question->video, weight lost, layouts A / B / C).
- **Keep three kinds of claims apart:** measured results (`reports/1-3`), hypotheses and plans
  (`reports/4`), and desk research (`notes/`, AI-assisted literature notes; nothing in `notes/` was run, and
  some paper-specific numbers there are unverified).
- **Offline vs live:** offline numbers come from dense-pass margins; live numbers come from runs with skipping
  on. They differ (50% offline was about 46% live in the pilot). Say which one you mean.
- **Do not overstate the accuracy pilot.** 65 / 61 / 64 out of 90 rests on 3-4 changed answers (p = 0.125 and
  0.25). It is a hint, not a result.
- **No speed claims.** The attention code is a PyTorch reference that computes every tile and masks the skipped
  ones. The timings in the reports are not speed-ups.
- Give absolute numbers (counts, percentages per setting), not only ratios.

## Where common questions are answered

| Question | Where |
|---|---|
| Did the BLASST reproduction work? | `reports/1_baseline_llama.md` section 1 |
| Why is fixed-lambda sparsity about half of the paper's? | `reports/1_baseline_llama.md` sections 4-5 |
| What are the regions, and how was % skipped per region measured? | `reports/2_vlm_regions_step1.md` "Words", "Setup" |
| Why is T->V skipped so much? | `reports/2_vlm_regions_step1.md` "Why: the margins" |
| Is the gap just distance, the transcript, or a few layers? | `reports/2_vlm_regions_step1.md` "Three checks" |
| How much attention does skipping remove? | `reports/3_vlm_regions_batch2.md` section 1 |
| Is the gap caused by our prompt layout? | `reports/3_vlm_regions_batch2.md` section 2 |
| Does skipping change answers? | `reports/3_vlm_regions_batch2.md` section 3 |
| What next? One lambda with a smarter bar? | `reports/4_future_directions.md`, `notes/one_threshold_designs.md` |
| What does the mid-semester report need? | `notes/midsem_report_requirements.md` |

## If asked to re-run or extend the code

Read `code/README.md`. The scripts were run on DeltaAI from a single folder and hard-code the path
`/work/nvme/beqg/zyang21/blasst` and the Slurm account `beqg-dtai-gh`; both must be changed. Large inputs
(model weights, Video-MME videos, saved margins) are not in this folder. Estimate GPU-hours before any run.

## Course policy

CS 498 allows AI tools for grammar checking and refining brainstorms, but the final reports and code must be
authored by the students. Help the team understand the work and check facts; do not write the report for them.
