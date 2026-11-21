# Notes: background desk research

Literature and reference notes written while planning (Oct 31 - Nov 3, 2026), with AI assistance. **Nothing in
this folder was run as an experiment**; measured results are only in `../reports/`. Paper quotes were taken
from the papers where the note says so, but some paper-specific numbers are unverified. A few notes mention files
or plans that are not in this package (an early team plan, other planning notes, DeltaAI paths); ignore those
references.

| File | What it covers | Most useful for |
|---|---|---|
| `blasst_paper.md` | The BLASST paper (arXiv 2512.12087): exact skip rule, calibration rule, every target number | understanding BLASST; the baseline comparison |
| `modelopt_and_hf_interface.md` | NVIDIA's reference code (ModelOpt), the Hugging Face attention-function contract, RULER details | reading or changing the code |
| `related_work.md` | Survey of sparse-attention work positioned against the proposal's ideas, with BibTeX | the related-work section |
| `directions_idea1.md` | Open directions for making BLASST's in-layer skip decision cheaper or more accurate (proposal idea 1) | why idea 1 was not pursued as proposed |
| `directions_idea2.md` | Reusing skip decisions across layers and steps (proposal idea 2): the published family of methods | why idea 2 was not pursued as proposed |
| `multimodal_qwen25vl.md` | Qwen2.5-VL reference: video tokenization, token ids, attention routing, memory, Video-MME, pitfalls | the VLM code and setup |
| `paper_2607.02980.md` | HiLS: judging a tile by its max vs its softmax mass | future direction "max vs mass" |
| `paper_osdi26_liu.md` | ECHO (OSDI 2026): KV cache in CPU memory with prefetching; mask stability | future direction "mask stability / prefetch" |
| `one_threshold_literature.md` | Survey: methods that use one threshold on some statistic, and whether it transfers across modalities | future direction "one lambda, smarter bar" |
| `one_threshold_designs.md` | Design analysis: which one-lambda designs are really per-region lambda in disguise; candidates; test plan | future direction "one lambda, smarter bar" |
| `midsem_report_requirements.md` | What the 4-page mid-semester report must contain, with sources | writing the report |
