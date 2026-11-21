# CS 498 (F26) Mid-Semester Report: what is actually required

Compiled 2026-10-31 for the "Improved Blocked Attention Sparsity" team (Haochen Ding, Lucas Cai, Yifei Li, Kevin Yang).
Due: **email, on or before 6:00 PM CST, Wed Nov 4, 2026**. Worth **10%** of the course grade (the middle term of "Project report 30% (5% + 10% + 15%)").

Sections:
- A. Verbatim requirements for THIS course (with sources)
- B. Unknowns to ask on Canvas
- C. Recommended 4-page outline with target lengths
- D. Grader checklist
- E. Conventions from similar courses (evidence for B and C)
- F. What could not be reached

---

## A. Verbatim requirements for THIS course

### A1. Course README (`C:\Users\kevin\coursework_local\CS_498\README.md`, branch `f26-mlsys`, = https://github.com/fanlai0990/cs498)

Project section, quoted in full:

> You will have to complete substantive work an instructor-approved problem and have original contribution. Surveys are not permitted as projects; instead, each project must contain a survey of background and related work.
>
> You must meet the following milestones (unless otherwise specified in future announcements) to ensure a high-quality project at the end of the semester:
>
> * Turn in a 2-page draft proposal ([template](https://www.overleaf.com/read/bsrcbphcvyzc#d075e8)), plus as many pages as needed for references, by **Sept 30**. Remember to include the names and UIUC email addresses of the group members.
> * Each group must schedule project discussion with the instructor during class hours or office hours in the week of **Oct 5** and **Oct 7**.
> * Each group must turn in a 4-page mid-semester report via email **on or before 6:00PM CST on Nov 4.**
> * Each group must turn in an 8-page final report and your code in Canvas **on or before 6:00PM CST on Dec 18.** The report must be submitted as a PDF file, with formatting similar to that of the papers you've read in the class. The self-contained (i.e., include ALL dependencies) code must be submitted as a zip file. Each zip file containing the code must include a README file with a step-by-step guide on how to compile and run the provided code.
> * You can find how to access GPU resources [here](./Resources/cloudlab.md).

Grading table row:

> | [Project report](#project)   | 30% (5% + 10% + 15%)    |

Schedule rows that touch the report:

> | Nov 4  | MoE Efficiency           | Fan Lai | | Mid-semester Report Due |
> | Nov 9  | Project Check-in & Feedback  |  Fan Lai       | | Assignment 2 Due, Assignment 3 Released (for 4-credit students) |
> | Nov 11     | Project Check-in & Feedback                                | Fan Lai        |        |                    |

Other README rules that apply to the report:

> **Canvas**:  *ALL* communication regarding this course must be via [Canvas](https://canvas.illinois.edu/courses/73559). This includes questions, discussions, announcements, assignments, as well as private messages.

> **Academic integrity**: [...] All material you submit in this course (reading summary, project reports, and presentation materials) must be your own. If you use someone else's material, you must cite them properly.

> **AI Tool Policy**: AI tools may be used for grammar checking and refining initial brainstorms, but the final reviews and codes **must** be authored by the student. Students are responsible for the entire content and must adhere to the Academic Integrity Policy.

### A2. Lecture 1 slides (`C:\Users\kevin\coursework_local\CS_498\Slides\L1_overview.pdf`, text extracted with pdfplumber)

This is the ONLY place the required contents of the mid-semester report are spelled out. Slide 40 (PDF page 59), verbatim:

> **Mid-Semester Report (Nov 4; 10%)**
> • Dedicated proposal feedback session (No class on Oct 5, 7)
> • This is to make sure you are receiving feedback and making progress
> • Feel free to stop by office hours on any day!
> • Extend your proposal to 4 pages (template), which must include
>   • What is the problem?
>   • Why is it important? (Motivating experiments)
>   • What are the most related work?
>   • What's your hypothesis so far?
>   • How are/will you evaluate it?

Slide 27 (PDF pages 42-45), grading table and the one-line rule under it:

> (Group) Project Presentation 15% One final presentation
> (Group) Project Report 30% Proposal, mid-term, final report
> While we value teamwork, clarify each member's contribution in group report.

Slide 44 (PDF page 63), milestone table:

> Date Milestone Details
> 09/05 Form Group Find 4-5 like-minded teammates
> 09/30 Submit Proposal Send your proposal by email to receive feedback either via email or in-person or both
> 10/05, 10/07 (no class) Proposal Feedback Refine the problem, overview, and form initial hypothesis/idea
> 11/04 Mid-term Report 4-page report
> 12/02 12/07 12/09 In-Class Project Presentations Present your research findings
> 12/18 Research paper Submit a report like the papers you read
> NO extensions! Delayed submission will receive NO CREDIT

Slide 37 (PDF pages 55-56), what counts as a project:

> **Research-Oriented Course Project**
> • The final project accounts for 45% of total grades
> • What can and cannot be a project?
>   • Just surveys are not allowed
>   • Measurements of new environments or of existing solutions in new environments are acceptable
>   • Proposing new research problems or addressing existing ones is highly preferred!
> • An ideal project should answer the questions you asked during paper summary and points you cared about in panel discussion

Slide 38 (PDF page 57), the research loop the instructor expects the report to reflect:

> **How to Approach it?**
> 1. Find a problem and motivate why this is worth solving
> 2. Quickly survey background and related work
>   • Might require you to go back to the first step
> 3. Form/update your hypothesis
> 4. Test your hypothesis
>   • Go back to 3 until you are happy
> 5. Present your findings to the instructor and in writing
>   • Discuss known limitations

Slide 39 (PDF page 58), the proposal spec the report "extends":

> **Draft Proposal (Sept 30; 5%)**
> • Two pages (template), plus references, must include
>   • Problem: What is the problem?
>   • Motivation: Why is it important?
>   • Solution overview: Any initial thoughts?
>   • Evaluation plan: How would you evaluate it?
> • Submit as a group, including your group information (form a group ASAP)
> • Approved by the instructor and agreed upon by you
> • Forms the basis of expectation

Slide 42 (PDF page 61), the final report the mid-semester report should grow into:

> **Final Report (Dec 18; 15%)**
> • Final report (8 pages excluding references by Dec 18):
>   • Should be written like the papers you've read
>   • Abstract, Introduction
>   • Background/Motivation
>   • Design
>   • Implementation and Evaluation
>   • Conclusion
> • As if you'd submit it to a workshop with ~3 more months of work or to a conference after ~6 more months of work
> • How to Write a Great Research Paper by Simon Peyton Jones

Slide 41 (PDF page 60), the instructor's stated grading lens for the presentation (the only rubric-like wording anywhere in the course material):

> • Graded by instruction team (50%) and your classmates (50%)
> • Instructor: based on format, correctness, depth, clarity, insights

Slides L2-L10: grepped for mid-semester / midterm / report / project / milestone / grading / rubric. Only hit is L10 ("A1 proposal due this Wednesday~", "(Remaining) Proposal feedback session"); nothing about the report.

### A3. Git history of the README (Fall'25 wording, same repo, `git log -p README.md`)

The Fall'25 syllabus (commits before `75be502 Syllabus`) said:

> * Each group must turn in a 3/4-page mid-semester report via email **on or before 6:00PM CST on Nov 7.**
> | [Project report](#project)   | 25% (5% + 5% + 15%)    |

So for F26 the instructor (i) tightened "3/4-page" to "4-page" and (ii) doubled the weight from 5% to 10%. The report is being taken more seriously this year.

### A4. Sibling course by the same instructor: CS 598 Systems for GenAI (https://github.com/fanlai0990/cs598)

Same README skeleton, but the reference rule is stated explicitly:

> Each group must turn in a 4-page mid-semester report, plus as many pages as needed for references, via email **on or before 11:59PM CST on April 5.**
> Each group must turn in an 8-page final report, plus as many pages as needed for references, and your code via email [...]

This is the strongest evidence that in Fan Lai's courses references do NOT count toward the 4 pages (also consistent with the F26 proposal bullet and with slide 42 "8 pages excluding references"). Still confirm (see B).

### A5. Prof. Minjia Zhang's CS 498 (the course this one "alternates with")

- Spring 2025 (https://minjiazhang.github.io/courses/cs-mlsys-498-2025spring.html): project "consists of a proposal, mid-term report, final presentation, and final report"; "Project report 25% (5% + 5% + 15%)"; groups of 2-3. No page limits, sections, or rubric on the page.
- Spring 2026 (https://minjiazhang.github.io/courses/cs-mlsys-498-2026spring.html): "Each report follows a conference-style structure" with sections introduction, problem formulation, related work, design, implementation, evaluation, discussion; midterm report due 3/31; "Project report 25% (5% + 5% + 14%)". No page limit or rubric.
- Fall 2024 CS 598 AIE (https://minjiazhang.github.io/courses/cs-aie-598-2024fall.html): midterm report "This should be about 3 pages long (excluding references)", summarizing progress, obstacles, scope adjustments, and a remaining-task schedule. Final report "about 6-8 pages long (excluding references)" with suggested lengths: abstract 0.25 p, intro 0.75, related work 0.5, background 0.75, analysis/characterization 1, main design 1-2, implementation 0.5, evaluation 1-2, conclusion 0.25.

Net: Minjia Zhang's pages add no rubric, but they confirm (a) references excluded from page counts and (b) a conference-style section structure is the norm in this course family.

---

## B. Unknowns: ask on Canvas (one post, before Nov 2)

Nothing below is stated anywhere in the F26 README, slides, or repo history.

1. **Email recipient and subject line.** README says "via email" and slide 44 says proposal "by email", but no address. Ask: send to fanlai@illinois.edu only, or cc TAs (banruol2, tonyhong)? Suggested subject: `[CS 498 F26] Mid-semester report - <team name>`. (Canvas rule says all communication must be via Canvas, yet the report is explicitly "via email"; ask which wins.)
2. **Do references count toward the 4 pages?** Proposal bullet and CS 598 say "plus as many pages as needed for references"; the F26 mid-semester bullet omits this phrase. Assume excluded, but confirm.
3. **Template / format.** Slide 40 says "Extend your proposal to 4 pages (template)". Confirm: same Overleaf proposal template (2-column, same font)? Is an abstract wanted? Are figures/tables inside the 4 pages (almost certainly yes)?
4. **Hard cap or target?** F25 said "3/4-page"; F26 says "4-page". Ask whether 4 is a maximum (treat it as a maximum regardless).
5. **Code / repo link.** Only the final report requires a code zip. Ask whether a GitHub link or a short artifact appendix is welcome now.
6. **What the 10% is graded on.** No rubric is published. Ask whether it is progress (results obtained), writing, or both, and whether the Nov 9/11 "Project Check-in & Feedback" sessions are driven by this report (so we should book a slot).
7. **Late policy.** Slide 44: "NO extensions! Delayed submission will receive NO CREDIT." Confirm it applies to the mid-semester report (assume yes; submit by Nov 4 noon to be safe).
8. **Member contributions.** Slide 27 requires clarifying each member's contribution "in group report". Confirm a short contributions paragraph/table is wanted at mid-semester, not only in the final.
9. **Proposal feedback.** If the Oct 5/7 feedback was oral, ask whether the instructor wants a one-paragraph "changes since proposal / response to feedback" block (CS224N-style). Cheap to include either way.

---

## C. Recommended 4-page outline (2-column, same Overleaf template as the proposal)

Rationale: slide 40 lists five mandatory questions; the proposal template already has sections 1-5; slide 27 demands per-member contributions; every comparable course (section E) adds "preliminary results", "revised plan", and "concerns". The outline below keeps the proposal's numbering so the instructor sees the extension, and adds the three things a mid-semester report must newly contain: **motivating experiments, preliminary results, and a revised plan**.

Budget: a 2-column 10-11 pt page holds roughly 900-1000 words with no figures; plan on ~3,000 words of prose + 3-4 figures/tables in 4 pages. Fix the date line (the proposal PDF currently prints "November 1, 2026" from `\today`; set it explicitly to the submission date).

| # | Section | Target | What goes in it |
|---|---------|--------|-----------------|
| - | Title, authors + UIUC emails, date | 0.05 p | Keep the proposal header. Title may be sharpened, e.g. "Cheaper and More Accurate Block Selection for Training-Free Sparse Attention". |
| 0 | Abstract (optional, recommended) | 0.1 p | 5-6 sentences: problem, hypothesis, what has been measured so far (give one or two concrete numbers), what remains. The final report needs one anyway. |
| 1 | What is the problem? | 0.4 p | Revised from proposal, incorporating Oct 5/7 feedback. Define the two BLASST limitations crisply: (L1) row-max skip rule keeps unimportant blocks because of outliers; (L2) QK^T is still computed for blocks that are later skipped. State the concrete scope now (which models, which workloads, prefill vs decode). End with 2-3 explicit goals. |
| 2 | Why does it matter? + Motivating experiments | 0.7 p | NEW and explicitly required ("Motivating experiments"). 1-2 figures with measured numbers from the BLASST reproduction, e.g. (a) attention share of prefill latency vs. context length on your GPU; (b) fraction of QK^T FLOPs/time spent on blocks that are subsequently skipped, at the thresholds BLASST uses; (c) block-sparsity vs. output-error curve showing where the row-max rule wastes blocks. Each figure gets one sentence of "so what". |
| 3 | Most related work | 0.5 p | Required ("What are the most related work?") and README says every project "must contain a survey of background and related work". Group by selection signal: softmax-statistics skipping (BLASST), proxy/approximate-score selection (MInference, SpargeAttention, CoSA, FlexPrefill, XAttention), trained sparse attention as a contrast (SeerAttention, NSA/MoBA), and dense baselines (FlashAttention). A small table (method / selection signal / where cost is paid / training-free? / modality) is worth the space. One sentence per method on how yours differs. |
| 4 | Hypotheses and approach so far | 0.9 p | Required ("What's your hypothesis so far?"). Turn the proposal's three ideas into testable hypotheses: H1 downsampled Q/K give a skip signal that matches BLASST's decisions at X% agreement while cutting score-FLOPs by Y; H2 the block mask can be reused across k adjacent layers with bounded error growth; H3 per-layer/head/modality thresholds beat one global threshold on the accuracy-speed curve. For each: design sketch (a figure of the modified kernel pipeline), what is implemented today, what is not. State the key design choices and why (slide 38 step 3). |
| 5 | Evaluation: setup and preliminary results | 1.0 p | Required ("How are/will you evaluate it?"). Setup: hardware (which GPU, CloudLab/NCSA), models and sizes, workloads (text long-context, image/video if started), baselines (dense FlashAttention, BLASST), metrics (output error vs dense, achieved block sparsity, peak GPU memory, kernel latency, end-to-end latency incl. selection/mask overhead). Results: a table reproducing BLASST's headline numbers vs. the paper, plus 1-2 figures for whichever of H1-H3 has data. Interpret: expected vs. surprising, what the numbers say about what to do next. Be honest about what is not yet measured. |
| 6 | Revised plan, timeline, risks | 0.3 p | Week-by-week Nov 4 to Dec 18 (presentations Dec 2/7/9, report Dec 18). Which hypotheses are kept, cut, or re-ordered since the proposal, and why. Risks and mitigations: GPU access, Triton/CUDA kernel engineering time, multimodal workloads as a stretch goal. |
| 7 | Member contributions | 0.15 p | Required by slide 27. Table: member / done so far / owns next. Keep it factual. |
| - | References | not counted (confirm, B2) | BibTeX; include everything cited in sections 3-5. |

Total ~4.1 pages before trimming; the abstract and the related-work table are the first things to shrink if over.

Reuse rule (from CS224N and CS229, consistent with "Extend your proposal"): copying proposal text is fine, but spend less space re-introducing and more on what changed and what was measured.

---

## D. Checklist: what a grader most likely rewards

Derived from slide 40 (required contents), slide 27 (contributions), slide 38 (research loop), slide 41 (instructor's lens: "format, correctness, depth, clarity, insights"), slide 44 (no late credit), and the milestone rubrics in section E (progress / understanding / writing).

Hard requirements (lose points if missing):
- [ ] Each of the five slide-40 questions is answered under a heading a skimmer can find: problem, importance (with motivating experiments), related work, hypothesis so far, evaluation.
- [ ] At least one motivating experiment with measured numbers (seconds, FLOPs, % of blocks), not only citations of other papers' numbers.
- [ ] Preliminary results on your own setup: at minimum a reproduction of BLASST against dense FlashAttention with the proposal's four metrics (output error, block sparsity, peak memory, speed).
- [ ] Hypotheses stated as falsifiable claims (H1-H3), each tied to a planned or completed experiment.
- [ ] Related-work survey that positions you against BLASST, MInference, SpargeAttention, CoSA at least (README: "each project must contain a survey of background and related work").
- [ ] Per-member contributions, explicit (slide 27).
- [ ] Revised timeline through Dec 18 and what changed since the proposal.
- [ ] 4 pages, same template as proposal, names + UIUC emails, PDF, emailed before 6:00 PM CST Nov 4 (no credit if late).
- [ ] Everything cited; no uncredited reuse of BLASST/MInference code or text (academic-integrity clause). AI tools only for grammar/brainstorm polish.

Things that separate a strong report:
- [ ] Figures with axis labels and units; every figure referenced in text with a one-line takeaway.
- [ ] A results table that includes the paper's reported numbers next to yours (shows reproduction fidelity and that you understand the deltas).
- [ ] An explicit "what surprised us / what we learned" paragraph (slide 38 steps 3-4: form/update hypothesis, test, iterate).
- [ ] Known limitations and risks named (slide 38 step 5: "Discuss known limitations").
- [ ] One paragraph acknowledging and responding to the Oct 5/7 proposal feedback.
- [ ] The report already reads like the first half of the final paper (abstract, intro, background, design, eval), so the Dec 18 report is an extension rather than a rewrite.
- [ ] Absolute numbers everywhere (ms per kernel call, GB peak memory, % blocks skipped), with the hardware and sequence lengths they were measured on.

---

## E. Conventions from comparable courses (used to fill the gaps in C and D)

Verbatim or near-verbatim; URLs are the sources.

1. **Stanford CS224N, Project Milestone Instructions (Feb 17, 2026)** - https://web.stanford.edu/class/cs224n/project/Project_Milestone_Instructions.pdf
   - "not exceeding two pages in length (excluding figures and references), which is worth 6% of your overall grade."
   - Required sections: Key information (title, members, emails), Abstract, Approach ("describe your baselines"; "If any part of your approach is original, make it clear"; "If you are using any code that you did not write yourself, make it clear and provide a reference or link"), Experiments (Data, Evaluation method, Experimental details, Results: "Use a table or plot to compare multiple results and compare against your baselines"; "Comment on your quantitative results. Are they what you expected?"), Future work, References.
   - Rubric: "Progress: Has the team made good progress on the project? You should have done approximately half of the work of your project." "Understanding: Does the milestone show a strong understanding of its problem, tasks, methods, metrics, and research context?" "Writing quality: Does the milestone clearly communicate what you've done and why [...]?"
   - "spend less space re-introducing your approach and more space on what you've accomplished and what new plans you've made since the original proposal."

2. **Stanford CS229, Milestone** - https://cs229.stanford.edu/projects.html-backup-fall21
   - "Your milestone should be at most 3 pages, excluding references."
   - Contents: Motivation ("What problem are you tackling, and what's the setting you're considering?"), Method, Preliminary experiments ("You should have tried at least one baseline"), Next steps, team names + title, and "a section that describes what each team member worked on and contributed to the project."
   - Graded mainly as a progress check: "you should do well if you follow the instructions and have tested any assumptions that could prevent your team from finishing the project."

3. **CMU 15-745 Optimizing Compilers (systems course), Milestone Report** - https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15745-s19/www/project.html
   - "The milestone report should be at most 3 pages." It "should discuss": "A summary of the motivation for the project, and how it compares with related work."; "A re-iteration of your proposed goals, what progress you have made to date on those goals"; "A list of the major components of your evaluation plan. What's been completed and what did you learn?"; "What are the current concerns with this project and its progress? Any stumbling blocks discovered?"

4. **CMU 10-601 Machine Learning (Fall 2010), Midway Report** - https://www.cs.cmu.edu/afs/cs/academic/class/10601-f10/projects.html
   - "This should be a 2 pages short report, and it serves as a check-point." "pretty complete introduction and related work sections"; "discuss the methods, code and experiments you plan to perform"; "Also include (very briefly) whatever results you have obtained so far." Worth 5%.

5. **KAUST CS 294E Machine Learning Systems (Fall 2021), mid-semester progress** - https://sands.kaust.edu.sa/classes/CS294E/F21/project.html
   - Mid-semester deliverable is a presentation, but the content list is the closest ML-systems analogue: "Motivate the project and provide background."; "Explain the difference between this project and related work."; "Report on early results as appropriate."; "Recap the status of the project."; "Illustrate the plan for the remaining time."
   - Grading philosophy: "More than the success of the final results, we look for how thorough your efforts have been." Final report: "8 page report [...] with formatting similar to that of the papers you've read in the class" (identical phrasing to Fan Lai's README).

6. **CMU 11-868 LLM Systems (Spring 2025), Projects page** - https://llmsystem.github.io/llmsystem2025spring/docs/Projects
   - Report sections: Introduction/Motivation, Related Work/Background, Methodology, Experiments, Analysis/Ablations, Conclusion/Discussion, Limitations, "[final report only] Team Member Contributions". No midterm page limit or rubric published.

7. **UIUC CS 598 AIE (Minjia Zhang, Fall 2024)** - https://minjiazhang.github.io/courses/cs-aie-598-2024fall.html
   - Midterm report "about 3 pages long (excluding references)": progress summary, obstacles, scope adjustments, remaining-task schedule. Final-report section budget quoted in A5.

Common denominator across 1-7, which C follows: (problem + motivation, revised) -> (related work, near-final) -> (approach/hypotheses, what is built) -> (preliminary results vs. a baseline, with a table/plot and interpretation) -> (revised plan + risks) -> (per-member contributions) -> references outside the page count.

---

## F. What could not be reached or does not exist

- **Overleaf proposal template** (https://www.overleaf.com/read/bsrcbphcvyzc#d075e8): fetch returns Overleaf's generic site shell (the read-only project needs a JavaScript client / login). Open it in a browser; it is the format to extend. No mid-semester guidance is expected inside it.
- **Canvas** (https://canvas.illinois.edu/courses/73559): not fetchable; any announcement that overrides the README ("unless otherwise specified in future announcements") would live there. Check Announcements before writing.
- **Berkeley CS 294-162 ML Systems (Fall 2024)** (https://ucbsky.github.io/aisys-fa2024/): project section not in the fetched HTML; policies read "TBD". Berkeley AI-Sys Fall 2019 (https://github.com/ucbrise/cs294-ai-sys-fa19/blob/master/index.md) has no midterm report at all (final 6 pages + references).
- **CMU 15-884 ML Systems (Spring 2021)** (https://catalyst.cs.cmu.edu/15-884-mlsys-sp21/): only "Students will form teams to work on a final project"; no milestone spec public.
- **UW CSE 599 systems-for-ML**: no public milestone-report spec found via search; the CSE 599 handout that surfaced is an Advanced ML course, not systems.
- The current GitHub README (fetched 2026-10-31) lists the project check-ins as Nov 11 and Nov 16 rather than the local copy's Nov 9 and Nov 11; the local repo may be one commit behind. `git pull` before relying on the check-in dates. The Nov 4 report deadline is identical in both.
