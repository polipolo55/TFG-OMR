# Thesis Full Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perform a comprehensive, cross-referenced audit of every chapter of the TFG-OMR thesis, producing an edited document and a prioritised master improvement plan that accounts for incomplete/stub sections.

**Architecture:** One master coordinator (Claude in the main session) builds a cross-chapter intelligence brief from the current document state, then dispatches one orchestrating audit agent per chapter group. Each agent reads its chapters, the brief, and relevant project docs; audits text quality, citations, figures, and technical accuracy; makes targeted edits; and returns a structured report. The master synthesises all reports into a single master improvement plan.

**Tech Stack:** LaTeX (chapters in `latex_documents/main/chapters/` and `appendices/`), BibLaTeX (`references.bib`), TikZ figures, Python/PyTorch source in `src/`, docs in `docs/`, `latexmk` for build verification.

---

## Current Document State (as of 2026-05-20)

| File | Lines | Status |
|------|-------|--------|
| `00_abstracts.tex` | 77 | Complete — SER numbers corrected |
| `01_introduction.tex` | 533 | Complete — some ref? todos remain |
| `02_soa.tex` | 432 | Complete — TikZ figures added |
| `03_management.tex` | 985 | Complete — risk matrix figure placeholder |
| `04_design.tex` | 584 | Complete — 4 ref? + 2 figure placeholders |
| `05_implementation.tex` | 664 | Complete — 3 figure placeholders + 2 ref? |
| `06_results.tex` | 449 | Partial — domain gap §6.5 has 3 experiment TODOs |
| `07_sustainability.tex` | 433 | Complete |
| `08_conclusions.tex` | 36 | **STUB** — needs full writing |
| `app_a.tex` | 3 | **STUB** — hyperparameter tables |
| `app_b.tex` | 3 | **STUB** — extra qualitative figures |
| `app_c.tex` | 3 | **STUB** — code snippets |

### Known Open TODOs

**Missing citations (ref? markers):**
- `01_introduction.tex`: transcription-time per-page figure
- `04_design.tex`: CTC (Graves 2006), ResNet (He et al. 2016), LSTM (Hochreiter & Schmidhuber 1997)
- `05_implementation.tex`: AdamW (Loshchilov & Hutter 2019), OneCycleLR (Smith 2018)

**Missing figures (FIGURE markers):**
- `03_management.tex`: risk 3×3 likelihood/impact matrix (R1–R6)
- `04_design.tex`: pipeline overview block diagram; CRNN-CTC block diagram
- `05_implementation.tex`: data pipeline flow; augmentation samples; two-stream inference

**Experiment-pending sections (do NOT fill in — flag and note dependency):**
- `06_results.tex §6.5`: zero-shot eval on real pages, fine-tune quantitative results, full-page inference example — all depend on experiments not yet run

---

## Files Touched

| Chapter(s) | Primary Files | Docs to Cross-Check |
|------------|--------------|---------------------|
| 00+01 | `00_abstracts.tex`, `01_introduction.tex` | `docs/overview.md` |
| 02 | `02_soa.tex` | `docs/overview.md`, `docs/lmx_format.md` |
| 03 | `03_management.tex` | `docs/overview.md` |
| 04 | `04_design.tex` | `docs/model.md`, `docs/data_pipeline.md`, `docs/lmx_format.md`, `docs/training.md`, `src/CRNN_CTC/model.py`, `src/CRNN_CTC/config.py` |
| 05 | `05_implementation.tex` | `docs/data_pipeline.md`, `docs/inference_pipeline.md`, `docs/cli.md`, `docs/api.md`, `docs/configuration.md`, `src/CRNN_CTC/train.py`, `src/cli.py` |
| 06 | `06_results.tex` | `docs/overview.md` (Performance section) |
| 07+08 | `07_sustainability.tex`, `08_conclusions.tex` | `docs/overview.md` |
| App | `app_a.tex`, `app_b.tex`, `app_c.tex` | `docs/configuration.md`, `docs/training.md` |

---

## Task 0 — Master Pre-Flight Brief

> Executed by the master (main session), not a subagent. Produces the cross-chapter brief injected into every subsequent agent prompt.

**Files:** Read-only pass across all `.tex` files.

- [ ] **Step 0.1: Scan all section headings and labels**

```bash
grep -nE '^\\(chapter|section|subsection)\{' \
  latex_documents/main/chapters/*.tex \
  latex_documents/main/appendices/*.tex
```

- [ ] **Step 0.2: Scan all labels (for cross-ref verification)**

```bash
grep -hoE '\\label\{[^}]+\}' \
  latex_documents/main/chapters/*.tex \
  latex_documents/main/appendices/*.tex | sort -u
```

- [ ] **Step 0.3: Scan all citations used**

```bash
grep -hoE '\\(cite|textcite|parencite)[^{]*\{[^}]+\}' \
  latex_documents/main/chapters/*.tex | sort -u
```

- [ ] **Step 0.4: Scan all figure includes and placeholders**

```bash
grep -nE '\\includegraphics|\\todo\{FIGURE' \
  latex_documents/main/chapters/*.tex
```

- [ ] **Step 0.5: Scan all TODO markers**

```bash
grep -nE '\\todo\{' \
  latex_documents/main/chapters/*.tex \
  latex_documents/main/appendices/*.tex
```

- [ ] **Step 0.6: List available bib keys**

```bash
grep -E '^@' latex_documents/main/references.bib
```

- [ ] **Step 0.7: List available figure files on disk**

```bash
ls latex_documents/main/figures/*.{png,pdf,svg} 2>/dev/null
```

- [ ] **Step 0.8: Build the cross-chapter brief**

Synthesise findings from above into a structured markdown document (kept in memory for injection into each agent prompt) covering:
- Chapter → what it claims about other chapters (forward/backward refs)
- Known TODOs per chapter
- Known figure placeholders
- Labels that are referenced cross-chapter
- Which chapters depend on experiments not yet run

---

## Task 1 — Audit: Abstracts (Ch00) + Introduction (Ch01)

**Files:** `00_abstracts.tex`, `01_introduction.tex`
**Docs:** `docs/overview.md`
**Agent receives:** Chapter files + cross-chapter brief + references.bib key list

- [ ] **Step 1.1: Read both chapters in full**

- [ ] **Step 1.2: Verify abstracts against ch06 headline numbers**

Check: SER 1.19%/1.28%, perfect 71%/73%, melodic SER 0.18%/0.11%, test set 4604 samples. All three language versions (EN/CA/ES) must match.

- [ ] **Step 1.3: Audit introduction flow**

Verify section order: Motivation → Musical Notation Primer → Problem Formulation → Objectives → Proposed Approach → Scope → Document Structure. Flag any logical gaps.

- [ ] **Step 1.4: Check objectives against ch04 design and ch06 results**

Each objective stated in §"Objectives" should be traceable: either met in ch06 results or acknowledged as in-progress. Flag any objective with no evidence.

- [ ] **Step 1.5: Citation audit**

Every factual claim about music, OMR, Real Book, or existing tools must have a `\cite{}`. For each uncited claim: search `references.bib`, then web; add entry or `\todo{ref?}`.

- [ ] **Step 1.6: Cross-chapter repetition check**

Using the cross-chapter brief: trim any intro content that is already explained in full in ch02 or ch04. Replace with a single sentence + `\Cref{}`.

- [ ] **Step 1.7: Prose quality pass**

Fix passive voice overuse, overly long paragraphs, missing transitions between sections.

- [ ] **Step 1.8: Build + verify**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
```

Expected: no errors.

- [ ] **Step 1.9: Commit**

```bash
git add latex_documents/main/chapters/00_abstracts.tex \
        latex_documents/main/chapters/01_introduction.tex \
        latex_documents/main/references.bib
git commit -m "audit(ch00-01): abstracts + introduction pass"
```

- [ ] **Step 1.10: Write audit report**

Return a structured markdown report covering: changes made, remaining TODOs with descriptions, flagged cross-chapter issues, improvement suggestions.

---

## Task 2 — Audit: State of the Art (Ch02)

**Files:** `02_soa.tex`
**Docs:** `docs/overview.md`, `docs/lmx_format.md`
**Agent receives:** Chapter + brief + references.bib key list

- [ ] **Step 2.1: Read chapter in full**

- [ ] **Step 2.2: Verify citations for every system/dataset mentioned**

Systems: Audiveris, SmartScore, CRNN, PrIMuS, Camera-PrIMuS, SMT, SMT++, LEGATO, DoReMi, CVC-MUSCIMA, MUSCIMA++, DeepScoresV2, OLiMPiC. Bib keys must resolve for each.

- [ ] **Step 2.3: Verify figure labels**

`fig:soa-arch-bainbridge`, `fig:soa-arch-rebelo`, `fig:soa-crnn-pipeline`, `fig:soa-transformer-pipeline` — each must be: (a) defined with `\label{}`, (b) referenced from text with `\Cref{}` or `\ref{}`, (c) captioned, (d) rendered correctly in build.

- [ ] **Step 2.4: Check dataset table accuracy**

Table `tab:omr-datasets`: verify every size/label claim against the cited paper abstract (from references.bib metadata or web search). Flag any that look wrong with `\todo{verify:}`.

- [ ] **Step 2.5: Check §"OMR for Jazz Lead Sheets" positions thesis correctly**

The section must acknowledge `martinezSevillaJazzLeadSheets2025` (the concurrent jazz OMR corpus) and explain why it's insufficient for training from scratch.

- [ ] **Step 2.6: Summary section must preview design choices**

The §"Summary" section should end by signposting ch04 (Design) for the architecture choice rationale.

- [ ] **Step 2.7: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/02_soa.tex latex_documents/main/references.bib
git commit -m "audit(ch02): state of the art pass"
```

- [ ] **Step 2.8: Write audit report**

---

## Task 3 — Audit: Project Management (Ch03)

**Files:** `03_management.tex`
**Docs:** `docs/overview.md`
**Agent receives:** Chapter + brief + references.bib key list

- [ ] **Step 3.1: Read chapter in full**

- [ ] **Step 3.2: Verify architecture justification matches docs**

The "Chosen alternative" subsection must describe: two-stream (music CRNN + chord CRNN), LMX output, FastAPI endpoint. Check against `docs/overview.md` §"High-Level Architecture".

- [ ] **Step 3.3: Check budget arithmetic**

For `tab:hr-budget`: verify row totals (hours × role rate) match column sums. Verify grand total matches `tab:incidentals`. Social Security rate must be cited (`segSocialCotizacion2025`).

- [ ] **Step 3.4: Risk table R-code consistency**

Risk IDs R1–R6 in `tab:risk-register` must match codes in `tab:incidentals`. Verify no mismatch.

- [ ] **Step 3.5: Implement risk matrix figure**

Replace the `\todo{FIGURE: 3×3 likelihood/impact matrix}` placeholder with an actual TikZ or tabular figure. Positions:
- R1 (model fails): High likelihood, High impact → red
- R2 (data quality): Medium likelihood, High impact → orange
- R3 (timeline): Medium likelihood, Medium impact → yellow
- R4 (vocab retrain): Low likelihood, Medium impact → green
- R5 (hardware): Low likelihood, Medium impact → green
- R6 (cloud GPU): Low likelihood, Low impact → green

Read the actual risk register to get the correct assessments before drawing.

- [ ] **Step 3.6: Methodology section cross-refs**

Every reference to sprint structure must be consistent with the Gantt (four-week WPs, biweekly check-ins).

- [ ] **Step 3.7: Citation completeness**

GEP module, PMI, INE salary, Seguridad Social, REE/MITECO must all have `\cite{}`.

- [ ] **Step 3.8: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/03_management.tex
git commit -m "audit(ch03): management pass + risk matrix figure"
```

- [ ] **Step 3.9: Write audit report**

---

## Task 4 — Audit: Design (Ch04)

**Files:** `04_design.tex`
**Docs:** `docs/model.md`, `docs/data_pipeline.md`, `docs/lmx_format.md`, `docs/training.md`
**Source:** `src/CRNN_CTC/model.py`, `src/CRNN_CTC/config.py`, `src/omr_pipeline/grammar_fix.py`
**Agent receives:** Chapter + brief + docs content + references.bib key list

- [ ] **Step 4.1: Read chapter + relevant docs in full**

- [ ] **Step 4.2: Add missing fundamental citations**

For each `\todo{ref?}` in ch04:
- CTC: search bib for Graves; if absent, add `gravesConnectionistTemporal2006` from DOI 10.1145/1143844.1143891.
- ResNet: search bib for He; if absent, add `heDeepResidualLearning2016` from CVPR 2016.
- LSTM: search bib for Hochreiter; if absent, add `hochreiterLongShortTermMemory1997` from Neural Computation.
Add entries to `references.bib` in matching style and replace `\todo{ref?}` with `\cite{}`.

- [ ] **Step 4.3: Implement pipeline overview figure**

Replace `\todo{Block diagram: PDF/image → Preprocess → Staff Detect → {Melody CRNN | Chord CRNN} → Grammar Fix → JSON}` with a TikZ block diagram. Use the two-stream architecture: preprocessing → staff detect → [music branch | chord branch] → grammar fix → JSON output.

- [ ] **Step 4.4: Implement CRNN-CTC design figure**

The `\todo{FIGURE: CRNN--CTC block diagram}` in §"Model Design" should be replaced with a TikZ figure equivalent to `fig:soa-crnn-pipeline` in ch02 but showing design-level detail: exact layer names (ResNet18 backbone, stride table, BiLSTM 2×256, linear|V|+softmax).

- [ ] **Step 4.5: Verify LMX grammar section**

§"Grammar" production rules must match `docs/lmx_format.md`. Accidentals: bare tokens (`flat`, `sharp`, `natural`), appear after duration. No `acc:` prefix anywhere.

- [ ] **Step 4.6: Verify training strategy section**

§"Header-stripping augmentation": training-only (must be stated). §"Data splits": val=10%, test=10% — verify against `Config` defaults.

- [ ] **Step 4.7: Cross-ref SoA**

Every design decision must reference its SoA justification: "CRNN–CTC because..." → `\Cref{sec:soa-crnn}`. "LMX because..." → `\Cref{sec:soa-representations}`.

- [ ] **Step 4.8: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/04_design.tex latex_documents/main/references.bib
git commit -m "audit(ch04): design pass + missing citations + pipeline figure"
```

- [ ] **Step 4.9: Write audit report**

---

## Task 5 — Audit: Implementation (Ch05)

**Files:** `05_implementation.tex`
**Docs:** `docs/data_pipeline.md`, `docs/inference_pipeline.md`, `docs/cli.md`, `docs/api.md`, `docs/configuration.md`
**Source:** `src/cli.py`, `src/CRNN_CTC/train.py`, `src/CRNN_CTC/config.py`, `src/omr_pipeline/pipeline.py`, `src/api/`
**Agent receives:** Chapter + brief + docs + references.bib key list

- [ ] **Step 5.1: Read chapter + all implementation docs**

- [ ] **Step 5.2: Add missing citations**

For each `\todo{ref?}` in ch05:
- AdamW: add `loshchilovDecoupledWeightDecay2019` (ICLR 2019, arXiv:1711.05101).
- OneCycleLR: add `smithSuperConvergenceVeryFast2019` (arXiv:1708.07120 / ICCSI 2019 proceedings).
Add BibTeX entries matching existing style; replace `\todo{ref?}` with `\cite{}`.

- [ ] **Step 5.3: Implement data pipeline flow figure**

Replace `\todo{FIGURE: Data pipeline flow}` with TikZ: PrIMuS raw → `generate_realbook.py` (render) → `semantic_to_lmx.py` (convert) → `generate_headerless_twins.py` (twins) → `augment_scanned.py` (augment) → `vocab` → train/val/test split. Each node cites the corresponding script file.

- [ ] **Step 5.4: Implement augmentation samples figure**

Replace `\todo{FIGURE: augmentation samples}` with a placeholder figure that clearly specifies what should go here: one clean render + four progressively distorted variants (blur → warp → ink-bleed → JPEG). Write the `\todo{}` clearly enough that the author can generate it from the augmentation script.

- [ ] **Step 5.5: Implement two-stream inference figure**

Replace `\todo{FIGURE: inference pipeline two-stream}` with TikZ mirroring the design in `docs/inference_pipeline.md`: PDF/image → preprocess → staff detect → [music CRNN | chord CRNN] → grammar fix → JSON.

- [ ] **Step 5.6: CLI table accuracy**

`tab:cli-subcommands` must list exactly 10 subcommands: `render, convert, augment, vocab, train, evaluate, evaluate-ab, api, pipeline, pipeline-train`. Verify by reading `src/cli.py` subparser definitions.

- [ ] **Step 5.7: Hyperparameter table accuracy**

Every value in `tab:hyperparams` must match `Config` defaults in `src/CRNN_CTC/config.py`. Read the dataclass and compare line by line.

- [ ] **Step 5.8: API section completeness**

§"Web Service" must mention: `/api/omr/lead-sheet` (main endpoint), `/health`, and the chord labeler UI. Verify against `src/api/` files.

- [ ] **Step 5.9: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/05_implementation.tex latex_documents/main/references.bib
git commit -m "audit(ch05): implementation pass + missing citations + figures"
```

- [ ] **Step 5.10: Write audit report**

---

## Task 6 — Audit: Results (Ch06)

**Files:** `06_results.tex`
**Docs:** `docs/overview.md` §"Performance" (canonical number source)
**Source:** `scripts/evaluate_full.py`, training logs if available
**Agent receives:** Chapter + brief + docs Performance section + references.bib key list

- [ ] **Step 6.1: Read chapter + docs/overview.md Performance section**

- [ ] **Step 6.2: Verify all headline numbers**

Against canonical `docs/overview.md`:
- Aggregate SER: 1.28% (scanned), 1.19% (clean)
- Melodic SER: 0.18% (scanned), 0.11% (clean)
- Perfect transcriptions: 71% (scanned), 73% (clean)
- Error breakdown: ~74% barline, ~13% ties, ~87% structural
- Test split: 4604 samples
- Best epoch: 37
- Beam search: ≤1 edit/1000 at ~6× cost

Flag any discrepancy with `\todo{verify:}`.

- [ ] **Step 6.3: Verify all figure references**

List: `fig:training-curves`, `fig:ser-distribution`, `fig:error-pie`, `fig:top-token-errors`, `fig:ser-vs-length`, `fig:qual-perfect-input`, `fig:qual-perfect-pred`, `fig:qual-median-input`, `fig:qual-median-pred`, `fig:weight-dist`. Each must: (a) have a corresponding PNG in `figures/`, (b) be referenced from text, (c) have a caption.

```bash
ls latex_documents/main/figures/fig_*.png
```

- [ ] **Step 6.4: Handle experiment-pending TODOs correctly**

For §6.5 (domain gap): do NOT invent results. Each `\todo{}` should clearly state what experiment is needed. Rewrite vague TODOs to be more specific:
- Zero-shot eval: specify what metric (SER on N=? manually annotated staves)
- Fine-tune: specify checkpoint path and expected metric improvement
- Full-page example: specify which Real Book page to use and what the expected output JSON structure looks like

- [ ] **Step 6.5: Add worst-case example**

`\todo{Add a worst-case example from the right tail}` — either: (a) if an evaluation script can be run quickly, find a high-SER example and add it; or (b) expand the TODO to specify: "Find the 95th-percentile SER sample from the held-out test set using `scripts/evaluate_full.py --split test --export-worst 5` and include its input strip + predicted vs. ground-truth token diff."

- [ ] **Step 6.6: Cross-reference to design and implementation**

Results section must reference the model decisions it validates: e.g., "The low melodic SER (see \Cref{sec:design-model}) confirms..." and "The scan augmentation strategy described in \Cref{sec:design-data} successfully bridges..."

- [ ] **Step 6.7: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/06_results.tex
git commit -m "audit(ch06): results pass + experiment TODOs clarified"
```

- [ ] **Step 6.8: Write audit report**

---

## Task 7 — Audit: Sustainability (Ch07) + Conclusions Scaffolding (Ch08)

**Files:** `07_sustainability.tex`, `08_conclusions.tex`
**Docs:** `docs/overview.md`
**Agent receives:** Chapter + brief + references.bib key list + objectives list from ch01

- [ ] **Step 7.1: Read ch07 in full**

- [ ] **Step 7.2: Verify ch07 economic cross-refs to ch03**

€16097.16 total must match ch03 `tab:budget-total`. The €/page correction cost figure must be derivable from the stated hours and total.

- [ ] **Step 7.3: Verify ch07 environmental numbers**

Emission factor citation (`reeInformeSistemaElectrico2024` or `mitecoFactorEmision2025`). Arithmetic: hours × 250 W/1000 × emission_factor = kg CO₂. Check calculation is consistent with the stated 57.5 kWh.

- [ ] **Step 7.4: Write ch08 Conclusions prose**

Ch08 is currently a stub. Write the full conclusions chapter (~600–800 words) covering:
1. **What was built** (one paragraph): CRNN-CTC system for monophonic jazz lead sheets; two-stream architecture; LMX token format; FastAPI endpoint.
2. **What was achieved** (one paragraph): headline SER 1.19%/1.28%; perfect 71%/73%; melodic SER <0.2%; PrIMuS training set; scan augmentation bridges domain gap.
3. **Objectives review** (one paragraph per objective from ch01 §"Objectives"): state whether each was met or remains for future work, with `\Cref{}` to the evidence.
4. **Lessons learned** (one paragraph): LMX vs. MusicXML tradeoff; CTC vs. transformer tradeoff; synthetic-to-real gap challenge.
5. **Scope and limitations** (one paragraph): monophonic-only, treble-clef-only, no tuplets/repeats, page-0-only multi-page PDFs.

- [ ] **Step 7.5: Write ch08 Future Work prose**

Expand each bullet in the existing `\itemize` skeleton into 2–3 sentences: what would be done, what it would enable, what the technical blocker is.

- [ ] **Step 7.6: Scaffold appendix stubs**

For each appendix, replace the bare `\todo{}` with a more specific placeholder that tells the author exactly what goes there:
- `app_a.tex`: "Add `tab:hyperparams-full` — full grid search results for img_height, batch_size, BiLSTM hidden size. Source: `scripts/hyperparameter_search.py` output."
- `app_b.tex`: "Add `fig:qual-worst-5` — five highest-SER predictions from the test set. Run `scripts/evaluate_full.py --split test --export-worst 5 --output-dir figures/worst/`."
- `app_c.tex`: "Add the `generate_realbook.py` render loop, the `augment_scanned.py` pipeline config, and the `grammar_fix.py` BNF grammar — the three most self-contained and instructive scripts."

- [ ] **Step 7.7: Build + commit**

```bash
cd latex_documents/main && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -iE '! |undefined control'
git add latex_documents/main/chapters/07_sustainability.tex \
        latex_documents/main/chapters/08_conclusions.tex \
        latex_documents/main/appendices/app_a.tex \
        latex_documents/main/appendices/app_b.tex \
        latex_documents/main/appendices/app_c.tex
git commit -m "audit(ch07-08+app): sustainability verified, conclusions written, appendices scaffolded"
```

- [ ] **Step 7.8: Write audit report**

---

## Task 8 — Master Synthesis: Global Improvement Plan

> Executed by the master (main session). Collect all 7 agent audit reports and produce a single document.

**Output file:** `docs/superpowers/plans/2026-05-20-thesis-improvement-plan.md`

- [ ] **Step 8.1: Collect all reports**

Read the returned report from each of Tasks 1–7.

- [ ] **Step 8.2: Cross-chapter consistency check**

Verify: objectives (ch01) ↔ design (ch04) ↔ results (ch06) ↔ conclusions (ch08) form a coherent narrative. Flag any broken link.

- [ ] **Step 8.3: Final build verification**

```bash
cd latex_documents/main && latexmk -C && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | grep -cE 'Warning: Reference|Warning: Citation|! '
```

Expected: 0.

- [ ] **Step 8.4: Write master improvement plan**

Prioritised into three tiers:
- **Critical** (must fix before submission): broken claims, wrong numbers, missing mandatory citations
- **Important** (significantly improves quality): incomplete sections, weak cross-refs, missing key figures
- **Nice-to-have** (polish): prose clarity, figure aesthetics, table formatting

- [ ] **Step 8.5: Final commit**

```bash
git add docs/superpowers/plans/2026-05-20-thesis-improvement-plan.md
git commit -m "docs: master thesis improvement plan"
```

---

## Audit Agent Spec (inject into every agent prompt)

Each chapter audit agent must return a structured report with these sections:

```markdown
## Chapter Audit Report: Ch0N [Title]

### Changes Made
- Bullet list of every edit made, with file:line references

### Citations Added / Changed
- `\cite{key}` added for: [claim] (file:line)

### Figures Addressed
- Placeholder implemented / verified / left as TODO: [slug] (file:line)

### Open TODOs Remaining
- [todo text] at file:line — blocked on: [reason]

### Cross-Chapter Issues Found
- [issue] — involves chapters: [list]

### Improvement Suggestions (prioritised)
1. [Critical] ...
2. [Important] ...
3. [Nice-to-have] ...
```

---

## Definition of Done

- All `\todo{FIGURE:}` placeholders either replaced with real figures or rewritten with precise specifications.
- All `\todo{ref?}` placeholders either replaced with `\cite{}` or confirmed no good source exists.
- All `\todo{verify:}` markers either resolved or confirmed against `docs/overview.md`.
- `ch08_conclusions.tex` has real prose (not just bullets or stubs).
- `latexmk` produces 0 undefined references and 0 undefined citations.
- Master improvement plan exists at `docs/superpowers/plans/2026-05-20-thesis-improvement-plan.md`.
