---
name: writing-tfg-fib
description: Use when writing, editing, or reviewing chapters and sections of the TFG memoir for FIB/UPC (Grau Enginyeria Informàtica). Covers Article 12/22 formal rules (cover, three-language abstract, mandatory sections), the FIB tribunal rubric criteria, per-chapter scope decisions (Design vs Implementation, etc.), and academic writing voice for a Bachelor's thesis in English. Triggers: writing or revising chapters under latex_documents/main/chapters/, drafting the abstract or cover, deciding where content belongs, reviewing prose for tone or citation hygiene.
---

# Writing the TFG memoir at FIB/UPC (GEI)

## Overview

This skill encodes the formal rules and writing conventions for a Bachelor's thesis (TFG, Treball de Fi de Grau) at the Facultat d'Informàtica de Barcelona (FIB), Universitat Politècnica de Catalunya (UPC), in the Grau en Enginyeria Informàtica (GEI), Computació specialty. Grounded in *Normativa del treball de fi d'estudis de la FIB* (juny/25) and the tribunal rubric.

Goal of the memoir:
1. Pass the formal checks — cover, abstracts, mandatory sections (Art. 12 + 22).
2. Map cleanly onto the tribunal rubric — contextualisation, work plan, methodology, alternatives, knowledge integration, regulations, commitment, initiative.
3. Read like a Bachelor's thesis, not a lab report or a blog post.

## When to use

- Drafting any chapter of `latex_documents/main/chapters/*.tex`
- Reviewing existing thesis prose for tone, citations, or structure
- Deciding where content goes (Design vs Implementation, Management vs Conclusions, etc.)
- Building or revising the abstract, cover, or appendices

**Not for:** GEP submodule deliverables (E1–E4 have their own rubrics under `latex_documents/gep/*/Rubric*.pdf`); their scope differs from the final memoir.

## Hard rules — normativa (non-negotiable)

| Rule | Source | Detail |
|---|---|---|
| Abstract in **three languages** | Art. 22 | 1 or 2 *pages* each in Catalan, Spanish, English. **Not** word-counted. Missing any one is a formal defect. |
| Cover includes specialty | Art. 22 | After degree name (apartat e), append specialty (Computació). |
| Cover required fields | Art. 12 a–g | Title, author, defense date, director (+ dept), degree, FIB, UPC-BarcelonaTech. Modality B/D adds h (company). |
| Sustainability & ethics section | Art. 12 | Mandatory per Acord Consell de Govern UPC 07/2023. Not optional. |
| Source code only | Art. 22 | If software is submitted: source files only, no compiled binaries or object files. |
| Memoir language | Art. 16 | Catalan, Spanish, **or** English. Stay consistent throughout the body. |
| Workload accounting | Art. 17 | 18 ECTS × 30 h/ECTS = **540 h** total. Budget chapter must reconcile to this envelope. |
| Grade composition | Art. 21 | 60% technical competencies + 40% transversal. GEP fita inicial weights 25% of transversal; fita de seguiment 25%; fita final 50%. Up to +1 bonus point possible. |

When citing a rule in chat with the user, name the article — it's how disagreements get resolved.

## Rubric anchors

The tribunal scores against these criteria. Every chapter should visibly contribute to at least one row. If it doesn't, ask why it exists.

| Rubric criterion | Where the evidence lives |
|---|---|
| Contextualisation of the project | Intro + SoA |
| Work plan and adjustments | Management |
| Methodology and rigour | Management + Design intro |
| **Analysis of alternative solutions** | Design (explicit "Alternatives considered" subsection per major decision) |
| Integration of knowledge | Implementation + Conclusions (map to coursework) |
| Identification of applicable laws & regulations | Sustainability + Management (IP, GDPR, dataset licenses) |
| Commitment to the project | Tone, depth, citation hygiene across the whole memoir |
| Initiative and decision-making | Design (justified choices) + Management (replanning) |
| Objectives reached vs proposed | Conclusions (explicit list: met / partial / dropped) |
| Application of prior knowledge | Conclusions ("competences" subsection) |

## Per-chapter scope

Use this as the deciding rule when content could go in multiple chapters.

**Ch. 1 — Introducció, motivació i objectius**
Domain context, problem statement in one sentence, **measurable** objectives (no "improve X" — say "achieve SER ≤ Y on dataset Z"), scope and out-of-scope, methodology summary (one paragraph), document structure (one paragraph).

**Ch. 2 — Estat de la qüestió / State of the Art**
Survey of relevant prior work, **every claim about prior art carries a `\cite{key}` handle** — not just an author name. Existing tools / products and how they fall short for this specific problem. Last paragraph positions the thesis in the gap (the contribution claim). Do not teach basics here; cite a survey if the reader might need one.

**Ch. 3 — Gestió del projecte / Management**
Initial plan (from GEP E2), methodology (Agile/incremental — justify), risk register (likelihood × impact), changes made and *why* (the rubric's "adjustments and justification"), budget (hours × rate + amortisation + indirects) reconciled to 540 h. Identify applicable laws/regulations (IP, data, GDPR) here or in Ch. 7 — pick one place.

**Ch. 4 — Especificació i disseny / Design**
**Decisions and their justification.** What/why, not how. For each major decision, include an "Alternatives considered" subsection naming at least two rejected options with the reason. Architecture diagrams, data flow, interface contracts, vocabularies/schemas at the conceptual level, performance/quality targets.

**Ch. 5 — Desenvolupament / Implementation**
**How the Ch. 4 decisions were realised.** Engineering details a competent reader needs to reproduce or audit. Concrete libraries, versions, hyperparameters, magic numbers. Implementation-forced trade-offs (memory, batch size). Do **not** re-justify the architecture — that's Ch. 4's job.

**Ch. 6 — Experimentació i avaluació / Results**
Experimental protocol (datasets, splits, metrics, hardware, repetitions). Quantitative results in `booktabs` tables. Qualitative results with figures. **Error analysis** — what fails and why. Comparison to SoA where possible.

**Ch. 7 — Anàlisi de sostenibilitat i implicacions ètiques**
Mandatory per Art. 12. Cover three dimensions: **environmental** (training energy, hardware lifecycle), **economic** (cost of development, cost at scale), **social** (who benefits, who is excluded, accessibility). Map to the UN Sustainable Development Goals (SDGs) where applicable. Ethics: dataset provenance, IP of training material, potential misuse.

**Ch. 8 — Conclusions**
- Objectives revisited — explicit list of met / partially met / dropped, one sentence each
- Key quantitative results restated
- Critical assessment and limitations
- Future work — 3–5 concrete, scoped extensions tied to the limitations
- Competences map — the GEI/Computació technical competences declared at TFG inscription, with where in the work each is demonstrated (tribunal looks for this)
- Brief methodological reflection
- Target 4–6 pages, not one. A recap is not a conclusion.

## Writing voice

- **Tense.** Past for work done ("the model was trained on …"); present for what the system does ("the network outputs …"); past for the state of the art ("prior work has shown …").
- **Person.** Pick one — impersonal/passive, "we", or third-person — and stay consistent. **Never first-person singular "I"** in the body (acceptable only in the acknowledgments). "We" is the safe default in a CS thesis even when solo.
- **Hedging.** Every claim is (a) cited, (b) measured in your own results, or (c) explicitly argued. "X is the dominant approach" is a defect without one of these. Soften with "is among the dominant" or add a citation.
- **Definitions.** Define every acronym on first use — "Optical Music Recognition (OMR)". Then use the acronym throughout.
- **Citations.** Every reference to prior work carries a `\cite{key}` handle. "Calvo-Zaragoza showed …" with no citation is a defect; write "Calvo-Zaragoza et al. \cite{calvo2018} showed …".
- **Italics.** Dataset and book titles (*The Real Book*, *PrIMuS*), Latin abbreviations (*e.g.*, *i.e.*, *et al.*). Sparingly otherwise.
- **Numbers.** Digits for measurements and units ("5 ECTS", "12 GB", "180 px"); spell out one–nine in narrative prose ("three layers"); digits for 10+. `siunitx` is loaded — use `\SI{...}{...}` where unit consistency matters.
- **Cross-references.** Every figure, table, and equation is referenced in the text *before* it appears. The project uses `cleveref` — write `\Cref{fig:foo}` (and `\cref` mid-sentence), never `Fig.~\ref{fig:foo}`.
- **Captions.** Full sentences ending in a period. The figure should be understandable from caption alone.
- **Lists.** Prefer prose to bullets in the body. Bullets fit objectives, future work, and reference-style sections only.

## Common defects → fixes

| Defect | Fix |
|---|---|
| Abstract only in English | Add Catalan + Spanish versions, 1–2 pages each (Art. 22) |
| Citation-by-author-name with no `\cite` | Add `\cite{key}` for every prior-work mention |
| Design and Implementation duplicate content | *Why* in Design, *how* in Implementation |
| No "Alternatives considered" subsection | Add one per major design decision |
| Conclusions reads as a recap | Restructure: objectives-revisited + critical assessment + future work + competences |
| Methodology not stated explicitly | Add a methodology subsection in Management |
| Sustainability section is one paragraph | Expand to environmental + economic + social + SDG mapping |
| Vague claims ("CRNNs are dominant") | Add citation, measurement, or hedge |
| First-person singular "I built …" | Switch to "we" or impersonal |
| Captions are noun phrases | Rewrite as full sentences |
| `Fig.~\ref{}` mid-sentence | Use `\cref{fig:...}` (cleveref is loaded) |

## Linked artifacts

- Normativa (authoritative): `latex_documents/gep/normativa-tfe-fib-ca.pdf` (Catalan)
- Progress-review rubric: `latex_documents/progress_review/Rubric.md`
- Sustainability framework: Acord Consell de Govern UPC 07/2023 (referenced in Art. 12)
- GEP deliverables (E1 context+scope, E2 planning, E3 budget+sustainability, E4 final document): `latex_documents/gep/E{1..4}/`
- Current memoir source: `latex_documents/main/main.tex` + `chapters/`
