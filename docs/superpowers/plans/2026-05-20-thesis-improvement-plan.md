# Thesis Master Improvement Plan

**Generated:** 2026-05-20  
**Audit scope:** All chapters (Ch00–Ch08) + appendices (App A–C)  
**Build status:** ✅ Clean — 81 pages, 0 undefined references, 0 undefined citations

---

## What Was Done in This Audit

All chapters have been audited and edited in place. Summary of completed work:

| Chapter | Status before | Status after |
|---------|--------------|-------------|
| Ch00 (abstracts) | Missing melodic SER, imprecise arch description | All three languages updated with full headline numbers + two-stream description |
| Ch01 (introduction) | One ref? TODO; bare \ref{} calls | TODO resolved; all \Cref{}; 3 citations added |
| Ch02 (SoA) | Missing multi-cite, OLIMPIC naming inconsistency | Citations fixed; Summary signpost added; 2 verify TODOs added to table |
| Ch03 (management) | Risk matrix placeholder; bare \ref{} calls | TikZ 3×3 risk matrix implemented; all \Cref{} |
| Ch04 (design) | 3 ref? TODOs; 2 figure placeholders | 3 bib entries added (Graves 2006, He 2016, Hochreiter 1997); 2 TikZ figures added; all \Cref{} |
| Ch05 (implementation) | 2 ref? TODOs; 3 figure placeholders | 2 bib entries added (Loshchilov 2019, Smith 2019); 2 TikZ figures + 1 actionable TODO; all \Cref{} |
| Ch06 (results) | Bare \ref{} calls; vague experiment TODOs | All \Cref{}; 3 experiment TODOs expanded with exact specs; 2 design cross-refs added |
| Ch07 (sustainability) | Bare \ref{} calls | All \Cref{}; arithmetic verified ✅ |
| Ch08 (conclusions) | Stub (~36 lines) | Full chapter written (~730 words prose + 5 future-work bullets expanded) |
| App A–C | Bare \todo{} stubs | Replaced with specific, actionable placeholders |

**New bib entries added:** Graves 2006 (CTC), He 2016 (ResNet), Hochreiter 1997 (LSTM), Loshchilov 2019 (AdamW), Smith 2019 (OneCycleLR)  
**New figures added:** Risk matrix (Ch03), pipeline overview (Ch04), CRNN-CTC detail (Ch04), data pipeline flow (Ch05), two-stream inference (Ch05)

---

## Remaining TODOs (14 total)

### Blocked on running experiments / generating data (cannot be completed without author action)

| # | Location | TODO | What is needed |
|---|----------|------|----------------|
| 1 | `06_results.tex:401` | Zero-shot evaluation on real pages | 50 annotated staves from Real Book + `--split real_pages` config |
| 2 | `06_results.tex:405` | Fine-tuning quantitative comparison | Complete fine-tuning experiment + `data/chord_real/labels.jsonl` |
| 3 | `06_results.tex:409` | Full-page inference figure (Autumn Leaves) | Run pipeline on real page + generate overlay figure |
| 4 | `06_results.tex:346` | Worst-case SER example | Implement `--export-worst` flag in `src/cli.py evaluate` |
| 5 | `05_implementation.tex:235` | Augmentation samples figure | Run `scripts/generate_augment_samples.py` (may need to create this script) |

### Blocked on verifying values (author needs to check logs/papers)

| # | Location | TODO | What to check |
|---|----------|------|---------------|
| 6 | `06_results.tex:43` | Training loss plateau value (~0.06) | Check `training_log.csv` or TensorBoard |
| 7 | `06_results.tex:180` | Substitution share (~5.4% of edit budget) | Run `scripts/evaluate_full.py` and check per-op breakdown |
| 8 | `02_soa.tex:291` | DoReMi dataset size (~6,432 pages, ~1M objects) | Check arXiv:2107.07786 Table 1 |
| 9 | `02_soa.tex:292` | OLiMPiC split counts (~17,945 synth, 2,931 scanned) | Check Mayer et al. ICDAR 2024 Table 1 |

### Blocked on producing content (appendices)

| # | Location | TODO | What to produce |
|---|----------|------|----------------|
| 10 | `app_a.tex:3` | Hyperparameter grid search table | Run `scripts/hyperparameter_search.py` |
| 11 | `app_b.tex:3` | Worst-5 qualitative predictions | Same as #4 above |
| 12 | `app_c.tex:3` | Three key code snippets | Extract + format the three scripts |

---

## Priority Tiers for Remaining Work

### 🔴 Critical — Must fix before submission

**1. §6.5 Domain Gap — Three experiment results (TODOs #1, #2, #3)**  
This section currently contains no results — only well-specified TODOs. It is the weakest part of the thesis. Even partial results (zero-shot SER on a handful of real pages) would change this from a gap to a completed section. Without any results, the section undermines the scope of Objective O7 ("characterise the domain gap").  
- **Action:** Run zero-shot evaluation on any available Real Book scans, even if N=10–20 staves.

**2. Worst-case example figure (TODO #4)**  
The §Qualitative Results section ends without a worst-case example, which is specifically useful to assess failure modes. The `--export-worst` flag does not yet exist in `src/cli.py evaluate`.  
- **Action:** Either implement `--export-worst` (small addition to evaluate subcommand) or manually identify one high-SER prediction and include it.

**3. Augmentation samples figure (TODO #5)**  
`fig:augment-samples` is the only remaining FIGURE placeholder in the main body (all others have been implemented). Without it, §5.3 "Augment" has an empty placeholder box in the PDF.  
- **Action:** Create or run `scripts/generate_augment_samples.py`; produce a 6-panel figure (clean + 5 distortion stages).

### 🟡 Important — Significantly improves quality

**4. Verify training log values (TODOs #6, #7)**  
Two values in §6.1 are flagged as unverified: the 0.06 plateau and the ~5.4% substitution share. Both should be read from actual evaluation output, not approximated.

**5. Verify dataset table claims (TODOs #8, #9)**  
The Ch02 dataset table has two cells flagged `\todo{verify:}` — DoReMi and OLiMPiC sizes. These are cited from known papers; the author should spend 10 minutes looking up the correct numbers and removing the TODO markers.

**6. App C code snippets**  
Appendix C is entirely a placeholder. Including the three key scripts (render loop, augmentation config, grammar BNF) would give the appendix real value and serve as reference documentation.

**7. LEGATO error reduction claim (Ch02 §2.3)**  
Ch02 states "absolute error reductions of 47%–68%" for LEGATO. This is likely relative (not absolute). Verify against arXiv:2506.19065 and fix if needed — otherwise a reviewer will flag it.

### 🟢 Nice-to-have — Polish

**8. Harden TikZ library loading in main.tex**  
`\usetikzlibrary{positioning,arrows.meta,calc}` is currently loaded transitively through `pgfgantt`. Add it explicitly (after line 33 in `main.tex`) so the build stays robust if `pgfgantt` is removed.

**9. Beam-search ablation table**  
The beam-search ablation paragraph in §6.2 claims "6× cost" and "≤1 edit per 1000 tokens". A 2–3 row table (beam widths 1, 5, 10 × SER and decode time) would make this verifiable.

**10. Data-filter config flags in hyperparams table**  
`tab:hyperparams` omits `filter_multi_staff`, `filter_non_leadsheet_clef`, `filter_unusual_time`. Adding a short row group for domain-filter booleans completes the reproducibility picture.

**11. Rename subfigure labels `_pred` → `_rendered`**  
`fig:qual-perfect-pred` and `fig:qual-median-pred` (subfigure anchor labels in §6.3) should be renamed to `fig:qual-perfect-rendered` and `fig:qual-median-rendered` to match the actual file names on disk. Low risk; cosmetic only.

**12. Economic dimension: per-page correction cost**  
Ch07 §Economic dimension discusses cost but does not state a per-page number. Decide on an "expected output pages" denominator and add: "At the total project cost of €16,097.16, automated transcription brings the expected marginal cost to approximately €X per page, compared to €Y for manual entry."

---

## Cross-Chapter Narrative Consistency Check

The objectives → design → results → conclusions chain was verified to be coherent:

| Objective | Design ref | Results ref | Conclusions verdict |
|-----------|-----------|-------------|---------------------|
| O1: Transcribe monophonic staves to LMX | §Design §Model | §6.2 SER 1.19% | ✅ Met |
| O2: SER < 2% on PrIMuS | §Design §Training | §6.2 SER 1.19% clean | ✅ Met |
| O3: Scan augmentation strategy | §Design §Data | §6.2 scanned SER 1.28% | ✅ Met |
| O4: Two-stream melody+chord | §Design §Two-stream | §6.3 qualitative | ✅ Met |
| O5: FastAPI web service | §Impl §API | (no quantitative results) | ✅ Implemented |
| O6: Standard OMR metrics | §Design §Eval | §6.1–6.4 | ✅ Met |
| O7: Characterise domain gap | §Design §Domain-gap eval | §6.5 (pending) | ⚠️ Partially met — experiments pending |

One broken link: **Ch08 §Conclusions** states O7 as "partially met" and defers to §6.5. This is correct only if §6.5 is clearly framed as planned work. The expanded TODOs in §6.5 now make this framing explicit — however the section still has zero results. Add one paragraph of framing prose to §6.5 (even without numbers) before submission.

---

## Final Build State

```
latexmk -C && latexmk -pdf -interaction=nonstopmode main.tex
→ main.pdf: 81 pages
→ Undefined references: 0
→ Undefined citations: 0
→ LaTeX errors: 0
```

All previously failing `\citet{}` calls replaced with `\textcite{}`. All previously missing citation keys added. All `\Cref{}` cross-references resolve. TikZ figures compile without errors.
