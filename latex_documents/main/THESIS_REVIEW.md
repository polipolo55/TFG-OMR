# Thesis review — full linear read (2026-06-14)

Reviewer pass over every chapter + appendix of `latex_documents/main/`, plus
mechanical checks (citations, labels, cross-refs) and verification of
code-dependent claims against the actual source. Findings are ordered by
severity. Each item gives `file:line`, the problem, and a concrete fix.

`\todo{}` markers are **not** flagged as defects (intentional WIP per the
project's working notes). They are listed once at the end for completeness.

---

> **STATUS (2026-06-14): A1–A4, B1–B7 FIXED. C and D applied except verify-only
> items.**
>
> **Fixed in C:** C1 (stray `\`), C2 (chord baseline → 69.9% comparable pair),
> C3 (duplicate "most widely circulated" superlative), C5 (SDG cite dropped),
> C7 (embodied-energy "order of magnitude" softened), C8 (incidentals caption
> trimmed), C9 (comparison-table legend → math minus signs), C10 (`dilate_ink`
> wording), C11 (app_b "five worst-case"), C12 (app_b chapter retitled),
> C14 (`**kern`/ABC pairing: SMT→kern, LEGATO→ABC), C15 ("Every" → "Recent").
> **Fixed in D:** D1 (`\textit{The Real Book}`), D2 ("Overall,"), D4 ("real-music"
> dropped), D6 (PRAIG expanded + "University of Alicante"), D7 (systems/staff
> agreement), D8 (accidental persistence "for that pitch, at that octave").
> **Verified already-correct:** D10 (dependency pins match `pyproject.toml`
> exactly: torch ≥2.10, torchvision ≥0.25, OpenCV ≥4.13, NumPy ≥2.4, PyMuPDF ≥1.27).
>
> **STILL OPEN — verify-only / your call (NOT changed):**
> - C4 — intro digitisation-benefit lists (§1.1 vs §1.2) mildly overlap; left as
>   complementary framing (problems vs enabled benefits). Trim if you prefer.
> - C6 — grid carbon factor 0.23 kg CO₂/kWh: confirm it matches the cited
>   REE/MITECO year (it is cited and plausible).
> - C13 — `app_b.tex:159` weight-std range [0.22, 0.35] for *all four* groups:
>   verify against `fig_weight_distributions` (no generator script found to check).
> - C16 — Audiveris version "5.10, the current release" (`06:652`, `06:688`):
>   confirm the version you actually ran.
> - D3 — two uncited `references.bib` entries (`harteltOpticalMedievalMusic2022`,
>   `opticalmusicrecognitionresearchSelfLearningOpticalMusic2019`): harmless
>   (biblatex won't print them); cite or delete at your discretion.
> - D5 — "LilyJAZZ" `\term{}`: left — first use is in the intro and is
>   self-glossing ("the LilyJAZZ font").
> - D9 — reproduced *Black Orpheus* Real Book page (`app_notation` figure): add a
>   source/credit line? (copyright/attribution — your call).
> - D11 — confirm the GitHub repo is public at submission (`app_c.tex:4`).

## A. CRITICAL — factual errors / incoherent numbers (fix before submission)

### A1. ✅ FIXED — Wrong reject-gate threshold value in the hyperparameter table
- **`appendices/app_a.tex:202`** — `min_mean_logprob (CTC gate)` is printed as
  `-0.05`. The deployed value in `models/staff_reject/thresholds.json` is
  **`-0.15`**. Both log-prob rows currently show `-0.05`, which is itself a
  copy-paste tell.
- **Fix:** change the `min_mean_logprob` row to `-0.15`; leave
  `confident_override_logprob` at `-0.05`.

### A2. ✅ FIXED — `calibrate-reject` is described as producing the CTC gate — it does not
This contradicts the project's own hard-constraint #7 and the code
(`src/cli.py:657` copies `DEFAULT_THRESHOLDS.min_mean_logprob`; the deployed
`-0.15` is hand-tuned, not calibrated). `calibrate-reject` only sweeps the four
**geometric** gates (which are checkpoint-*independent*).
- **`appendices/app_a.tex:210`** — "the two log-probability thresholds … must be
  recomputed after every retraining via `… calibrate-reject …`". Wrong for the
  CTC gate.
- **`chapters/05_implementation.tex:532-535`** — "Thresholds … are produced by
  the `calibrate-reject` CLI subcommand … the file is checkpoint-specific and
  must be regenerated whenever the music CRNN is retrained." Misleading: the
  geometric gates are checkpoint-independent; the CTC gate is hand-tuned.
- Softer instances (same root): **`chapters/04_design.tex:307`**
  ("checkpoint-dependent recalibration"), **`chapters/08_conclusions.tex:205-211`**
  ("calibrated against the distribution of the current checkpoint … must be
  re-derived"). §8.3 is the closest to correct ("not self-calibrating").
- **Fix:** state that the four geometric gates are calibrated by
  `calibrate-reject` and are checkpoint-independent, while the CTC log-prob gate
  is a hand-tuned, music-preserving value that is re-tuned **by hand** only if
  real pages start being wrongly rejected.

### A3. ✅ FIXED — Energy figure stated inconsistently (57.5 vs 62.5 kWh)
- **`chapters/07_sustainability.tex:56` and `:111-112`** — energy printed as
  `≈57.5 kWh`, but the derivation at `:112-127` (250 h × 0.25 kW = 62.5 kWh),
  the carbon figure (62.5 × 0.23 = 14.4 kg), and the Final Reflection (`:361`)
  all use **62.5 kWh**. `57.5` is simply wrong.
- **Fix:** change both `57.5` occurrences to `62.5`.

### A4. ✅ FIXED — Conclusions §8.5 corpus/image counts don't reconcile
- **`chapters/08_conclusions.tex:376-380`** — "PrIMuS contributes 87 678
  monophonic incipits, each re-rendered in LilyJAZZ and paired with a synthetic
  scanned copy, yielding on the order of 170 000 training images **after
  filtering**."
- Ground truth (from the project's own records and the 4 608-test / 36 873-train
  figures in Ch. 6): 87 678 is the **raw** count; the three domain filters remove
  18 532 + 23 056, leaving **46 089 retained** samples
  (36 873 train / 4 608 val / 4 608 test). So after filtering it is *not* 87 678;
  and 46 089 × 2 (clean+scanned) = 92 178 — the "170 000" only appears if the
  two extra train-only scan variants are counted, which the sentence doesn't
  mention. As written, the derivation (87 678 × 2 ≈ 170 000) silently ignores
  the filtering it claims to include.
- **Fix:** restate precisely, e.g. "87 678 raw incipits; 46 089 retained after
  the three domain filters (36 873 train / 4 608 val / 4 608 test); with a clean
  render, a scan-simulated render, and two extra scan variants per training
  sample, ≈170 000 rendered training images."

---

## B. MAJOR — cross-chapter consistency & accuracy

### B1. ✅ FIXED — "Chord OCR" vs "chord CRNN" — the thesis contradicts itself on what the chord stream is
> Resolution: delivered-component refs renamed to "chord recognition / chord CRNN"
> (intro scope+pipeline+approach, §5.4 title, §6.9 title+caption+summary,
> conclusions O6/bullet/network/stream, app_a §title+caption, sustainability run
> list). "OCR" kept where it is task-framing (intro §1.2 "call for text OCR",
> design §4.2 "text-like OCR", app_notation primer), the rejected original plan
> (design §4.8.2 "Pipeline OCR"), or the EasyOCR reject gate.
The code is unambiguous: chords are recognised by a **character-level CRNN-CTC
network** (`src/omr_pipeline/chord_recognizer.py`, `src/CRNN_CTC/chord_train.py`
— the same `CRNN` class, ResNet18 + BiLSTM + CTC). EasyOCR is used **only** as a
text-area detector in the staff-reject gate, never for chord recognition.

- The **abstract** (all three languages), **SoA** (`02_soa.tex:319` "chord
  network"), **Design** (`04_design.tex:64-67` decision register: "Separate CRNN
  stream", and §4.8.2 `:799-804` which **explicitly rejects** "general OCR
  (EasyOCR, Tesseract)"), and the **Implementation body** (`05:378-379` "reuses
  the CRNN-CTC machinery") all correctly call it a CRNN.
- But these label it **"OCR"**, conflating the chosen CRNN with the rejected
  off-the-shelf-OCR alternative:
  - `01_introduction.tex:132-134` ("call for text OCR"), `:141-142` ("building a
    separate OCR pathway for chord symbols"), `:200` ("chord OCR"), `:228`
    ("Chord symbol extraction via an OCR-based module").
  - `05_implementation.tex:376` — **section title** "Chord OCR Implementation"
    (body immediately says it is a CRNN).
  - `06_results.tex:870` (subsection title "Chord OCR"), `:887` (table caption
    "Chord-OCR evaluation"), `:969` ("The chord OCR stream").
  - `08_conclusions.tex:57` (O6 "chord OCR"), `:122` ("Chord OCR:"), `:389`
    ("The chord-OCR network"), `:416` ("the chord-OCR stream").
- **Why it matters:** a reader who sees Design reject "general OCR" and then sees
  the delivered component called "chord OCR" cannot tell whether the system uses
  a CRNN or an OCR engine — especially since EasyOCR *is* used elsewhere.
- **Fix (thesis-wide):** call it the **chord-recognition network / chord CRNN**
  consistently; reserve "OCR" for the task type or for EasyOCR-in-the-reject-gate.
  Minimum: fix the Intro phrasings (B1 outliers), the §5.4 heading, and the §6.9
  heading; add one sentence clarifying it is a CRNN, not an off-the-shelf engine.

### B2. ✅ FIXED — Chord fine-tune `--synth-weight`: thesis says 0.4 "(the default)"; code default is 0.5
> Resolution: user confirmed the default (0.5) was used; both occurrences
> (impl §5.4, app_a) changed 0.4 → 0.5, so "(the default)" is now correct.
- **`05_implementation.tex:454-456`** — "synthetic strips weight 0.4 (the default
  `--synth-weight`)"; **`appendices/app_a.tex:167`** — "synth 0.4".
- Code: `src/CRNN_CTC/chord_finetune.py:12` "(default 0.5)"; the documented
  example uses `--synth-weight 0.5`.
- **Fix:** confirm the value actually used in the reported fine-tune. If 0.5,
  change both occurrences. If 0.4 was an explicit override, drop "(the default)".

### B3. ✅ FIXED — Garbled grammar in the Catalan abstract (a tribunal-read mandatory section)
- **`00_abstracts.tex:16`** — "els acords escrits **a al** damunt" → duplicated
  preposition. Fix: "els acords escrits **al** damunt" (or "a sobre").
- **`00_abstracts.tex:22-23`** — "la SER melòdica baixa **fins en el primer cas i
  en el segon cas al** 0,10% i el 0,14%" → broken word order (compare the clean
  Spanish at `:55`). Fix: e.g. "la SER melòdica baixa **fins al** 0,10% i al
  0,14% **en el primer i el segon cas, respectivament**."
- **`00_abstracts.tex:23`** — "el 72,7% en **el conjunt que simulat**" → broken.
  Fix: "el conjunt **simulat**" (or "el conjunt que **simula l'escaneig**").

### B4. ✅ FIXED — Unsupported "sub-percent SER" attributed to the published baseline (SoA)
- **`02_soa.tex:151`** — "where the CRNN-CTC baseline already attains sub-percent
  SER …". No published CRNN-CTC baseline reports sub-percent SER on both PrIMuS
  encodings; the only sub-percent SER in this thesis is its **own** result.
  (Cf. the SoA comparison table `06:630` which itself quotes Calvo-Zaragoza at
  **0.8%** semantic — that *is* sub-percent, but for one encoding only.)
- **Fix:** cite the specific figure/encoding or hedge to "low single-digit SER".

### B5. ✅ FIXED — App B: "highest per-sample SER in the test set" is unsupported
- **`appendices/app_b.tex:84-85`** — rank-4 caption: "SER 0.167 — the highest
  per-sample SER in the test set." The five figures are selected by **edit
  distance**, not SER (`:4-5`, `:19`), so the true max-SER sample need not be in
  this set; a short sequence (min length ≈20 tokens, per `06:386`) with a few
  errors can exceed 0.167.
- **Fix:** soften to "the highest SER among these five worst-by-edit-distance
  samples", or verify it is the global max and say so.

### B6. ✅ FIXED — Undefined acronyms in Management
- **`03_management.tex:180,192,196`** — **GEP** used but never expanded anywhere
  in the thesis. Expand on first use.
- **`03_management.tex:486`** — **ICT** ("INE-EAES … ICT salary survey") never
  expanded. Expand on first use.

### B7. ✅ FIXED — SoA metric acronyms used before definition; SER defined twice
- **`02_soa.tex:141`** uses **TEDn** and **OMR-NED** before they are defined at
  `:280-286`. Add a one-line gloss or a `\Cref{sec:soa-metrics}` pointer at first
  use.
- **`02_soa.tex:108` and `:266`** — **SER** is fully expanded twice. Keep the
  expansion once (first use) and use bare "SER" afterwards.

---

## C. MINOR — prose, redundancy, precision

- **`06_results.tex:23`** — stray trailing backslash: `…experimental front.\`
  (a control-space before the paragraph break). Delete the `\`.
- **Chord-improvement baseline quoted inconsistently.** The strictly comparable
  pair (both on the 29 held-out strips) is **69.9% → 5.9%** (`06:927`), but the
  summary (`06:971`) and `08_conclusions.tex:122-125` quote **74.0% → 5.9%**
  (74.0% is the zero-shot over all 291 strips). Pick one framing or state both
  denominators explicitly.
- **`01_introduction.tex:16` vs `:60`** — "one of the most widely circulated
  fake-books" then "the most widely circulated collection of jazz lead sheets".
  Mild near-contradiction; align the hedge.
- **`01_introduction.tex:38-40` vs `:71-74`** — the digitisation-benefits list
  (transposition / MIDI / analysis / search) is stated twice. Trim one.
- **`07_sustainability.tex:16-17`** — the four SDGs are cited to
  `gepSustainabilityReport2018`; the SDGs are a UN framework. Cite a UN source or
  drop the cite for that specific claim. (The GEP cite is correct for the matrix
  at `:7`.)
- **`07_sustainability.tex:128-131`** — grid factor 0.23 kg CO₂/kWh is on the
  high side for the recent Spanish mix; confirm it matches the cited REE/MITECO
  year, and reconcile CO₂ (factor) vs CO₂e (result) usage (`:129` vs `:132`).
- **`07_sustainability.tex:159-160`** — "embodied … energy would dwarf … by an
  order of magnitude" is an uncited quantitative claim tied to the open `\todo`;
  if the source isn't found, soften to qualitative.
- **`03_management.tex` incidentals** — the explanation prose (`:571-579`) is
  near-verbatim repeated in the table caption (`:608-613`). Keep it in one place.
- **`03_management.tex` comparison-table legend** — `--` ("poor", en-dash) vs
  `-{}-` ("very poor", two hyphens) are visually confusable and the *worse*
  rating is the *longer* mark. Use unambiguous tokens.
- **`appendices/app_a.tex:100`** — `dilate_ink` described as "erosion iterations"
  (name/description mismatch). Clarify ("morphological iterations that thicken
  the ink").
- **`appendices/app_b.tex:4`** — "collects the five worst-case predictions" but
  only ranks 2–5 (four figures) are here; rank-1 is in the body. Reword to
  "collects the remaining four of the five worst-case predictions".
- **`appendices/app_b.tex:143` (Model Weight Distribution)** — thematically
  misplaced under a "Qualitative Predictions" appendix. Move to the
  hyperparameters/diagnostics appendix or retitle the chapter.
- **`appendices/app_b.tex:159`** — claim that *all four* parameter groups have
  std in [0.22, 0.35] is suspect (conv/BN/LSTM usually < 0.1). Verify against
  `fig_weight_distributions`.
- **`02_soa.tex:218` vs the formats table (`:243-244`)** — prose lumps `**kern`
  and ABC onto both SMT and LEGATO, but the table pairs SMT→`**kern`,
  LEGATO→ABC. Fix the citation-to-format pairing.
- **`02_soa.tex:199-202`** — "Every recent end-to-end system … addresses … by
  synthetic pretraining followed by fine-tuning on a small annotated target set"
  is an over-strong universal; hedge to "Several".
- **`06_results.tex:652`** — "Audiveris … v5.10, the current release" — verify
  the version string (5.10 vs 5.1.0) and that it is current at submission.

---

## D. NITS

- **`00_abstracts.tex:75-76`** — English "the *Real Book* collection" drops "The"
  vs Catalan/Spanish "*The Real Book*". Align title rendering.
- **`00_abstracts.tex:88`** — "As overall results, 73.7% …" → "Overall, 73.7% …".
- **`references.bib`** — two entries are never cited (harmless with
  `sorting=none`, but dead): `harteltOpticalMedievalMusic2022`,
  `opticalmusicrecognitionresearchSelfLearningOpticalMusic2019`. Cite or remove.
- **`02_soa.tex:100`** — "real-music … incipits" → "real (non-synthetic) …".
- **`02_soa.tex:196`** — "LilyJAZZ" used without `\term{}` / gloss on first use.
- **`07_sustainability.tex:296`** — "PRAIG" acronym not expanded.
- **`appendices/app_notation.tex:239`** — number disagreement: "systems of
  single-voice treble-clef staff" → "systems, each a single-voice treble-clef
  staff".
- **`appendices/app_notation.tex:54`** — accidental-persistence rule could add
  "(for that note at that octave)".
- **`appendices/app_notation.tex:263`** — reproduced Real Book page ("Black
  Orpheus") may need a source/credit citation.
- **`appendices/app_a.tex:228`** — verify the PyTorch ≥2.10 / torchvision ≥0.25 /
  OpenCV ≥4.13 / NumPy ≥2.4 pins against `poetry.lock`/`pyproject.toml`.
- **`appendices/app_c.tex:4`** — confirm the GitHub repo is public at submission.

---

## E. Verified correct (no action — recorded for confidence)

- **Headline metrics are consistent** across abstract / Ch. 6 table / Ch. 8:
  aggregate SER 1.17% clean / 1.23% scanned; melodic 0.10% / 0.14%;
  perfect-transcription 73.7% / 72.7%; 4 608-sample test split.
- **Acceptance criteria vs results:** all four Ch. 4 §4.1 targets met
  (clean <2%, scanned <3% with gap <1 pp → 0.06 pp, melodic <0.5%,
  perfect >60% scanned → 72.7%).
- **Error-decomposition arithmetic** is internally consistent
  (3.4% + 38.8% + 57.8% = 100%; 119 subs = 3.4% × 3506; measure+tie ≈ 90%;
  77% + 13% = 90% in §8.2).
- **Management budget** fully reconciles: 509 h + 31 h reserve = 540 h (Art. 17);
  all per-WP and per-role hours sum to 509; all cost rows sum to €16 097.16;
  contingency/general/incidentals all check out.
- **Mechanical checks clean:** no undefined references, no undefined citations,
  no duplicate labels, no broken `\cref`/`\cite` (LaTeX log shows zero
  undefined). All cited keys resolve.
- **Objectives ↔ results** all accounted for; O7 honestly "partially met" (no
  held-out real melody split — 36 staves all in train), consistent with §6.8.
- **Chord eval counts** consistent (291 = 262 fine-tune + 29 held-out; 90/10).
- **Audiveris staff counts** consistent (9 detected, 2 rejected, 7 recognised;
  36 chords) across §6.7, §6.8.2, and the web-UI caption.

---

## F. Open `\todo{}` markers (intentional WIP — not defects)

- `06_results.tex:438` — no-augmentation ablation re-run against epoch-83.
- `06_results.tex:440` — virtual-header-injection ablation.
- `06_results.tex:446` — re-confirm hand-picked qual figures against epoch-83.
- `06_results.tex:737` — zero-shot real-melody SER (hand-label ~50 staves).
- `07_sustainability.tex:190` — embodied-carbon citation for a consumer GPU.

(The `06:713-723` reminder comment documents the plan to flip O7 to "met" once
the real melody split is labelled.)
