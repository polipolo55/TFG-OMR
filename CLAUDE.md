# CLAUDE.md

TFG-OMR — Optical Music Recognition for monophonic jazz lead sheets (The Real Book).
Bachelor's thesis (TFG) at FIB/UPC by Pol Casanovas Puig.

---

## Required Reading — Do This First

**Every agent must read `docs/overview.md` before doing any work.**
It has the architecture diagram, source map, and design decisions. Do not skip it.

Then read the doc most relevant to your task:

| Task | Read |
|------|------|
| General context / architecture | `docs/overview.md` |
| Data generation, rendering, augmentation | `docs/data_pipeline.md` |
| CRNN model, backbones, CTC, vocabulary | `docs/model.md` |
| Training loop, splits, hyperparameters, checkpoints | `docs/training.md` |
| Inference pipeline, staff detect, chord OCR | `docs/inference_pipeline.md` |
| LMX token grammar and encoding rules | `docs/lmx_format.md` |
| CLI subcommands and flags | `docs/cli.md` |
| FastAPI endpoints and response format | `docs/api.md` |
| Config dataclass fields and defaults | `docs/configuration.md` |

---

## Docs Self-Correction Protocol

The docs in `docs/` can go stale when code changes. Apply this protocol before acting on any doc claim:

1. **Verify before acting.** If a doc names a function, field, file path, or class — check it exists in the actual code before relying on it. A doc that says "function X does Y" is a claim about a past state, not a guarantee.

2. **Code is authoritative.** When a doc conflicts with the source, trust the source.

3. **Fix stale docs immediately.** If you find a discrepancy between a doc and the code, update the doc file before proceeding with your task. Do not leave stale documentation behind.

4. **Update docs after changes.** If your task changes architecture, conventions, config fields, CLI flags, or behavior — update the relevant `docs/` file as part of the same task.

---

## Hard Constraints — Never Violate These

**1. `lilypond_render.py` is the single source of truth for music notation mappings.**
All clef → LilyPond, key → `\key`, and duration → LilyPond tables live in
`src/CRNN_CTC/lilypond_render.py`. Never define or copy these tables elsewhere.
`generate_realbook.py` and `semantic_to_lmx.py` both import from there — that is what
keeps PNG renders and LMX labels in sync. A duplicate table is a bug waiting to happen.

**2. CTC blank is always index 0.**
`Vocabulary` hardcodes `<blank>=0`, `<pad>=1`, `<unk>=2`. The model's output layer,
CTC loss, and all decode paths depend on this. Never change it.

**3. `Config` is serialized inside every checkpoint.**
Every field you add, rename, or remove breaks checkpoint loading for existing runs.
If you must change `Config`, handle backward compatibility or document the breaking change explicitly.

**4. Sample filtering happens at dataset construction, not in the data scripts.**
`generate_realbook.py`, `semantic_to_lmx.py`, and `augment_scanned.py` write everything.
The three domain filters (`filter_multi_staff`, `filter_non_leadsheet_clef`,
`filter_unusual_time`) are applied in `src/CRNN_CTC/dataset.py` via `Config` flags at
training time.  See `docs/overview.md` → "Domain Specification" for the rationale.
Do not add filtering logic to the data processing scripts.

**5. `filter_unusual_time` and `grammar_fix.py` must agree on allowed time signatures.**
The allowed set is `{4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8}` and is defined in both
`src/CRNN_CTC/dataset.py` (`_COMMON_TIME_SIGS`) and `src/omr_pipeline/grammar_fix.py`
(`_COMMON_TIME_SIGS`). If you add or remove a time signature from one, update the other immediately.

**6. Always use `poetry run` to execute Python.**
The project uses Poetry for dependency isolation. Never run `python src/...` directly;
always `poetry run python src/...`.

**7. CTC mean-logprob threshold in `staff_reject.py` is checkpoint-dependent.**
The CTC-confidence gate uses `min_mean_logprob` calibrated against the current
CRNN checkpoint's log-prob distribution. After every CRNN re-train, re-run
`poetry run python src/cli.py calibrate-reject ...` and commit the updated
`models/staff_reject/thresholds.json`.

---

## Non-Obvious Conventions

- **ResNet18 is the default backbone; VGG is legacy.** New experiments use `backbone="resnet18"`.
  VGG is kept for backward compatibility with old checkpoints only.

- **Multiprocessing, not threading.** All CPU-bound stages (rendering, conversion, augmentation,
  vocab building) use `multiprocessing`, not `concurrent.futures.ThreadPoolExecutor`. Python's
  GIL makes threads useless for compute-bound work.

- **LMX accidentals are display-only.** A `flat` token means "draw a flat sign on this note",
  not "this pitch is a half-step lower". The pitch is already encoded in `pitch:X`. Do not
  infer pitch from accidental tokens. Accidental tokens are bare words: `flat`, `sharp`, `natural`
  (no `acc:` prefix). They appear **after** the duration in the token stream.

- **Clef normalization is lossy but intentional.** C1, C2, F3 clefs are converted to G2 (treble)
  during rendering. Pitches are preserved via ledger lines, but the clef identity is gone in the
  LMX label. Do not try to recover the original clef.

- **`filter_non_leadsheet_clef=True` is on by default** — it removes every clef except G2 (treble),
  which is the only clef the Real Book uses.  Disabling it does not generalise the model;
  it only forces it to spend capacity on visual patterns that never appear at inference.

- **Header-less continuation staves are first-class `__nh` twin samples, not a crop.**
  `generate_headerless_twins.py` renders a fraction of treble samples with the clef and
  time-signature glyphs hidden (key signature kept) and writes a matching label with the
  clef/time tokens removed, so image and label are aligned by construction. This step is
  **stage 3 of `cli.py pipeline` / `pipeline-train`** (before augment). `strip_header_prob`
  is DEPRECATED/inert (kept on `Config` only for checkpoint compatibility). Use
  `--force-all` to re-render twins together with clean and scanned data.

- **Vocabulary file excludes the three special tokens.** `<blank>`, `<pad>`, `<unk>` are not written
  to the vocab text file — they are injected at indices 0, 1, 2 by `Vocabulary` in code. Line N
  in the file maps to index N+3.

---

## Common Pitfalls

- **Changing image height without updating `max_source_height`.** `img_height=128` and
  `max_source_height=180` are separate concerns: the former is the training resize target;
  the latter is the multi-staff filter threshold (measured on the *original* pre-resize image).
  Do not conflate them.

- **Adding a new LMX token without rebuilding the vocabulary.** New tokens not in `vocab_path`
  will map to `<unk>` at training time. If you add tokens to the LMX grammar, rebuild vocab
  with `poetry run python src/cli.py vocab ...` and retrain from scratch.

- **Assuming input lengths are always ≥ label lengths.** `CTCLoss(zero_infinity=True)` silently
  ignores samples where the CTC input length is shorter than the label. If SER is unexpectedly
  low on training set and high on validation, check for label sequences that are too long for
  the image width after width compression (W/4).

- **Editing `generate_realbook.py` pitch logic without updating `semantic_to_lmx.py`.** Both
  scripts interpret PrIMuS pitch strings. If you fix a pitch parsing bug in one, check the other.

- **Running notebooks without `poetry run jupyter lab`.** The notebook kernel must use the
  Poetry virtualenv. Launching Jupyter from outside Poetry will miss all project dependencies.

---

## How to Validate Work

After making changes, verify with the minimum relevant check:

```bash
# Syntax / import check (fast):
poetry run python -c "from src.CRNN_CTC.model import CRNN; print('ok')"

# Config loads without error:
poetry run python -c "from src.CRNN_CTC.config import Config; Config()"

# Vocab round-trips correctly:
poetry run python -c "
from src.CRNN_CTC.vocab import Vocabulary
v = Vocabulary.from_file('data/vocab/primus_lmx.txt')
assert v.encode(['measure'])[0] >= 3
print('vocab ok, size:', len(v))
"

# Full evaluate on a small val slice (if checkpoint exists):
poetry run python src/cli.py evaluate \
  --checkpoint models/latest/best_model.pt \
  --split val --beam-width 1

# API smoke test:
poetry run python src/cli.py api &
curl -s http://localhost:8000/ | python -m json.tool
```

---

## Environment

- **Python:** 3.14 (managed by Poetry)
- **Package manager:** Poetry — `poetry install`, `poetry run <cmd>`
- **GPU target:** NVIDIA RTX 3060, 12 GB VRAM — batch size 16 fits comfortably
- **LilyPond** must be installed system-wide and on `PATH` for rendering stages
- **EasyOCR** downloads model weights on first use (~200 MB); ensure internet access or pre-cache
