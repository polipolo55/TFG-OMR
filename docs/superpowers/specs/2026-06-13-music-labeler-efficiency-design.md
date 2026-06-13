# Design: Music-labeler efficiency improvements

**Date:** 2026-06-13
**Status:** Approved (design)
**Scope:** `static/music_labeler.html` + one small endpoint in `src/api/main.py`.
No change to training, the extractor, the saved-label format, or the model.

## Problem

Hand-labeling real staff crops is slow and error-prone in the current UI:
typos silently become `<unk>` (observed: `eight`, `mesure`), the render only
updates on demand, misread durations aren't caught until they look wrong, and
every note is typed out longhand (`pitch:C octave:5 quarter …`).

## Approach (chosen)

Extend the existing `<textarea>` editor with live helpers rather than rewrite it
into a token-chip editor (the user edits LMX fluently; a chip editor is a large
rewrite for little gain). Four additive features.

### 1. Live token validation (hard block)

- New `GET /api/music-labeler/vocab` → `{"tokens": [...]}` read from
  `data/vocab/primus_lmx.txt` (the authoritative training vocabulary; anything
  outside it encodes to `<unk>`). Loaded once on page init.
- On every editor `input`, split into tokens and compute the set not in the
  vocab. A status line under the editor lists them in red: `Unknown: eight, mesure`.
- **Save & next is disabled while any unknown token is present** (button
  greyed + `Ctrl+Enter` no-ops with the status message). **Skip is still
  allowed** (skip saves no label).
- Special tokens that are valid but absent from the vocab file by convention
  are still accepted: the header tokens the grammar uses (`clef:G2`,
  `key:fifths:N`, `time`, `beats:N`, `beat-type:N`) and `measure`, `rest`,
  `dot`, `tied:start`, `tied:stop`, accidentals — these already appear in the
  vocab file, so no special-casing is expected, but if validation flags a token
  that the model genuinely emits, the vocab file is the bug, not the label.

### 2. Auto-preview (debounced)

- Debounced (~700 ms after typing stops) call to the existing `/render`.
- Skips auto-render when validation is failing (render would drop the bad
  tokens and mislead). Respects the existing `renderInFlight` guard.
- A checkbox "Auto-preview" (default on, persisted in `localStorage`) to disable
  it for manual-only rendering. `Ctrl+P` still forces a render.

### 3. Beat-sum / bar checker (informational)

- Pure JS. Split the token stream on `measure`; for each bar sum note/rest
  durations using a duration→whole-note map mirroring `_DUR_WHOLE` in
  `lilypond_render.py` (including `dot` = ×1.5 per dot).
- Expected bar length from the header `time` token, or the render bar-layout
  selector for header-less staves.
- Render a compact line: `bar1 3/4 (pickup) · bar2 4/4 ✓ · bar3 3/4 ⚠`.
  This is **informational only, never blocks** — pickups and cropped fragments
  legitimately don't sum. First and last bars that are short are tagged
  `(pickup)` / `(short)` rather than `⚠`; only interior short/over bars get `⚠`.

### 4. Atomic token buttons + hotkeys

A palette inserting one token at the cursor (textarea selection), grouped:

| Group | Buttons | Inserts |
|-------|---------|---------|
| Structure | measure · rest · dot | `measure` / `rest` / `dot` |
| Pitch | C D E F G A B | `pitch:<letter>` |
| Octave | 3 4 5 6 7 | `octave:<n>` |
| Duration | whole half quarter eighth 16th 32nd 64th | the word |
| Accidental | flat sharp natural | the word |
| Tie | tied:start tied:stop | the token |

Insertion adds a trailing space and keeps focus + cursor after the inserted
token, so chaining buttons/hotkeys builds a note (`pitch:C` `octave:5`
`quarter`) without leaving the keyboard.

**Hotkeys** (Alt-based so they fire while the textarea is focused without
clashing with existing `Ctrl+Enter/P/R` or normal typing):

- `Alt+C/D/E/F/G/A/B` → `pitch:<letter>`
- `Alt+3/4/5/6/7` → `octave:<n>`
- `Alt+M` → `measure`, `Alt+R` → `rest`, `Alt+.` → `dot`
- Durations / accidentals / ties: buttons only (no clash-free hotkey letters
  remain; durations are lower-frequency than pitch/octave).

Keymap is documented in the on-page help row and adjustable later.

## Rejected alternatives

- **Token-chip / structured editor.** Large rewrite, higher risk; current
  free-text editing already works.
- **Server-side validation/beat-sum.** Adds a round-trip per keystroke; the
  vocab and duration math are tiny and fine client-side (vocab fetched once).

## Out of scope

- Zoom/pan on the scan crop (deferred — not selected).
- Any change to saved-label format, training, or the model.

## Verification

- `GET /api/music-labeler/vocab` returns the vocab tokens; module imports.
- Typing `eight` → flagged red, Save disabled; fixing to `eighth` → re-enabled.
- Auto-preview re-renders after edits; off when validation fails; toggle works.
- Beat-sum line: pickup bar tagged `(pickup)`, full bar `✓`, a deliberately
  wrong interior bar `⚠`.
- Each atomic button and hotkey inserts the right token at the cursor and keeps
  focus.
