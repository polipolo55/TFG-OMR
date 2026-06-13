# Design: Header-inject re-predict helper for the music labeler

**Date:** 2026-06-13
**Status:** Approved (design)
**Scope:** `static/music_labeler.html` + `src/api/main.py` only. No change to training,
the extractor, the saved-label format, or the model.

## Problem

Real Book continuation staves carry no clef/key/time. At extraction time the CRNN
reads the **raw header-less crop**, which is *not* the distribution the model was
trained on — synthetic PrIMuS staves always have a full header. The pre-fill the
labeler shows for these crops is therefore noisier than it needs to be, slowing
hand-correction.

At inference, `pipeline.py` already handles this by injecting a virtual header
(`inject_header`) onto continuation staves with the page's key+time, then re-running
the CRNN on the injected image. The labeler does not currently offer this — it only
has a raw pre-fill and an LMX→image `render` endpoint.

## Idea

Give the labeler a "Re-predict with header" action: the user picks the correct key
and time (which they can read from the music), the tool injects the matching header
template onto the crop, runs the CRNN on the **injected** image (matching the
inference distribution → cleaner tokens), strips the header tokens back out, and
fills the editor with the result. The saved label stays header-less, exactly as the
current convention requires.

This closes the train/inference distribution seam *for the pre-fill only* — it does
not change what gets saved.

## Approach (chosen: A — live endpoint)

### Backend — new `POST /api/music-labeler/predict`

Request body:
```json
{ "filename": "page0020_staff5.png", "key": "key:fifths:0", "render_time": "3/4" }
```

Reuses the exact chain already in `extract_real_music_strips.py`:
1. Load the strip PNG (grayscale) from `_MUSIC_STRIPS_DIR`.
2. Parse `render_time` "B/T" → `("time", "beats:B", "beat-type:T")`.
3. `inject_header(img, key, time_tuple)` (from `omr_pipeline.header_injector`).
4. `recognize_music([injected], checkpoint)` → raw tokens.
5. `fix_sequence(raw, global_key=None, global_time=None, force_clef=True)`.
6. `_strip_header_tokens(...)` to drop clef/key/time.

Response:
```json
{ "tokens": "measure pitch:C octave:5 eighth ...", "header_injected": true }
```

`header_injected` is `false` when no template exists for that (key, time) combo —
`inject_header` silently returns the raw image in that case, so the endpoint compares
the injected width to the original and reports the truth. The UI warns instead of
silently giving a non-injected prediction.

Reuse note: factor the predict-from-image logic so the endpoint and the offline
extractor share one helper rather than duplicating the inject→recognize→fix→strip
sequence. `_strip_header_tokens` and `_likely_has_header` already live in
`extract_real_music_strips.py`; import or relocate to a shared spot rather than
copy.

### Frontend — `static/music_labeler.html`

- Add a small **key dropdown** next to the existing "Render bar layout in:" row.
  Like the bar-layout choice, it **persists across strips** (a whole 3/4 song in one
  key needs no re-selecting).
- Add a **"Re-predict w/ header"** button (keyboard shortcut `Ctrl+R`). On click:
  POST `{filename, key, render_time}` → overwrite the editor with `tokens`.
- If `header_injected` is `false`, show a small inline notice ("no template for
  key X / time Y — prediction is on the raw crop").
- "Restore original" still recovers the first stored prediction; the helper is just
  another way to populate the editor.

## Rejected alternatives

- **B. Offline batch pre-compute** of injected predictions into a second jsonl field.
  No live model call, but the key/time can't be corrected interactively — defeats the
  "fix the time → instant better prediction" loop.

## Out of scope

- No change to training, the split, the extractor's batch flow, or the saved-label
  schema.
- No auto-detection of key/time from the crop (the user supplies it).

## Verification

- API module imports; new endpoint returns 200 with tokens on a real strip.
- For a continuation strip, injected prediction differs from (and is no worse than)
  the raw prediction; header tokens are absent from the returned `tokens`.
- Missing-template combo returns `header_injected: false`.
- UI: dropdown persists across "next"; button fills editor; shortcut works.
