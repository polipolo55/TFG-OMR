# Notebooks

Located in `notebooks/`. Run with `poetry run jupyter lab`.

## 01_simple_baseline.ipynb

Early baseline experiments. Establishes initial SER before full training pipeline. Used to validate data loading and label encoding.

## 02_evaluate_model.ipynb

Post-training evaluation on the test set (Phase 5 model, epoch 46, val_SER=1.29%).
- Aggregate SER + melodic SER on clean and scanned test splits
- SER vs. beam width comparison (greedy vs. beam=5/10)
- Per-sample error distribution and worst-performing samples
- Error breakdown by token category (pitch, duration, accidental, structural)
- LilyPond rendering of model predictions for visual inspection

## 03_pipeline_walkthrough.ipynb

End-to-end demonstration.
1. Load a Real Book page image
2. Run full pipeline (staff detect → CRNN → grammar fix → chord OCR)
3. Display annotated page with detected staves
4. Show predicted LMX token sequences
5. Render prediction back to LilyPond for visual verification

## Running Notebooks

```bash
poetry run jupyter lab
```

Or to run a specific notebook headlessly:
```bash
poetry run jupyter nbconvert --to notebook --execute notebooks/02_evaluate_model.ipynb
```
