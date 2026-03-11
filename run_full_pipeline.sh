#!/bin/bash
set -e

# Setup directories
RAW_AA="data/primus/package_aa"
RAW_AB="data/primus/package_ab"
OUT_BASE="data/realbook_primus"
OUT_AUG="data/realbook_primus_augmented"

echo "=== Processing Package AA ==="
poetry run python src/cli.py render --source $RAW_AA --output $OUT_BASE/package_aa --workers $(nproc)
poetry run python src/cli.py convert --source $OUT_BASE/package_aa --workers $(nproc)
poetry run python src/cli.py augment --source $OUT_BASE/package_aa --output $OUT_AUG/package_aa --workers $(nproc)

echo "=== Processing Package AB ==="
poetry run python src/cli.py render --source $RAW_AB --output $OUT_BASE/package_ab --workers $(nproc)
poetry run python src/cli.py convert --source $OUT_BASE/package_ab --workers $(nproc)
poetry run python src/cli.py augment --source $OUT_BASE/package_ab --output $OUT_AUG/package_ab --workers $(nproc)

echo "=== Building Unified Vocabulary ==="
poetry run python src/cli.py vocab \
  --data-dir $OUT_BASE/package_aa \
  --extra-data-dir $OUT_BASE/package_ab \
  --output src/CRNN_CTC/vocabulary.txt

echo "Pipeline complete! Dataset ready for training."
