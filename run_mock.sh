#!/bin/bash
# From scratch on synthetic data: fake inputs, the full pipeline, then the analysis.
#   bash run_mock.sh [dir]        everything goes to dir (default demo/)
# The numbers mean nothing. check_reproduction.py compares against the real run, so its
# "DIFFERENCES FOUND" is expected here.
set -euo pipefail
D="$(realpath -m "${1:-demo}")"
cd "$(dirname "$0")"
python -m portstorm fake --out "$D"
mkdir -p "$D/processed"
cp "$D/storm_covariates.csv" "$D/processed/"
WORKERS="${WORKERS:-2}" TIER_SAMPLE=20 bash run_pipeline.sh "$D"
bash run_analysis.sh "$D/processed" "$D/results" | tee "$D/analysis.log" \
    || grep -q "^DIFFERENCES FOUND" "$D/analysis.log"
echo "mock run done: $D/processed (pipeline tables), $D/results (figures and tables)"
