#!/bin/bash
# The analysis (analysis/), from the pipeline tables to figures and tables.
#   bash run_analysis.sh                      unpack results/pipeline, write results/
#   bash run_analysis.sh <pipeline output> [results dir]
# Intermediate tables and logs go to HURR_OUT (default: next to the pipeline output, analysis/).
set -euo pipefail
DATA="${1:+$(realpath "$1")}"
RES="$(realpath -m "${2:-$(dirname "$0")/results}")"
cd "$(dirname "$0")"
if [ -z "$DATA" ]; then
    DATA="$PWD/data/processed"
    python -m portstorm archive unpack --src results/pipeline --out "$DATA"
fi
export HURR_DATA="$DATA" RESULTS="$RES" HURR_OUT="${HURR_OUT:-$(dirname "$DATA")/analysis}"
PKL="$HURR_OUT/_panel_inputs.pkl"
if [ -f "$PKL" ] && [ -n "$(find "$DATA" -name '*.csv' -newer "$PKL" -print -quit)" ]; then
    rm -f "$PKL"
fi
bash analysis/run_all.sh
