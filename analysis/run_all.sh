#!/usr/bin/env bash
# Run the whole analysis, from the upstream emission tables to the manuscript figures and tables.
# Usage (from this folder):  HURR_DATA=/path/to/upstream/tables bash run_all.sh
#   HURR_DATA  pipeline output folder, with the tier panels in tiers/ (see ../README.md)
#   HURR_OUT   intermediate tables and figures (default: ./out)
#   RESULTS    where step 20 copies figures, tables and source data (default: ../results)
set -euo pipefail
cd "$(dirname "$0")"
: "${HURR_DATA:?set HURR_DATA to the pipeline output folder}"
export HURR_OUT="${HURR_OUT:-$PWD/out}"
export PYTHONWARNINGS=ignore MPLBACKEND=Agg
mkdir -p "$HURR_OUT/logs"
for s in 01_score_events 02_reconcile_normals 03_pooled_summary 04_carbon_efficiency 05_arrival_management 06_robustness \
         07_pollutants 08_divergence_events 09_heterogeneity 10_port_locations 11_claims_by_group 12_divergence_placebo_targeting \
         13_fig_map 14_figs_main 15_figs_robustness_per_call 16_fig_wind_response 17_fig_zone_schematic 18_fig_engine_use \
         19_supp_tables 20_export_results; do
  echo "=== $s  $(date '+%H:%M:%S')"
  python3 "$s.py" > "$HURR_OUT/logs/$s.log" 2>&1 || { echo "failed: see $HURR_OUT/logs/$s.log"; exit 1; }
done
python3 check_reproduction.py
