#!/bin/bash
# Full pipeline, from daily AIS to every table in results/pipeline.
#   bash run_pipeline.sh <input dir> [output dir]
# The input dir holds ais/<year>/, usace_principal_ports.geojson, ibtracs.NA.csv,
# fleet_registry.csv and scrubber_install_dates.csv (README, "Reproduce from raw data").
# Output defaults to <input dir>/processed; per-day intermediates go to <input dir>/work.
# WORKERS (default 4) sets the process count, TIER_SAMPLE (default 40) the tier-map days.
set -euo pipefail
IN="$(realpath "${1:?usage: bash run_pipeline.sh <input dir> [output dir]}")"
OUT="$(realpath -m "${2:-$IN/processed}")"
cd "$(dirname "$0")"
W="${WORKERS:-4}"
WORK="$IN/work"
PORTS="$IN/usace_principal_ports.geojson"
DATES="$IN/scrubber_install_dates.csv"
EXP="$OUT/storm_exposure.csv"
mkdir -p "$OUT"

ps() { echo "=== portstorm $1  $(date +%H:%M:%S)" >&2; python -m portstorm "$@"; }

resilience() {
    local panel="$1" col="$2" floor="$3" w th sfx
    for setting in "0.95 lower" "0.80 cyport"; do
        read -r w th <<< "$setting"
        sfx=""; [ "$th" = cyport ] && sfx="_cyport"
        ps resilience --panel "$panel" --unit-cols port --value-col "$col" --exposure "$EXP" \
            --min-series-mean "$floor" --interval-width "$w" --threshold "$th" --workers "$W" \
            --out "$OUT/resilience_$col$sfx" || echo "no metrics for $col at $w/$th"
    done
}

elasticity() {
    local out="$1" co2="$2" so2="$3" col floor m
    shift 3
    mkdir -p "$out"
    ps elasticity --calls "$OUT/port_daily_calls.csv" --emissions "$co2" --exposure "$EXP" \
        --gaps "$OUT/port_daily_gaps.csv" "$@" --out "$out" --placebo | tee "$out/summary.txt"
    for m in vessels_present co2_ton; do
        ps window --panel "$out/matched_panel.csv" --value-col "$m" --exposure "$EXP" \
            --out "$out/window_$m.csv"
    done
    for spec in "so2_air_ton 0.005" "so2_washwater_ton 0.01" "so2_total_ton 0.01" "fc_total_ton 3"; do
        read -r col floor <<< "$spec"
        ps elasticity --calls "$OUT/port_daily_calls.csv" --emissions "$so2" --emission-col "$col" \
            --min-co2 "$floor" --exposure "$EXP" --gaps "$OUT/port_daily_gaps.csv" "$@" \
            --out "$out/$col" || echo "no elasticity for $col in $out"
    done
}

ps calls extract --src "$IN/ais" --out "$WORK/presence" --ports "$PORTS" --workers "$W"
ps calls visits --src "$WORK/presence" --out "$OUT"
ps storms --ibtracs "$IN/ibtracs.NA.csv" --ports "$PORTS" --season-min 2015 --season-max 2023 --out "$OUT"
resilience "$OUT/port_daily_calls.csv" port_calls 5
resilience "$OUT/port_daily_calls.csv" vessels_present 5
ps cyport validate --exposure "$EXP" --metrics "$OUT/resilience_vessels_present_cyport/resilience_metrics.csv"

ps fuel --src "$IN/ais" --out "$WORK/fuel" --fleet "$IN/fleet_registry.csv" --workers "$W"
ps zones --src "$WORK/fuel" --out "$OUT" --ports "$PORTS" --scrubber-dates "$DATES" \
    --cache-dir "$WORK/cache/ports" --workers "$W"
resilience "$OUT/port_daily_emissions.csv" co2_ton 10
resilience "$OUT/port_daily_emissions.csv" fc_total_ton "$(awk 'BEGIN { printf "%.4f", 10 / 3.114 }')"

ps gaps --src "$WORK/fuel" --ports "$PORTS" --out "$OUT/port_daily_gaps.csv" --workers "$W"
mkdir -p "$OUT/elasticity_all_vessels"
ps window --panel "$OUT/port_daily_gaps.csv" --value-col discard_share --exposure "$EXP" \
    --out "$OUT/elasticity_all_vessels/window_discard_share.csv"

ps sulfur --panel "$OUT/port_daily_emissions_by_class.csv" --group-cols date,port --scenarios \
    --out "$OUT/port_daily_sulfur.csv"
ps sulfur --panel "$OUT/port_daily_emissions_by_class.csv" --group-cols date,port,vclass \
    --out "$OUT/port_daily_sulfur_by_class.csv"
elasticity "$OUT/elasticity" "$OUT/port_daily_emissions_by_class.csv" \
    "$OUT/port_daily_sulfur_by_class.csv" --classes cargo,tanker
elasticity "$OUT/elasticity_all_vessels" "$OUT/port_daily_emissions.csv" "$OUT/port_daily_sulfur.csv"

ps tiers --src "$IN/ais" --ports "$PORTS" --out "$WORK/tier_map.parquet" --sample "${TIER_SAMPLE:-40}"
ps zones --src "$WORK/fuel" --out "$OUT/tiers" --tier-map "$WORK/tier_map.parquet" \
    --scrubber-dates "$DATES" --cache-dir "$WORK/cache/tiers" --workers "$W"
ps sulfur --panel "$OUT/tiers/emissions_by_zone.csv" --group-cols date,complex,tier --scenarios \
    --out "$OUT/tiers/tier_daily_sulfur.csv"

[ -s "$OUT/storm_covariates.csv" ] || ps covariates --exposure "$EXP" --ports "$PORTS" --out "$OUT" \
    --cache-dir "$WORK/cache/noaa" --asos-cache "$WORK/cache/asos_stations.csv"
echo "pipeline done: $OUT"
