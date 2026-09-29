"""Threshold-free storm-window test: how unusual is the window against same-length windows?

Each storm window's mean is ranked against every same-length window in its reference period
(REF_HALFWIDTH_DAYS either side, excluding the exposure period +/- MASK_DAYS), giving an
empirical p-value in each direction.

    python -m portstorm window --panel port_daily_calls.csv --value-col vessels_present \
        --exposure storm_exposure.csv --out window_vessels_present.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF_HALFWIDTH_DAYS = 45
MASK_DAYS = 10
MIN_REF_WINDOWS = 20
MIN_SERIES_MEAN = 0.0


def window_means(values: pd.Series, k: int) -> pd.Series:
    return values.rolling(k).mean().shift(-(k - 1)).dropna()


def score_pair(series: pd.Series, win_start: pd.Timestamp,
               win_end: pd.Timestamp) -> dict | None:
    k = int((win_end - win_start).days) + 1
    if k < 1:
        return None

    obs_days = series.reindex(pd.date_range(win_start, win_end, freq="D"))
    if obs_days.isna().any():
        return None
    obs = float(obs_days.mean())

    half = pd.Timedelta(days=REF_HALFWIDTH_DAYS)
    mask = pd.Timedelta(days=MASK_DAYS)
    near = series[(series.index >= win_start - half) & (series.index <= win_end + half)]
    if len(near) < k:
        return None

    starts = window_means(near, k).index
    ends = starts + pd.Timedelta(days=k - 1)
    ok = (ends < win_start - mask) | (starts > win_end + mask)
    ref = window_means(near, k)[ok]
    if len(ref) < MIN_REF_WINDOWS:
        return None

    ref_med = float(np.median(ref))
    n = len(ref)
    p_low = (int((ref <= obs).sum()) + 1) / (n + 1)
    p_high = (int((ref >= obs).sum()) + 1) / (n + 1)

    return {
        "k_days": k,
        "obs_mean": round(obs, 4),
        "ref_median": round(ref_med, 4),
        "ref_p05": round(float(np.quantile(ref, 0.05)), 4),
        "ref_p95": round(float(np.quantile(ref, 0.95)), 4),
        "n_ref": n,
        "ratio": round(obs / ref_med, 4) if ref_med > 0 else np.nan,
        "z": round((obs - ref_med) / float(ref.std(ddof=1)), 4) if ref.std(ddof=1) > 0 else np.nan,
        "p_lower": round(p_low, 4),
        "p_upper": round(p_high, 4),
        "sig_drop": int(p_low <= 0.05),
        "sig_rise": int(p_high <= 0.05),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Threshold-free storm-window test")
    p.add_argument("--panel", type=Path, required=True, help="long daily panel CSV")
    p.add_argument("--value-col", required=True)
    p.add_argument("--port-col", default="port")
    p.add_argument("--date-col", default="date")
    p.add_argument("--exposure", type=Path, required=True, help="storm_exposure.csv")
    p.add_argument("--out", type=Path, required=True, help="output CSV")
    p.add_argument("--min-series-mean", type=float, default=MIN_SERIES_MEAN,
                   help="drop ports whose daily mean is below this (CyPort excludes "
                        "ports under 5 vessels/day as too quiet to score)")
    args = p.parse_args()

    panel = pd.read_csv(args.panel, parse_dates=[args.date_col])
    expo = pd.read_csv(args.exposure, parse_dates=["exposure_start", "exposure_end"])

    covered = pd.DatetimeIndex(sorted(panel[args.date_col].unique()))
    gaps = pd.date_range(covered.min(), covered.max(), freq="D").difference(covered)
    if len(gaps):
        print(f"Panel gaps : {len(gaps)} unprocessed days will be skipped, "
              f"first {gaps[0].date()}", file=sys.stderr)

    panel_ports = set(panel[args.port_col].unique())
    expo_ports = set(expo["port"].unique())
    if not panel_ports & expo_ports:
        raise SystemExit("no port names shared between panel and exposure table")

    rows = []
    dropped_quiet = 0
    for port, grp in panel.groupby(args.port_col):
        s = (grp.set_index(args.date_col)[args.value_col]
             .groupby(level=0).sum().reindex(covered).fillna(0.0))
        if s.mean() < args.min_series_mean:
            dropped_quiet += 1
            continue
        for _, ev in expo[expo["port"] == port].iterrows():
            ws = ev["exposure_start"].normalize()
            we = ev["exposure_end"].normalize()
            if len(pd.date_range(ws, we, freq="D").difference(covered)):
                continue
            m = score_pair(s, ws, we)
            if m is None:
                continue
            rows.append({"port": port, "SID": ev["SID"], "name": ev["name"],
                         "season": ev["season"], "min_dist_km": ev["min_dist_km"],
                         "sshs_max": ev["sshs_max_in_window"],
                         "sshs_storm_lifetime": ev.get("sshs_storm_lifetime", np.nan),
                         "wind_max_kt": ev["wind_max_in_window_kt"], **m})

    out = pd.DataFrame(rows, columns=None if rows else ["port", "SID", "name"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_ports = out["port"].nunique() if len(out) else 0
    print(f"Scored     : {len(out)} storm-port windows over {n_ports} ports")
    if dropped_quiet:
        print(f"Excluded   : {dropped_quiet} ports below {args.min_series_mean} "
              f"{args.value_col}/day")
    if len(out):
        print(f"Sig. drops : {int(out['sig_drop'].sum())} ({out['sig_drop'].mean():.1%})")
        print(f"Sig. rises : {int(out['sig_rise'].sum())} ({out['sig_rise'].mean():.1%})")
        print(f"Median ratio obs/ref: {out['ratio'].median():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
