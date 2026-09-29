"""CyPort resilience metrics from a daily series and port-cyclone exposure records.

For each interaction the exposure period +/- MASK_DAYS is masked, a baseline is fitted to the
rest of the series (seasonal least squares "ols", "prophet", or the "local" neighbourhood
quantiles) and predicted over the mask with an interval. Days below the interval's lower bound
form runs; the run overlapping the exposure period is the impact window [t_o, t_e], summarised
as total impact, its unnormalised value, t_s, t_c = t_e + 1 and the recovery duration.

--threshold lower is the paper's Eq. (1). --threshold cyport with --interval-width 0.80
reproduces CyPort's released metrics: flag against the 80% bound, measure against the
prediction, count a flat day as recovering.

    python -m portstorm resilience --panel port_daily_calls.csv --unit-cols port \
        --value-col vessels_present --port-col port --exposure storm_exposure.csv --out out/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from .parallel import imap

MASK_DAYS = 10
INTERVAL_WIDTH = 0.95
MIN_TRAIN_DAYS = 45
MIN_MEAN_VALUE = 1e-9

MIN_SERIES_MEAN = 0.0
UNCERTAINTY_SAMPLES = 1000
RANDOM_SEED = 20260913
LOCAL_HALFWIDTH_DAYS = 45
MIN_LOCAL_DAYS = 30

N_CHANGEPOINTS = 12
CHANGEPOINT_RANGE = 0.8
WEEKLY_ORDER = 3
YEARLY_ORDER = 10

warnings.filterwarnings("ignore")


def _quiet_prophet() -> None:
    for name in ("prophet", "cmdstanpy", "prophet.models"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.handlers.clear()
        lg.addHandler(logging.NullHandler())
        lg.propagate = False
        lg.disabled = True
    logging.getLogger().setLevel(logging.WARNING)


def _seasonal_orders(span_days: float) -> tuple[int, int]:
    weekly = WEEKLY_ORDER if span_days >= 28 else 0
    yearly = YEARLY_ORDER if span_days >= 730 else (3 if span_days >= 365 else 0)
    return weekly, yearly


def _design(days: np.ndarray, knots: np.ndarray, weekly: int, yearly: int) -> np.ndarray:
    cols = [np.ones_like(days), days]
    cols += [np.clip(days - k, 0.0, None) for k in knots]
    for order, period in ((weekly, 7.0), (yearly, 365.25)):
        for k in range(1, order + 1):
            cols.append(np.sin(2 * np.pi * k * days / period))
            cols.append(np.cos(2 * np.pi * k * days / period))
    return np.column_stack(cols)


def fit_baseline_ols(series: pd.Series, mask_start, mask_end) -> pd.DataFrame | None:
    train = series[(series.index < mask_start) | (series.index > mask_end)]
    if len(train) < MIN_TRAIN_DAYS or train.mean() < MIN_MEAN_VALUE:
        return None

    origin = series.index[0]
    t_train = (train.index - origin).days.to_numpy().astype(float)
    span = pd.date_range(mask_start, mask_end, freq="D")
    t_pred = (span - origin).days.to_numpy().astype(float)

    span_days = float(t_train.max() - t_train.min())
    weekly, yearly = _seasonal_orders(span_days)
    n_knots = int(np.clip(span_days / 90.0, 0, N_CHANGEPOINTS))
    if n_knots:
        hi = np.quantile(t_train, CHANGEPOINT_RANGE)
        knots = np.linspace(t_train.min(), hi, n_knots + 2)[1:-1]
    else:
        knots = np.empty(0)

    x = _design(t_train, knots, weekly, yearly)
    y = train.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=1e-8)

    resid = y - x @ beta
    lo_q, hi_q = np.quantile(resid, [(1 - INTERVAL_WIDTH) / 2, (1 + INTERVAL_WIDTH) / 2])

    yhat = _design(t_pred, knots, weekly, yearly) @ beta
    ceiling = float(train.max()) * 2.0 + float(train.std()) + 1.0
    out = pd.DataFrame({"yhat": yhat, "yhat_lower": yhat + lo_q, "yhat_upper": yhat + hi_q},
                       index=span)
    return out.clip(lower=0.0, upper=ceiling)


def fit_baseline_prophet(series: pd.Series, mask_start, mask_end) -> pd.DataFrame | None:
    from prophet import Prophet
    _quiet_prophet()

    train = series[(series.index < mask_start) | (series.index > mask_end)]
    if len(train) < MIN_TRAIN_DAYS or train.mean() < MIN_MEAN_VALUE:
        return None

    np.random.seed(RANDOM_SEED)
    m = Prophet(interval_width=INTERVAL_WIDTH, weekly_seasonality=True,
                yearly_seasonality=len(train) > 365, daily_seasonality=False,
                uncertainty_samples=UNCERTAINTY_SAMPLES)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}),
          seed=RANDOM_SEED, algorithm="LBFGS")

    span = pd.date_range(mask_start, mask_end, freq="D")
    fc = m.predict(pd.DataFrame({"ds": span}))
    fc = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    return fc.clip(lower=0.0)


def fit_baseline_local(series: pd.Series, mask_start, mask_end) -> pd.DataFrame | None:
    half = pd.Timedelta(days=LOCAL_HALFWIDTH_DAYS)
    near = series[(series.index >= mask_start - half) & (series.index <= mask_end + half)]
    train = near[(near.index < mask_start) | (near.index > mask_end)]
    if len(train) < MIN_LOCAL_DAYS or train.mean() < MIN_MEAN_VALUE:
        return None

    lo, hi = np.quantile(train.to_numpy(dtype=float),
                         [(1 - INTERVAL_WIDTH) / 2, (1 + INTERVAL_WIDTH) / 2])
    span = pd.date_range(mask_start, mask_end, freq="D")
    centre = float(np.median(train))
    return pd.DataFrame({"yhat": centre, "yhat_lower": lo, "yhat_upper": hi},
                        index=span).clip(lower=0.0)


BACKENDS = {"ols": fit_baseline_ols, "prophet": fit_baseline_prophet,
            "local": fit_baseline_local}
_BACKEND = "ols"


def fit_baseline(series: pd.Series, mask_start, mask_end) -> pd.DataFrame | None:
    return BACKENDS[_BACKEND](series, mask_start, mask_end)


def _runs(flags: pd.Series) -> list[tuple]:
    out, start, prev = [], None, None
    for d, flag in flags.items():
        if flag and start is None:
            start = d
        elif not flag and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, flags.index[-1]))
    return out


THRESHOLDS = {
    "lower":  {"flag": "yhat_lower", "measure": "yhat_lower", "upper": "yhat_upper",
               "rise": "strict"},
    "cyport": {"flag": "yhat_lower", "measure": "yhat", "upper": "yhat_upper",
               "rise": "nondecreasing"},
    "center": {"flag": "yhat", "measure": "yhat", "upper": "yhat",
               "rise": "strict"},
}
THRESHOLD = "lower"


def excess_metrics(obs: pd.Series, base: pd.DataFrame, exp_start, exp_end) -> dict:
    upper, days = base[THRESHOLDS[THRESHOLD]["upper"]], base.index
    above = pd.Series([obs.get(d, np.nan) > upper.loc[d] for d in days],
                      index=days).fillna(False)
    hits = [r for r in _runs(above) if r[0] <= exp_end and r[1] >= exp_start]
    if not hits:
        return {"excess_days": 0, "total_excess": 0.0, "total_excess_value": 0.0}
    t_a, t_b = max(hits, key=lambda r: (r[1] - r[0]).days)
    win = pd.date_range(t_a, t_b, freq="D")
    o = pd.Series([obs.get(d, 0.0) for d in win], index=win)
    up = upper.reindex(win)
    ratio = (o / up.replace(0, np.nan)) - 1.0
    return {"excess_days": len(win),
            "total_excess": round(float(ratio.clip(lower=0).sum()), 4),
            "total_excess_value": round(float((o - up).clip(lower=0).sum()), 3)}


def curve_metrics(obs: pd.Series, base: pd.DataFrame,
                  exp_start, exp_end) -> dict | None:
    setting = THRESHOLDS[THRESHOLD]
    lower = base[setting["flag"]]
    days = base.index
    below = pd.Series([obs.get(d, np.nan) < lower.loc[d] for d in days], index=days)
    below = below.fillna(False)
    if not below.any():
        return None

    overlapping = [r for r in _runs(below) if r[0] <= exp_end and r[1] >= exp_start]
    if not overlapping:
        return None
    t_o, t_e = max(overlapping, key=lambda r: (r[1] - r[0]).days)

    window = pd.date_range(t_o, t_e, freq="D")
    o = pd.Series([obs.get(d, 0.0) for d in window], index=window)
    lo = base[setting["measure"]].reindex(window)

    t_c = t_e + pd.Timedelta(days=1)
    nxt = pd.Series([obs.get(d + pd.Timedelta(days=1), np.nan) for d in window], index=window)
    forward = nxt - o

    def trace_back(rising) -> pd.Timestamp:
        t = t_e
        for d in reversed(window):
            if rising(forward.loc[d]):
                t = d
            else:
                break
        return t

    t_strict = trace_back(lambda x: x > 0)
    t_s = t_strict if setting["rise"] == "strict" else trace_back(lambda x: x >= 0)
    recovery_observed = bool(t_strict < t_e)

    ratio = (o / lo.replace(0, np.nan)).clip(upper=1.0)
    total_impact = float((1.0 - ratio).sum())
    total_impact_value = float((lo - o).clip(lower=0).sum())
    max_impact = float((1.0 - ratio).max())

    return {
        "t_initial_disruption": t_o.date(), "t_start_recovery": t_s.date(),
        "t_end_disruption": t_c.date(),
        "disruption_days": len(window),
        "recovery_days": int((t_c - t_s).days),
        "recovery_observed": int(recovery_observed),
        "days_at_zero": int((o <= 0).sum()),
        "total_impact": round(total_impact, 4),
        "total_impact_value": round(total_impact_value, 3),
        "max_impact": round(max_impact, 4),
        "observed_in_window": round(float(o.sum()), 3),
        "baseline_in_window": round(float(lo.sum()), 3),
    }


def score_pair(task: tuple) -> tuple[dict | None, object | None]:
    key, dates, values, e, units, backend, threshold, iwidth = task
    global _BACKEND, THRESHOLD, INTERVAL_WIDTH
    _BACKEND = backend
    THRESHOLD = threshold
    INTERVAL_WIDTH = iwidth
    try:
        s = pd.Series(values, index=pd.DatetimeIndex(dates))
        base = fit_baseline(s, pd.Timestamp(e["mask_start"]), pd.Timestamp(e["mask_end"]))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", **key,
                "SID": e["SID"], "name": e["name"]}, None
    if base is None:
        return None, None

    rec = {**key, "SID": e["SID"], "name": e["name"], "season": e["season"],
           "min_dist_km": e["min_dist_km"], "sshs_max": e["sshs_max"],
           "sshs_storm_lifetime": e.get("sshs_storm_lifetime", np.nan),
           "wind_max_kt": e["wind_max_kt"],
           "landfall_in_window": e.get("landfall_in_window", np.nan)}
    es, ee = pd.Timestamp(e["exp_start"]), pd.Timestamp(e["exp_end"])
    rec.update(excess_metrics(s, base, es, ee))
    m = curve_metrics(s, base, es, ee)
    if m is None:
        rec.update({"t_initial_disruption": "", "t_start_recovery": "",
                    "t_end_disruption": "", "disruption_days": 0, "recovery_days": 0,
                    "recovery_observed": 0, "days_at_zero": 0,
                    "total_impact": 0.0, "total_impact_value": 0.0, "max_impact": 0.0,
                    "observed_in_window": "", "baseline_in_window": ""})
    else:
        rec.update(m)

    curve = base.copy()
    curve["observed"] = [s.get(d, np.nan) for d in curve.index]
    for col in units:
        curve[col] = key[col]
    curve["SID"], curve["name"] = e["SID"], e["name"]
    curve = curve.reset_index()
    curve = curve.rename(columns={curve.columns[0]: "date"})
    return rec, curve


def main() -> int:
    p = argparse.ArgumentParser(description="CyPort resilience metrics")
    p.add_argument("--panel", type=Path, required=True, help="long daily panel CSV")
    p.add_argument("--unit-cols", required=True, help="columns identifying a series")
    p.add_argument("--value-col", required=True)
    p.add_argument("--date-col", default="date")
    p.add_argument("--exposure", type=Path, required=True, help="storm_exposure.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port-col", default="port", help="unit column matched to the exposure")
    p.add_argument("--min-series-mean", type=float, default=MIN_SERIES_MEAN)
    p.add_argument("--interval-width", type=float, default=INTERVAL_WIDTH)
    p.add_argument("--threshold", choices=sorted(THRESHOLDS), default="lower")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="ols")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    units = [c.strip() for c in args.unit_cols.split(",")]
    panel = pd.read_csv(args.panel, parse_dates=[args.date_col])
    missing = [c for c in units + [args.value_col] if c not in panel.columns]
    if missing:
        p.error(f"panel is missing columns {missing}")
    if args.port_col not in units:
        p.error(f"--port-col {args.port_col!r} must be one of --unit-cols")
    exp = pd.read_csv(args.exposure, parse_dates=["exposure_start", "exposure_end"])
    series_all = panel.groupby(units + [args.date_col])[args.value_col].sum().reset_index()
    covered = pd.DatetimeIndex(sorted(series_all[args.date_col].unique()))
    if not set(series_all[args.port_col].astype(str)) & set(exp["port"].astype(str)):
        print("no port name appears in both the panel and the exposure table")
        return 1

    tasks = []
    for key, grp in series_all.groupby(units):
        key = dict(zip(units, key if isinstance(key, tuple) else (key,)))
        s_i = grp.set_index(args.date_col)[args.value_col].reindex(covered).fillna(0.0)
        if s_i.mean() < args.min_series_mean:
            continue
        for _, e in exp[exp["port"] == key[args.port_col]].iterrows():
            ms = e["exposure_start"].normalize() - pd.Timedelta(days=MASK_DAYS)
            me = e["exposure_end"].normalize() + pd.Timedelta(days=MASK_DAYS)
            if ms < s_i.index.min() or me > s_i.index.max():
                continue
            if len(pd.date_range(ms, me, freq="D").difference(covered)):
                continue
            tasks.append((
                key, s_i.index.to_numpy(), s_i.to_numpy(),
                {"SID": e["SID"], "name": e["name"], "season": e["season"],
                 "min_dist_km": e["min_dist_km"], "sshs_max": e["sshs_max_in_window"],
                 "sshs_storm_lifetime": e.get("sshs_storm_lifetime", np.nan),
                 "landfall_in_window": e.get("landfall_in_window", np.nan),
                 "wind_max_kt": e["wind_max_in_window_kt"],
                 "mask_start": ms, "mask_end": me,
                 "exp_start": e["exposure_start"].normalize(),
                 "exp_end": e["exposure_end"].normalize()},
                units, args.backend, args.threshold, args.interval_width,
            ))

    out_rows, curve_rows, failures = [], [], []
    for rec, curve in imap(score_pair, tasks, args.workers):
        if rec is not None and "_error" in rec:
            failures.append(f"{rec.get('name')} / {rec.get(units[0])}: {rec['_error']}")
        elif rec is not None:
            out_rows.append(rec)
            curve_rows.append(curve)
    for f in failures[:10]:
        print(f"  failed: {f}")
    if not out_rows:
        print("no unit-storm pair produced metrics")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(out_rows).sort_values(units + ["season", "SID"])
    res.to_csv(args.out / "resilience_metrics.csv", index=False)
    curves = pd.concat(curve_rows, ignore_index=True).sort_values(units + ["SID", "date"])
    curves.to_csv(args.out / "resilience_curves.csv", index=False)
    print(f"{len(res):,} interactions, {int((res['total_impact'] > 0).sum()):,} disrupted "
          f"-> {args.out / 'resilience_metrics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
