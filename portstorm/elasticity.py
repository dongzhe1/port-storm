"""Elasticity of port emissions to port traffic during cyclone exposure.

Each port-storm window is compared with a local reference period (REF_HALFWIDTH_DAYS either
side, excluding the exposure period +/- MASK_DAYS). With N vessels present per day, H
vessel-hours per vessel and I emissions per vessel-hour, emissions per day = N x H x I
exactly, so the slope b of log(r_CO2) on log(r_N) splits as b = 1 + b_H + b_I. Intervals are a
port-cluster bootstrap; --placebo repeats the estimate on time-shifted windows. --classes
restricts the emission side to the vessel classes that vessels_present counts.

    python -m portstorm elasticity --calls port_daily_calls.csv \
        --emissions port_daily_emissions_by_class.csv --classes cargo,tanker \
        --exposure storm_exposure.csv --out elasticity/ --placebo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF_HALFWIDTH_DAYS = 45
MASK_DAYS = 10
MIN_REF_DAYS = 20
MIN_VESSELS_PER_DAY = 5.0
MIN_CO2_PER_DAY = 10.0
N_BOOTSTRAP = 2000
RANDOM_SEED = 20260913
PLACEBO_SHIFTS = [("-2y", -730), ("-1y", -365), ("-60d", -60), ("-30d", -30),
                  ("+30d", 30), ("+60d", 60), ("+1y", 365), ("+2y", 730)]

POPULATION_NOTE = ("vessels_present counts cargo and tanker calls only, so b_CO2 here is "
                   "total port emissions against merchant traffic, and b_H / b_I mix two "
                   "populations -- they are not per-vessel quantities. Use --classes "
                   "cargo,tanker for the decomposition.")


def build_panel(calls: pd.DataFrame, emis: pd.DataFrame, value_col: str = "co2_ton",
                date_col: str = "date") -> tuple[pd.DataFrame, list[str]]:
    dc, ec = set(calls[date_col]), set(emis[date_col])
    covered = pd.DatetimeIndex(sorted(dc & ec))
    dropped = (dc | ec) - (dc & ec)
    if dropped:
        print(f"Calendar   : {len(covered):,} days processed by both panels; "
              f"{len(dropped)} day(s) only one of them has, excluded "
              f"(first {min(dropped).date()})", file=sys.stderr)
    ports = sorted(set(calls["port"]) & set(emis["port"]))

    frames = []
    for p in ports:
        n = calls.loc[calls["port"] == p].set_index(date_col)["vessels_present"]
        e = emis.loc[emis["port"] == p].set_index(date_col)[["hours", value_col]]
        n = n.groupby(level=0).sum().reindex(covered).fillna(0.0)
        e = e.groupby(level=0).sum().reindex(covered).fillna(0.0)
        frames.append(pd.DataFrame({date_col: covered, "port": p,
                                    "vessels_present": n.to_numpy(),
                                    "hours": e["hours"].to_numpy(),
                                    "co2_ton": e[value_col].to_numpy()}))
    return pd.concat(frames, ignore_index=True), ports


def _factors(block: pd.DataFrame) -> tuple[float, float, float, float]:
    n_days = len(block)
    n = float(block["vessels_present"].sum())
    h = float(block["hours"].sum())
    c = float(block["co2_ton"].sum())
    return (n / n_days,
            h / n if n > 0 else np.nan,
            c / h if h > 0 else np.nan,
            c / n_days)


def score_windows(panel: pd.DataFrame, expo: pd.DataFrame,
                  gaps: pd.DataFrame | None = None,
                  date_col: str = "date") -> pd.DataFrame:
    covered = pd.DatetimeIndex(sorted(panel[date_col].unique()))
    half = pd.Timedelta(days=REF_HALFWIDTH_DAYS)
    mask = pd.Timedelta(days=MASK_DAYS)
    series = {p: g.set_index(date_col)[["vessels_present", "hours", "co2_ton"]]
              for p, g in panel.groupby("port")}
    gap_series = {}
    if gaps is not None:
        for p, g in gaps.groupby("port"):
            gap_series[p] = (g.set_index(date_col)[["hours_credited", "hours_discarded"]]
                             .groupby(level=0).sum())

    rows = []
    for _, ev in expo.iterrows():
        s = series.get(ev["port"])
        if s is None:
            continue
        ws = ev["exposure_start"].normalize()
        we = ev["exposure_end"].normalize()
        days = pd.date_range(ws, we, freq="D")
        if len(days.difference(covered)):
            continue
        near = s[(s.index >= ws - half) & (s.index <= we + half)]
        ref = near[(near.index < ws - mask) | (near.index > we + mask)]
        if len(ref) < MIN_REF_DAYS:
            continue

        o = _factors(s.loc[days])
        r = _factors(ref)

        d_win = d_ref = np.nan
        gs = gap_series.get(ev["port"])
        if gs is not None:
            gw = gs.reindex(days).dropna()
            gr = gs.reindex(ref.index).dropna()
            if len(gw) and len(gr):
                tw = gw["hours_credited"].sum() + gw["hours_discarded"].sum()
                tr = gr["hours_credited"].sum() + gr["hours_discarded"].sum()
                if tw > 0 and tr > 0:
                    d_win = float(gw["hours_discarded"].sum() / tw)
                    d_ref = float(gr["hours_discarded"].sum() / tr)

        rows.append({
            "port": ev["port"], "SID": ev["SID"], "name": ev["name"],
            "season": ev["season"], "min_dist_km": ev["min_dist_km"],
            "sshs_max": ev["sshs_max_in_window"],
            "sshs_storm_lifetime": ev.get("sshs_storm_lifetime", np.nan),
            "wind_max_kt": ev["wind_max_in_window_kt"],
            "k_days": len(days), "n_ref_days": len(ref),
            "N_obs": o[0], "N_ref": r[0], "co2_obs_t": o[3] * len(days),
            "co2_ref_t": r[3] * len(days),
            "r_N": o[0] / r[0] if r[0] > 0 else np.nan,
            "r_H": o[1] / r[1] if r[1] and r[1] > 0 else np.nan,
            "r_I": o[2] / r[2] if r[2] and r[2] > 0 else np.nan,
            "r_CO2": o[3] / r[3] if r[3] > 0 else np.nan,
            "discard_win": d_win, "discard_ref": d_ref,
        })

    out = pd.DataFrame(rows)
    if len(out):
        adj = ((1.0 - out["discard_ref"]) / (1.0 - out["discard_win"])).astype(float)
        out["gap_adj"] = adj.where(adj.notna(), 1.0)
        out["r_CO2_gapadj"] = out["r_CO2"] * out["gap_adj"]
        out["co2_shortfall_t"] = out["co2_ref_t"] - out["co2_obs_t"]
        out["co2_shortfall_naive_t"] = (1 - out["r_N"]) * out["co2_ref_t"]
    return out


def fit_elasticity(scored: pd.DataFrame, col: str, rng: np.random.Generator,
                   n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float, int]:
    q = scored.dropna(subset=["r_N", col])
    q = q[(q["r_N"] > 0) & (q[col] > 0)]
    if len(q) < 10:
        return np.nan, np.nan, np.nan, len(q)
    x, y = np.log(q["r_N"].to_numpy()), np.log(q[col].to_numpy())
    b = float(np.polyfit(x, y, 1)[0])

    port_of = q["port"].to_numpy()
    ports = np.unique(port_of)
    rows_of = [np.flatnonzero(port_of == p) for p in ports]
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([rows_of[k] for k in rng.integers(0, len(ports), len(ports))])
        if len(ii) > 20 and np.unique(x[ii]).size > 5:
            draws.append(np.polyfit(x[ii], y[ii], 1)[0])
    if not draws:
        return b, np.nan, np.nan, len(q)
    return b, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(q)


def summarise(scored: pd.DataFrame, rng: np.random.Generator) -> None:
    n_ok = scored["r_CO2"].notna().sum()
    print(f"Scored     : {len(scored)} windows, {scored['port'].nunique()} ports, "
          f"{scored['SID'].nunique()} cyclones ({n_ok} usable for the log fit)")

    print("\nElasticity to vessels present, log-log OLS, 95% CI by port bootstrap")
    total = None
    for col, label in [("r_CO2", "CO2 per day"), ("r_H", "  hours per vessel"),
                       ("r_I", "  CO2 per vessel-hour")]:
        b, lo, hi, n = fit_elasticity(scored, col, rng)
        if col == "r_CO2":
            total = b
        print(f"  {label:24s} b = {b:6.3f}   [{lo:6.3f}, {hi:6.3f}]   n = {n}")
    if total is not None:
        bh = fit_elasticity(scored, "r_H", rng)[0]
        bi = fit_elasticity(scored, "r_I", rng)[0]
        print(f"  identity check: 1 + b_H + b_I = {1 + bh + bi:.3f} vs b_CO2 = {total:.3f}")

    if "r_CO2_gapadj" in scored and scored["discard_win"].notna().any():
        cov = scored["discard_win"].notna()
        b_raw, lo, hi, n = fit_elasticity(scored, "r_CO2", rng)
        b_adj, lo2, hi2, n2 = fit_elasticity(scored, "r_CO2_gapadj", rng)
        excess = (scored["discard_win"] - scored["discard_ref"])
        print(f"\nGap adjustment, on the {int(cov.sum())} of {len(scored)} windows with "
              "gap coverage")
        print(f"  excess discarded time: median {excess.median():+.3%}, "
              f"90th pct {excess.quantile(0.9):+.3%}, max {excess.max():+.3%}")
        print(f"  b_CO2 raw          {b_raw:6.3f}   [{lo:6.3f}, {hi:6.3f}]")
        print(f"  b_CO2 gap-adjusted {b_adj:6.3f}   [{lo2:6.3f}, {hi2:6.3f}]")

    print("\nMedian window ratio by distance to the eye")
    bins = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    for lo, hi in bins:
        s = scored[(scored["min_dist_km"] > lo) & (scored["min_dist_km"] <= hi)]
        if not len(s):
            continue
        print(f"  {lo:3d}-{hi:3d} km  n={len(s):4d}   CO2 {s['r_CO2'].median():.3f}"
              f" = N {s['r_N'].median():.3f}"
              f" x H {s['r_H'].median():.3f}"
              f" x I {s['r_I'].median():.3f}")

    cat_cols = [("sshs_max", "peak while this port was exposed")]
    if (scored.get("sshs_storm_lifetime") is not None
            and scored["sshs_storm_lifetime"].notna().any()):
        cat_cols.append(("sshs_storm_lifetime",
                         "storm's peak anywhere -- CyPort's definition"))
    for col, note in cat_cols:
        print(f"\nMedian window ratio by category [{col}: {note}]")
        for cat in sorted(scored[col].dropna().unique()):
            s = scored[scored[col] == cat]
            if len(s) < 10:
                continue
            print(f"  cat {int(cat):3d}     n={len(s):4d}   CO2 {s['r_CO2'].median():.3f}"
                  f" = N {s['r_N'].median():.3f}"
                  f" x H {s['r_H'].median():.3f}"
                  f" x I {s['r_I'].median():.3f}")

    act = scored["co2_shortfall_t"].sum()
    naive = scored["co2_shortfall_naive_t"].sum()
    down = scored.loc[scored["co2_shortfall_t"] > 0, "co2_shortfall_t"].sum()
    up = -scored.loc[scored["co2_shortfall_t"] < 0, "co2_shortfall_t"].sum()
    print("\nCO2 shortfall over all windows")
    print(f"  measured              {act:12,.0f} t   (suppressed {down:,.0f} t, "
          f"raised {up:,.0f} t elsewhere in the same windows)")
    print(f"  traffic-scaled, b=1   {naive:12,.0f} t")
    if naive:
        print(f"  the traffic route understates it by {100 * (1 - naive / act):.0f}% "
              f"({act / naive:.2f}x)")


def main() -> int:
    p = argparse.ArgumentParser(description="Elasticity of port emissions to port traffic")
    p.add_argument("--calls", type=Path, required=True, help="port_daily_calls.csv")
    p.add_argument("--emissions", type=Path, required=True, help="port_daily_emissions.csv")
    p.add_argument("--exposure", type=Path, required=True, help="storm_exposure.csv")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--min-vessels", type=float, default=MIN_VESSELS_PER_DAY)
    p.add_argument("--min-co2", type=float, default=MIN_CO2_PER_DAY)
    p.add_argument("--emission-col", default="co2_ton",
                   help="emission column to estimate the elasticity of; the internal\n"
                        "column names stay CO2-flavoured because the arithmetic is\n"
                        "identical, only the quantity changes")
    p.add_argument("--gaps", type=Path, default=None,
                   help="port_daily_gaps.csv from `portstorm gaps`; when given, a "
                        "gap-adjusted elasticity is reported alongside the raw one")
    p.add_argument("--placebo", action="store_true",
                   help="also run the estimate on time-shifted storm windows")
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--classes", default=None,
                   help="comma-separated vessel classes to keep on the emission side, e.g.\n"
                        "cargo,tanker. Needs a panel with a vclass column. vessels_present\n"
                        "counts cargo and tanker calls only, so without this the two sides\n"
                        "are different ships -- see POPULATION_NOTE")
    args = p.parse_args()

    calls = pd.read_csv(args.calls, parse_dates=["date"])
    emis = pd.read_csv(args.emissions, parse_dates=["date"])
    expo = pd.read_csv(args.exposure, parse_dates=["exposure_start", "exposure_end"])

    if args.classes:
        keep_cls = [c.strip() for c in args.classes.split(",") if c.strip()]
        if "vclass" not in emis.columns:
            raise SystemExit(f"--classes needs a vclass column; {args.emissions.name} has "
                             f"{list(emis.columns)}")
        unknown = sorted(set(keep_cls) - set(emis["vclass"].unique()))
        if unknown:
            raise SystemExit(f"--classes {unknown} not in the panel, which has "
                             f"{sorted(emis['vclass'].unique())}")
        emis = emis[emis["vclass"].isin(keep_cls)]
        num = [c for c in emis.select_dtypes("number").columns if c != "scrubber_fitted"]
        emis = emis.groupby(["date", "port"], as_index=False)[num].sum()
        print(f"Population : emissions from {', '.join(keep_cls)} only")
    else:
        print("Population : emissions from every registered vessel")
        print(f"             NOTE {POPULATION_NOTE}")

    keep_n = calls.groupby("port")["vessels_present"].mean()
    keep_c = emis.groupby("port")[args.emission_col].mean()
    ports = sorted(set(keep_n[keep_n >= args.min_vessels].index)
                   & set(keep_c[keep_c >= args.min_co2].index))
    print(f"Ports      : {(keep_n >= args.min_vessels).sum()} pass "
          f">={args.min_vessels} vessels/day, "
          f"{(keep_c >= args.min_co2).sum()} pass >={args.min_co2} t "
          f"{args.emission_col}/day, "
          f"{len(ports)} pass both")
    if not ports:
        raise SystemExit("no port passes both activity filters")

    panel, _ = build_panel(calls[calls["port"].isin(ports)],
                           emis[emis["port"].isin(ports)], args.emission_col)
    expo = expo[expo["port"].isin(ports)]

    gaps = None
    if args.gaps is not None:
        gaps = pd.read_csv(args.gaps, parse_dates=["date"])
        gaps = gaps[gaps["port"].isin(ports)]
        print(f"Gap panel  : {len(gaps):,} port-days, "
              f"{gaps['date'].dt.year.min()}-{gaps['date'].dt.year.max()}")

    for col, label in [("vessels_present", args.calls.name),
                       ("co2_ton", args.emissions.name)]:
        nz = (panel[col] > 0).mean()
        if nz < 0.05:
            raise SystemExit(
                f"{label} is effectively empty: only {nz:.1%} of the joined panel has "
                f"{col} > 0 over {panel['date'].nunique():,} days. The usual cause is a "
                "stale or truncated input -- check its date range against the exposure "
                "table before re-running.")
    span = panel["date"].nunique()
    for f, name in [(calls, args.calls.name), (emis, args.emissions.name)]:
        days = f["date"].nunique()
        if days < 0.5 * span:
            print(f"WARNING   : {name} covers {days:,} days against the panel's {span:,}; "
                  "missing days are read as zero activity", file=sys.stderr)

    scored = score_windows(panel, expo, gaps)
    if scored.empty:
        raise SystemExit("no exposure window could be scored")

    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out / "matched_panel.csv", index=False)
    scored.to_csv(args.out / "elasticity_windows.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    summarise(scored, rng)

    if args.placebo:
        print("\nPlacebo: the same estimate on shifted windows")
        print(f"  {'shift':>8s}  {'n':>5s}  {'b_CO2':>7s}  "
              f"{'CO2 shortfall':>15s}  {'<=200km median r_CO2':>21s}")
        real = scored[scored["min_dist_km"] <= 200]["r_CO2"].median()
        b0 = fit_elasticity(scored, "r_CO2", np.random.default_rng(RANDOM_SEED), 0)[0]
        print(f"  {'REAL':>8s}  {len(scored):5d}  {b0:7.3f}  "
              f"{scored['co2_shortfall_t'].sum():13,.0f} t  {real:21.3f}")
        lo_d = panel["date"].min() + pd.Timedelta(days=REF_HALFWIDTH_DAYS + 15)
        hi_d = panel["date"].max() - pd.Timedelta(days=REF_HALFWIDTH_DAYS + 15)
        placebo_b = []
        for label, shift in PLACEBO_SHIFTS:
            fake = expo.copy()
            off = pd.Timedelta(days=shift)
            fake["exposure_start"] += off
            fake["exposure_end"] += off
            fake = fake[(fake["exposure_start"] >= lo_d) & (fake["exposure_end"] <= hi_d)]
            s = score_windows(panel, fake)
            if s.empty:
                continue
            b = fit_elasticity(s, "r_CO2", np.random.default_rng(RANDOM_SEED), 0)[0]
            near = s[s["min_dist_km"] <= 200]["r_CO2"].median()
            print(f"  {label:>8s}  {len(s):5d}  {b:7.3f}  "
                  f"{s['co2_shortfall_t'].sum():13,.0f} t  {near:21.3f}")
            if np.isfinite(b):
                placebo_b.append(b)
        if placebo_b:
            lo_p, hi_p = min(placebo_b), max(placebo_b)
            where = ("ABOVE every placebo" if b0 > hi_p else
                     "BELOW every placebo" if b0 < lo_p else
                     "INSIDE the placebo range -- not distinguishable from ordinary weather")
            print(f"  real b {b0:.3f} is {where} [{lo_p:.3f}, {hi_p:.3f}] "
                  f"over {len(placebo_b)} shifts")

    print(f"\nWrote {args.out / 'elasticity_windows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
