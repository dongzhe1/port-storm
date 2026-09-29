"""Compare with CyPort (Kuai et al. 2025, arXiv:2509.22656) and its released data.

    validate  scope, zero inflation, category-4 step and regional pattern of our metrics
              against the figures CyPort published
    inputs    turn CyPort's released tables into a panel and exposure table, so
              `portstorm resilience` can be run on exactly their inputs
    compare   record-level agreement of a resilience_metrics.csv with CyPort's
              interaction_data.csv

    python -m portstorm cyport inputs --cyport cyport/ --out cyport_inputs/
    python -m portstorm cyport compare --metrics resilience_metrics.csv --cyport cyport/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .storms import haversine_km, load_usace_ports

CYPORT = {"ports": 145, "storms": 90, "interactions": 1927, "seasons": (2015, 2023),
          "tipping_point_sshs": 4}
GULF_STATES = ("TX", "LA", "MS", "AL", "FL")
NEAR_KM = 200.0
MATCH_KM = 10.0


def line(label: str, theirs, ours, note: str = "") -> None:
    ok = ""
    if isinstance(theirs, (int, float)) and isinstance(ours, (int, float)) and theirs:
        ok = f"{(ours - theirs) / theirs * 100:+6.1f}%"
    print(f"  {label:34} {str(theirs):>10} {str(ours):>10}  {ok:>8}  {note}")


def by_category(frame: pd.DataFrame, label: str, cat_col: str, impact_col: str) -> None:
    g = (frame.assign(cat=frame[cat_col].fillna(-1).astype(int))
              .groupby("cat")
              .agg(n=("cat", "size"), disrupted=(impact_col, lambda x: (x > 0).mean()),
                   mean_impact=(impact_col, "mean"), mean_recovery=("recovery_days", "mean")))
    g["disrupted"] = (100 * g["disrupted"]).round(1)
    g[["mean_impact", "mean_recovery"]] = g[["mean_impact", "mean_recovery"]].round(2)
    print(f"\n  {label}")
    print(g[g.index >= 0].to_string())
    if 4 in g.index and 3 in g.index and g.loc[3, "mean_impact"]:
        jump = g.loc[4, "mean_impact"] / g.loc[3, "mean_impact"]
        print(f"    category 3 -> 4 mean impact ratio: {jump:.2f}x "
              f"({'a step' if jump > 1.5 else 'no step'})")


def validate(args) -> int:
    exp = pd.read_csv(args.exposure)
    met = pd.read_csv(args.metrics) if args.metrics and args.metrics.is_file() else pd.DataFrame()
    print(f"  {'':34} {'CyPort':>10} {'ours':>10}  {'diff':>8}")
    s0, s1 = CYPORT["seasons"]
    e = exp[exp["season"].between(s0, s1)]
    line("ports with an interaction", CYPORT["ports"], e["port"].nunique())
    line("cyclones with an interaction", CYPORT["storms"], e["SID"].nunique())
    line("port-cyclone interactions", CYPORT["interactions"], len(e))
    if met.empty:
        return 0
    zero = (met[args.impact_col] <= 0).mean()
    line("share of interactions with no impact", "zero-inflated", f"{100 * zero:.1f}%")
    if "sshs_storm_lifetime" in met.columns and met["sshs_storm_lifetime"].notna().any():
        by_category(met, "by storm lifetime peak category (CyPort's SSHS)",
                    "sshs_storm_lifetime", args.impact_col)
    else:
        print("\n  no sshs_storm_lifetime column: the CyPort-comparable check cannot run")
    if "sshs_max" in met.columns:
        by_category(met, "by peak category while the port was exposed", "sshs_max",
                    args.impact_col)
        near = met[met["min_dist_km"] <= NEAR_KM] if "min_dist_km" in met.columns else met[:0]
        if len(near) > 50:
            by_category(near, f"within {NEAR_KM:.0f} km ({len(near)} interactions)",
                        "sshs_max", args.impact_col)
    if "port" in met.columns:
        state = met["port"].astype(str).str.extract(r",\s*([A-Z]{2})\b")[0]
        region = np.where(state.isin(GULF_STATES), "Gulf/SE", "other")
        r = (met.assign(region=region).groupby("region")
                .agg(n=("region", "size"), disrupted=(args.impact_col, lambda x: (x > 0).mean()),
                     mean_impact=(args.impact_col, "mean")))
        print("\n  by region\n" + r.round(3).to_string())
    return 0


def read_release(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    it = pd.read_csv(folder / "interaction_data.csv")
    names = pd.read_csv(folder / "port&resources.csv").set_index("id")["name"]
    counts = pd.read_csv(folder / "port_vessel_count.csv").set_index("port")
    it["port_mean"] = it["PID"].map(counts.mean(axis=1))
    counts.index = counts.index.map(names)
    counts.columns = pd.to_datetime(counts.columns)
    return it, counts, names


def inputs(args) -> int:
    it, counts, _ = read_release(args.cyport)
    args.out.mkdir(parents=True, exist_ok=True)
    panel = counts.T.stack().rename("vessels_present").reset_index()
    panel.columns = ["date", "port", "vessels_present"]
    panel.to_csv(args.out / "panel.csv", index=False)
    exposure = pd.DataFrame({
        "SID": it["SID"], "name": it["SID"], "season": it["SID"].str[:4].astype(int),
        "port": it["port"], "exposure_start": pd.to_datetime(it["start_date"]),
        "exposure_end": pd.to_datetime(it["end_date"]), "min_dist_km": it["DISTANCE"],
        "sshs_max_in_window": it["SSHS"], "sshs_storm_lifetime": it["SSHS"],
        "wind_max_in_window_kt": it["WIND"], "landfall_in_window": it["IF_LANDFALL"]})
    exposure.to_csv(args.out / "exposure.csv", index=False)
    print(f"{len(panel):,} port-days, {len(exposure):,} interactions -> {args.out}")
    return 0


def match_ports(it: pd.DataFrame, ports_path: Path) -> pd.Series:
    pts = load_usace_ports(ports_path)
    names = list(pts)
    lat = np.array([pts[n][0] for n in names])
    lon = np.array([pts[n][1] for n in names])
    near = {}
    for pid, g in it.drop_duplicates("PID").set_index("PID").iterrows():
        d = haversine_km(g["PORT_LAT"], g["PORT_LON"], lat, lon)
        if d.min() <= MATCH_KM:
            near[pid] = names[int(np.argmin(d))]
    return it["PID"].map(near)


def compare(args) -> int:
    it, _, _ = read_release(args.cyport)
    for c in ("start_recovery_date", "end_recovery_date"):
        it[c] = pd.to_datetime(it[c].where(it[c].astype(str) != "0"), errors="coerce")
    met = pd.read_csv(args.metrics)
    for c in ("t_start_recovery", "t_end_disruption"):
        met[c] = pd.to_datetime(met[c], errors="coerce")
    if args.ports:
        it["port"] = match_ports(it, args.ports)
    j = it.merge(met, on=["port", "SID"], suffixes=("", "_ours"))
    print(f"  {'subset':<6} {'n':>5} {'ours':>6} {'theirs':>7} {'agree':>6} {'jacc':>5} "
          f"{'impact':>7} {'t_s':>6} {'t_c':>6} {'recovery':>13}")
    for label, s in (("all", j), (">=1", j[j["port_mean"] >= 1]), (">=5", j[j["port_mean"] >= 5])):
        ours, theirs = s["total_impact_ours"] > 0, s["total_impact"] > 0
        both = s[ours & theirs]
        either = int((ours | theirs).sum())
        print(f"  {label:<6} {len(s):5d} {ours.mean():6.1%} {theirs.mean():7.1%} "
              f"{(ours == theirs).mean():6.1%} {len(both) / either if either else np.nan:5.2f} "
              f"{(both['total_impact_ours'] / both['total_impact']).median():7.2f} "
              f"{(both['t_start_recovery'] == both['start_recovery_date']).mean():6.1%} "
              f"{(both['t_end_disruption'] == both['end_recovery_date']).mean():6.1%} "
              f"{both['recovery_days'].mean():6.2f}/{both['day_of_recover'].mean():.2f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare with CyPort")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="against CyPort's published figures")
    v.add_argument("--exposure", type=Path, required=True)
    v.add_argument("--metrics", type=Path, default=None)
    v.add_argument("--impact-col", default="total_impact")
    i = sub.add_parser("inputs", help="CyPort's released tables as panel + exposure")
    i.add_argument("--cyport", type=Path, required=True, help="folder of CyPort's CSVs")
    i.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="record-level agreement with CyPort")
    c.add_argument("--metrics", type=Path, required=True)
    c.add_argument("--cyport", type=Path, required=True)
    c.add_argument("--ports", type=Path, default=None,
                   help="our port polygons, to match ports by location when the metrics "
                        "were built from our own panel")
    args = p.parse_args()
    return {"validate": validate, "inputs": inputs, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
