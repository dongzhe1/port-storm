"""Vessel-time the fuel model's DT_MAX_HOURS rule discards, per port and day.

The raw interval between consecutive messages of a ship is rebuilt from timestamps and
compared with the credited dt_h. discard_share = discarded / (credited + discarded); test it
with `portstorm window` to see whether storms make the rule discard more.

    python -m portstorm gaps --src fuel/ --ports ports.geojson --out port_daily_gaps.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .ghg4 import DT_MAX_HOURS
from .parallel import imap
from .ports import assign_ports
from .zones import _geometry

SPAN_CAP_HOURS = 24.0


def day_gaps(task: tuple) -> tuple:
    path, ports_path = task
    tag = Path(path).stem.rsplit("_", 1)[-1]
    df = pd.read_parquet(path, columns=["imo", "base_date_time", "longitude",
                                        "latitude", "dt_h"])
    if df.empty:
        return tag, pd.DataFrame()

    polys, idx = _geometry(ports_path)
    df["port"] = assign_ports(df["longitude"].to_numpy(), df["latitude"].to_numpy(),
                              polys, idx)
    df = df[df["port"] != ""]
    if df.empty:
        return tag, pd.DataFrame()

    df = df.sort_values(["imo", "base_date_time"])
    t = pd.to_datetime(df["base_date_time"])
    raw = t.groupby(df["imo"].to_numpy()).diff().dt.total_seconds() / 3600.0
    df["raw_h"] = raw.fillna(0.0).to_numpy()

    over = df["raw_h"] > DT_MAX_HOURS
    df["discarded_h"] = np.where(over, df["raw_h"].clip(upper=SPAN_CAP_HOURS), 0.0)
    df["date"] = t.dt.date

    out = (df.groupby(["date", "port"], as_index=False)
             .agg(n_messages=("dt_h", "size"),
                  n_vessels=("imo", "nunique"),
                  hours_credited=("dt_h", "sum"),
                  hours_discarded=("discarded_h", "sum"),
                  n_gaps_over=("raw_h", lambda s: int((s > DT_MAX_HOURS).sum()))))
    return tag, out


def main() -> int:
    p = argparse.ArgumentParser(description="Vessel-time discarded by the 3-hour rule")
    p.add_argument("--src", type=Path, required=True, help="directory of fuel parquets")
    p.add_argument("--ports", type=Path, required=True, help="USACE Principal Ports GeoJSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    files = sorted(args.src.glob("ais_fuel_messages_v2_*.parquet"))
    if not files:
        raise SystemExit(f"no fuel parquet under {args.src}")
    frames = [out for _, out in imap(day_gaps, [(str(f), str(args.ports)) for f in files],
                                     args.workers) if not out.empty]
    if not frames:
        raise SystemExit("no port-day produced any rows")
    panel = pd.concat(frames, ignore_index=True).sort_values(["port", "date"])
    total = panel["hours_credited"] + panel["hours_discarded"]
    panel["discard_share"] = np.where(total > 0, panel["hours_discarded"] / total, 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    share = panel["hours_discarded"].sum() / total.sum()
    print(f"{len(panel):,} port-days, {share:.2%} of observed time discarded -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
