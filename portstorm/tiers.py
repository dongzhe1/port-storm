"""Build the national operational tier map from AIS navigational status.

Each grid cell is labelled from what vessels report over sampled days: moored -> berth
("port"); at anchor within APPROACH_KM of a port -> anchorage, within HOLD_KM ->
offshore_hold, farther -> offshore; under way within CHANNEL_KM -> channel, farther ->
offshore. Stationary tiers use a RES_FINE grid, moving tiers RES_COARSE.

    python -m portstorm tiers --src ais/ --ports ports.geojson --out tier_map.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .ais import find_day, list_days, read_polars
from .ports import load_polygons

RES_FINE = 0.005
RES_COARSE = 0.05
MIN_CELL_MSGS = 20
MOORED, AT_ANCHOR = 5, 1
UNDERWAY_SOG = 3.0

APPROACH_KM = 30.0
HOLD_KM = 100.0
CHANNEL_KM = 25.0

VERTEX_STRIDE = 25
TIERS = ["port", "channel", "anchorage", "offshore_hold", "offshore"]


def port_vertices(ports_path: Path) -> tuple[np.ndarray, np.ndarray]:
    polys = load_polygons(ports_path)
    verts, names = [], []
    for entry in polys:
        name, ring = entry[0], entry[1]
        a = np.asarray(ring)[::VERTEX_STRIDE]
        if len(a) == 0:
            a = np.asarray(ring)[:1]
        verts.append(a)
        names.extend([name] * len(a))
    return np.vstack(verts), np.asarray(names)


def nearest_port(lat: np.ndarray, lon: np.ndarray, verts: np.ndarray,
                 names: np.ndarray, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    d = np.empty(len(lat))
    n = np.empty(len(lat), dtype=object)
    for i in range(0, len(lat), chunk):
        s = slice(i, min(i + chunk, len(lat)))
        dy = (lat[s][:, None] - verts[None, :, 1]) * 110.57
        dx = ((lon[s][:, None] - verts[None, :, 0]) * 111.32
              * np.cos(np.radians(lat[s]))[:, None])
        dist = np.hypot(dx, dy)
        j = dist.argmin(1)
        d[s] = dist[np.arange(len(dist)), j]
        n[s] = names[j]
    return d, n


def accumulate(paths: list[Path], res: float) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_polars(p, columns=["IMO", "LAT", "LON", "SOG", "Status"])
        if df is None or df.height == 0:
            continue
        d = df.to_pandas()
        d = d[d["IMO"].notna() & (d["IMO"].astype(str).str.strip() != "")]
        for c in ("LAT", "LON", "SOG"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["Status"] = pd.to_numeric(d["Status"], errors="coerce")
        d = d.dropna(subset=["LAT", "LON"])
        if d.empty:
            continue
        sog = d["SOG"].fillna(0.0)
        moving = sog >= UNDERWAY_SOG
        d["n_moored"] = ((d["Status"] == MOORED) & ~moving).astype(np.int32)
        d["n_anchored"] = ((d["Status"] == AT_ANCHOR) & ~moving).astype(np.int32)
        d["n_underway"] = moving.astype(np.int32)
        d["gy"] = np.round(d["LAT"] / res).astype(np.int32)
        d["gx"] = np.round(d["LON"] / res).astype(np.int32)
        frames.append(d.groupby(["gy", "gx"], as_index=False)
                      .agg(n_msg=("SOG", "size"), n_moored=("n_moored", "sum"),
                           n_anchored=("n_anchored", "sum"),
                           n_underway=("n_underway", "sum")))
    if not frames:
        raise SystemExit("no usable AIS day")
    g = (pd.concat(frames, ignore_index=True).groupby(["gy", "gx"], as_index=False)
         .agg({"n_msg": "sum", "n_moored": "sum", "n_anchored": "sum",
               "n_underway": "sum"}))
    g["lat"] = g["gy"] * res
    g["lon"] = g["gx"] * res
    g["res_deg"] = res
    return g


KEEP = ["lat", "lon", "gy", "gx", "res_deg", "complex", "tier", "d_km",
        "n_msg", "n_moored", "n_anchored", "n_underway"]


def classify(cells: pd.DataFrame, ports_path: Path, keep: str) -> pd.DataFrame:
    g = cells[cells["n_msg"] >= MIN_CELL_MSGS].copy()
    if g.empty:
        return pd.DataFrame(columns=KEEP)
    verts, names = port_vertices(ports_path)
    g["d_km"], g["complex"] = nearest_port(g["lat"].to_numpy(), g["lon"].to_numpy(),
                                           verts, names)

    stopped = g["n_moored"] + g["n_anchored"]
    is_stationary = stopped >= g["n_underway"]

    if keep == "stationary":
        g = g[is_stationary & ((stopped > 0))].copy()
        if g.empty:
            return pd.DataFrame(columns=KEEP)
        mw = g["n_moored"] > g["n_anchored"]
        g["tier"] = np.where(
            mw, "port",
            np.where(g["d_km"] <= APPROACH_KM, "anchorage",
                     np.where(g["d_km"] <= HOLD_KM, "offshore_hold", "offshore")))
    else:
        g = g[~is_stationary & (g["n_underway"] > 0)].copy()
        if g.empty:
            return pd.DataFrame(columns=KEEP)
        g["tier"] = np.where(g["d_km"] <= CHANNEL_KM, "channel", "offshore")
    return g[KEEP]


def main() -> int:
    p = argparse.ArgumentParser(description="Build the operational tier map from AIS")
    p.add_argument("--src", type=Path, required=True, help="AIS root (flat or per-year)")
    p.add_argument("--ports", type=Path, required=True, help="USACE port polygons")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=40, help="days to sample, spread evenly")
    p.add_argument("--days", nargs="*", default=None,
                   help="explicit days (YYYY-MM-DD) instead of an even sample")
    args = p.parse_args()

    if args.days:
        wanted = list(args.days)
    else:
        available = list_days(args.src)
        if not available:
            raise SystemExit(f"no AIS day under {args.src}")
        step = max(1, len(available) // args.sample)
        wanted = available[::step][:args.sample]
    paths = [q for q in (find_day(args.src, d) for d in wanted) if q is not None]
    if not paths:
        raise SystemExit("none of the requested days could be located")

    fine = classify(accumulate(paths, RES_FINE), args.ports, "stationary")
    coarse = classify(accumulate(paths, RES_COARSE), args.ports, "moving")
    tiers = pd.concat([fine, coarse], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tiers.to_parquet(args.out, index=False)
    summ = tiers.groupby("tier").agg(cells=("n_msg", "size"), complexes=("complex", "nunique"))
    print(summ.reindex([t for t in TIERS if t in summ.index]).to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
