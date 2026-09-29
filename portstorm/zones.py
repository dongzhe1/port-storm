"""Aggregate message-level fuel and CO2 to ports or operational tiers, per day.

Ports mode (--ports) assigns each message to a USACE port polygon and writes
port_daily_emissions.csv and port_daily_emissions_by_class.csv. Tier mode (--tier-map) labels
each message berth/channel/anchorage/offshore_hold/offshore from the map built by
`portstorm tiers` and writes emissions_by_zone.csv.

    python -m portstorm zones --src fuel/ --out out/ --ports ports.geojson
    python -m portstorm zones --src fuel/ --out out/tiers --tier-map tier_map.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .parallel import imap
from .ports import assign_ports, grid_index, load_polygons

EMISSION_COLS = ["fc_main_ton", "fc_AE_ton", "fc_boiler_ton", "fc_total_ton", "co2_ton"]
TIERS = ["port", "channel", "anchorage", "offshore_hold", "offshore"]
TIER_RES_DEG = 0.005
TRANSIT_SOG = 3.0
NEVER = np.datetime64("2262-01-01")
TYPE_TO_CLASS = {
    "Container": "cargo", "General": "cargo", "Refrigerated-Cargo": "cargo",
    "Auto": "cargo", "Bulk": "cargo",
    "Oil": "tanker", "Chemical": "tanker", "Tanker": "tanker",
    "Other-Liquids": "tanker", "Liquified-Gas": "tanker",
    "Passenger": "passenger", "Fishing": "fishing",
}
_POLYS: dict = {}
_TIER_MAPS: dict = {}
_INSTALL: dict = {}


def _geometry(ports_path: str):
    if ports_path not in _POLYS:
        polys = load_polygons(Path(ports_path))
        _POLYS[ports_path] = (polys, grid_index(polys))
    return _POLYS[ports_path]


def _install_dates(path: str) -> dict:
    if path not in _INSTALL:
        d = pd.read_csv(path, parse_dates=["install_date"])
        _INSTALL[path] = dict(zip(d["imo"].astype("int64").to_numpy(),
                                  d["install_date"].to_numpy()))
    return _INSTALL[path]


def _tier_map(path: str) -> list:
    if path not in _TIER_MAPS:
        m = pd.read_parquet(path)
        if "res_deg" not in m.columns:
            m = m.assign(res_deg=TIER_RES_DEG)
        levels = []
        for res in sorted(m["res_deg"].unique()):
            s = m[m["res_deg"] == res]
            levels.append((float(res),
                           dict(zip(zip(s["gy"].to_numpy(), s["gx"].to_numpy()),
                                    zip(s["complex"].to_numpy(), s["tier"].to_numpy())))))
        _TIER_MAPS[path] = levels
    return _TIER_MAPS[path]


def add_fitted(df: pd.DataFrame, dates_path: str | None,
               date_col: str = "base_date_time") -> pd.DataFrame:
    if not dates_path:
        return df
    lut = _install_dates(dates_path)
    when = pd.to_datetime(df[date_col]).to_numpy()
    inst = np.array([lut.get(int(i), NEVER) if i == i else NEVER for i in df["imo"].to_numpy()],
                    dtype="datetime64[ns]")
    df = df.copy()
    df["scrubber_fitted"] = (when >= inst).astype(np.int8)
    return df


def by_usace_port(df: pd.DataFrame, polys: list, idx: dict) -> pd.DataFrame:
    out = df.copy()
    out["port"] = assign_ports(out["longitude"].to_numpy(), out["latitude"].to_numpy(),
                               polys, idx)
    out = out[out["port"] != ""]
    out["vclass"] = out["vessel_type_str"].map(TYPE_TO_CLASS).fillna("other")
    return out


def classify_by_map(df: pd.DataFrame, map_path: str) -> pd.DataFrame:
    df = df.copy()
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    hit: list = [None] * len(df)
    for res, lut in _tier_map(map_path):
        gy = np.round(lat / res).astype(np.int32)
        gx = np.round(lon / res).astype(np.int32)
        for i, k in enumerate(zip(gy, gx)):
            if hit[i] is None:
                hit[i] = lut.get(k)
    df["complex"] = [h[0] if h else "Outside" for h in hit]
    df["tier"] = [h[1] if h else "outside" for h in hit]
    moving = df["sog"].to_numpy() >= TRANSIT_SOG
    df.loc[moving & (df["tier"] == "port"), "tier"] = "channel"
    df["in_region"] = df["tier"] != "outside"
    df["vclass"] = df["vessel_type_str"].map(TYPE_TO_CLASS).fillna("other")
    return df


def cache_fingerprint(src: Path, **inputs) -> str:
    files = sorted(src.glob("ais_fuel_messages_v2_*.parquet"))
    newest = max((f.stat().st_mtime for f in files), default=0.0)
    parts = [f"fuel={len(files)}@{int(newest)}"]
    for key in sorted(inputs):
        v = inputs[key]
        if v is None:
            parts.append(f"{key}=none")
        elif isinstance(v, (str, Path)) and Path(v).is_file():
            parts.append(f"{key}={Path(v).name}:{Path(v).stat().st_size}")
        else:
            parts.append(f"{key}={v}")
    return " ".join(parts)


def prepare_cache(cache_dir: Path | None, fingerprint: str) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = cache_dir / ".config"
    have = stamp.read_text().strip() if stamp.is_file() else None
    if have != fingerprint:
        stale = list(cache_dir.glob("*.parquet"))
        if stale:
            print(f"cache: configuration changed, discarding {len(stale)} cached days")
        for f in stale:
            f.unlink()
    stamp.write_text(fingerprint)
    return cache_dir


def _cached(cache: Path | None):
    if cache is not None and cache.is_file() and cache.stat().st_size > 0:
        return pd.read_parquet(cache)
    return None


def _store(cache: Path | None, out: pd.DataFrame) -> None:
    if cache is not None and not out.empty:
        tmp = cache.with_suffix(".parquet.part")
        out.to_parquet(tmp, index=False)
        tmp.replace(cache)


def day_ports(task: tuple):
    path, ports_path, cache_dir, dates_path = task
    tag = Path(path).stem.rsplit("_", 1)[-1]
    cache = Path(cache_dir) / f"portemis_{tag}.parquet" if cache_dir else None
    if (hit := _cached(cache)) is not None:
        return tag, hit, "cached"
    polys, idx = _geometry(ports_path)
    df = add_fitted(by_usace_port(pd.read_parquet(path), polys, idx), dates_path)
    if df.empty:
        return tag, pd.DataFrame(), "ok"
    df["date"] = pd.to_datetime(df["base_date_time"]).dt.date
    keys = ["date", "port", "vclass", "scrubber_tech"]
    if "scrubber_fitted" in df.columns:
        keys.append("scrubber_fitted")
    out = (df.groupby(keys, as_index=False)
             .agg(n_messages=("sog", "size"), n_vessels=("imo", "nunique"),
                  hours=("dt_h", "sum"), **{c: (c, "sum") for c in EMISSION_COLS}))
    _store(cache, out)
    return tag, out, "ok"


def day_tiers(task: tuple):
    path, tier_map, cache_dir, dates_path = task
    tag = Path(path).stem.rsplit("_", 1)[-1]
    cache = Path(cache_dir) / f"tieremis_{tag}.parquet" if cache_dir else None
    if (hit := _cached(cache)) is not None:
        return tag, hit, "cached"
    df = add_fitted(classify_by_map(pd.read_parquet(path), tier_map), dates_path)
    df["date"] = pd.to_datetime(df["base_date_time"]).dt.date
    keys = ["date", "complex", "tier", "vclass", "scrubber_tech"]
    if "scrubber_fitted" in df.columns:
        keys.append("scrubber_fitted")
    out = (df.groupby(keys, as_index=False)
             .agg(n_messages=("sog", "size"), n_vessels_in_cell=("imo", "nunique"),
                  hours=("dt_h", "sum"), **{c: (c, "sum") for c in EMISSION_COLS}))
    _store(cache, out)
    return tag, out, "ok"


def run_days(fn, tasks, workers: int) -> list:
    return [out for _, out, _ in imap(fn, tasks, workers) if len(out)]


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate fuel and CO2 to ports or tiers")
    p.add_argument("--src", type=Path, required=True, help="directory of fuel parquets")
    p.add_argument("--out", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ports", type=Path, help="USACE Principal Ports GeoJSON")
    mode.add_argument("--tier-map", type=Path, help="tier map from `portstorm tiers`")
    p.add_argument("--scrubber-dates", type=Path, default=None,
                   help="CSV of imo,install_date; adds the scrubber_fitted flag")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    files = sorted(args.src.glob("ais_fuel_messages_v2_*.parquet"))
    if not files:
        p.error(f"no ais_fuel_messages_v2_*.parquet in {args.src}")
    geo = args.ports or args.tier_map
    if not geo.is_file():
        p.error(f"not found: {geo}")
    args.out.mkdir(parents=True, exist_ok=True)
    dates = str(args.scrubber_dates) if args.scrubber_dates else None
    cache = prepare_cache(args.cache_dir, cache_fingerprint(
        args.src, mode="ports" if args.ports else "tiers", ports=args.ports,
        tier_map=args.tier_map, scrubber_dates=args.scrubber_dates))
    tasks = [(str(f), str(geo), str(cache) if cache else None, dates) for f in files]

    if args.ports:
        frames = run_days(day_ports, tasks, args.workers)
        if not frames:
            print("no message fell inside any port polygon")
            return 1
        keys = ["date", "port", "vclass", "scrubber_tech"]
        detail = pd.concat(frames, ignore_index=True)
        detail = detail.sort_values(keys + [c for c in ("scrubber_fitted",) if c in detail])
        daily = (detail.groupby(["date", "port"], as_index=False)
                       .agg(n_messages=("n_messages", "sum"), hours=("hours", "sum"),
                            **{c: (c, "sum") for c in EMISSION_COLS}))
        daily.to_csv(args.out / "port_daily_emissions.csv", index=False)
        detail.to_csv(args.out / "port_daily_emissions_by_class.csv", index=False)
        print(f"{len(daily):,} port-days -> {args.out / 'port_daily_emissions.csv'}")
        return 0

    frames = run_days(day_tiers, tasks, args.workers)
    if not frames:
        print("no messages")
        return 1
    zone = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "complex", "tier", "vclass", "scrubber_tech"])
    zone.to_csv(args.out / "emissions_by_zone.csv", index=False)
    inside = zone[zone["tier"] != "outside"]
    share = inside.groupby("tier")["fc_total_ton"].sum().reindex(
        [t for t in TIERS if t in set(inside["tier"])])
    print(f"{len(zone):,} rows -> {args.out / 'emissions_by_zone.csv'}")
    print((100 * share / share.sum()).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
