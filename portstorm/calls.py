"""Port calls and daily vessel counts from AIS, following CyPort's definition.

Cargo and tanker vessels only, positions matched to USACE port polygons, visits shorter than
MIN_CALL_HOURS dropped. Two stages, because a call can span midnight:

    python -m portstorm calls extract --src ais/ --out presence/ --ports ports.geojson
    python -m portstorm calls visits  --src presence/ --out out/
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

from .ais import find_day, list_days, read_polars
from .parallel import imap
from .ports import assign_ports, grid_index, load_polygons

MIN_CALL_HOURS = 4.0
STITCH_GAP_HOURS = 6.0

CARGO_TANKER = frozenset(set(range(70, 90)) | {1003, 1004, 1016, 1017, 1024})
HEADER = b"mmsi,imo,port,first_seen,last_seen,n_msgs\n"
_GEOM: dict = {}


def _geometry(path: str):
    if path not in _GEOM:
        polys = load_polygons(Path(path))
        _GEOM[path] = (polys, grid_index(polys))
    return _GEOM[path]


def read_day(path: Path) -> pd.DataFrame:
    df = read_polars(path, ["MMSI", "BaseDateTime", "LAT", "LON", "VesselType",
                            "IMO"]).to_pandas()
    df = df[df["VesselType"].isin(CARGO_TANKER)]
    return df.dropna(subset=["BaseDateTime", "LAT", "LON"])


def extract_day(args: tuple) -> tuple[str, int, int, str]:
    date, src, out, ports_path, overwrite = args
    dest = Path(out) / f"presence_{date}.csv.gz"
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return date, 0, 0, "cached"
    try:
        polys, idx = _geometry(ports_path)
        day_file = find_day(Path(src), date)
        if day_file is None:
            return date, 0, 0, "FAILED: no archive found for this date"
        df = read_day(day_file)
        if df.empty:
            dest.write_bytes(gzip.compress(HEADER))
            return date, 0, 0, "ok"
        df["port"] = assign_ports(df["LON"].to_numpy(), df["LAT"].to_numpy(), polys, idx)
        df = df[df["port"] != ""]
        if df.empty:
            dest.write_bytes(gzip.compress(HEADER))
            return date, 0, 0, "ok"
        g = (df.groupby(["MMSI", "port"], as_index=False)
               .agg(imo=("IMO", "first"), first_seen=("BaseDateTime", "min"),
                    last_seen=("BaseDateTime", "max"), n_msgs=("BaseDateTime", "size"))
               .rename(columns={"MMSI": "mmsi"}))
        tmp = dest.with_suffix(".gz.part")
        g.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(dest)
        return date, len(df), len(g), "ok"
    except Exception as exc:
        return date, 0, 0, f"FAILED: {type(exc).__name__}: {exc}"


def build_visits(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(src.glob("presence_*.csv.gz"))
    if not files:
        raise SystemExit(f"no presence_*.csv.gz in {src} -- run the extract stage first")
    pres = pd.concat([pd.read_csv(f, parse_dates=["first_seen", "last_seen"])
                      for f in files], ignore_index=True)
    pres = pres.sort_values(["mmsi", "port", "first_seen"])

    visits = []
    for (mmsi, port), g in pres.groupby(["mmsi", "port"], sort=False):
        start = g["first_seen"].iloc[0]
        end = g["last_seen"].iloc[0]
        imo = g["imo"].iloc[0]
        msgs = g["n_msgs"].iloc[0]
        for row in g.iloc[1:].itertuples(index=False):
            if (row.first_seen - end).total_seconds() / 3600 <= STITCH_GAP_HOURS:
                end = max(end, row.last_seen)
                msgs += row.n_msgs
            else:
                visits.append((mmsi, imo, port, start, end, msgs))
                start, end, msgs = row.first_seen, row.last_seen, row.n_msgs
        visits.append((mmsi, imo, port, start, end, msgs))

    v = pd.DataFrame(visits, columns=["mmsi", "imo", "port", "arrival", "departure",
                                      "n_msgs"])
    v["duration_h"] = (v["departure"] - v["arrival"]).dt.total_seconds() / 3600
    v["is_call"] = v["duration_h"] >= MIN_CALL_HOURS
    v["date"] = v["arrival"].dt.normalize()

    calls = v[v["is_call"]]
    daily = (calls.groupby(["date", "port"], as_index=False)
                  .agg(port_calls=("mmsi", "size"), distinct_vessels=("mmsi", "nunique"),
                       median_duration_h=("duration_h", "median")))
    daily["date"] = pd.to_datetime(daily["date"])

    span = calls[["mmsi", "port", "arrival", "departure"]].copy()
    span["d0"] = span["arrival"].dt.normalize()
    span["d1"] = span["departure"].dt.normalize()
    rows = []
    for r in span.itertuples(index=False):
        for d in pd.date_range(r.d0, r.d1, freq="D"):
            rows.append((d, r.port, r.mmsi))
    present = pd.DataFrame(rows, columns=["date", "port", "mmsi"])
    present["date"] = pd.to_datetime(present["date"])
    present = (present.groupby(["date", "port"], as_index=False)
                      .agg(vessels_present=("mmsi", "nunique")))
    daily = daily.merge(present, on=["date", "port"], how="outer")
    for c in ("port_calls", "distinct_vessels", "vessels_present"):
        daily[c] = daily[c].fillna(0).astype(int)
    return v, daily


def main() -> int:
    p = argparse.ArgumentParser(description="Port calls and daily vessel counts from AIS")
    sub = p.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract", help="per-day port presence intervals")
    e.add_argument("--src", type=Path, required=True, help="directory of daily AIS files")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--ports", type=Path, required=True, help="USACE Principal Ports GeoJSON")
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("visits", help="stitch intervals into calls and aggregate")
    v.add_argument("--src", type=Path, required=True, help="directory of presence_*.csv.gz")
    v.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.stage == "extract":
        if not args.ports.is_file():
            p.error(f"--ports not found: {args.ports}")
        dates = list_days(args.src)
        if not dates:
            p.error(f"no AIS day files in {args.src}")
        args.out.mkdir(parents=True, exist_ok=True)
        tasks = [(d, str(args.src), str(args.out), str(args.ports), args.overwrite)
                 for d in dates]
        done = failed = cached = 0
        for date, _, _, status in imap(extract_day, tasks, args.workers):
            done += 1
            if status == "cached":
                cached += 1
            elif status != "ok":
                failed += 1
                print(f"  {date}  {status}", flush=True)
        print(f"days {done} | cached {cached} | failed {failed}")
        return 1 if failed else 0

    args.out.mkdir(parents=True, exist_ok=True)
    visits, daily = build_visits(args.src)
    visits.to_csv(args.out / "port_visits.csv", index=False)
    daily.to_csv(args.out / "port_daily_calls.csv", index=False)
    n_files = len(list(Path(args.src).glob("presence_*.csv.gz")))
    n_days = daily["date"].nunique()
    if n_files and n_days < 0.9 * n_files:
        print(f"WARNING: the panel covers {n_days:,} of {n_files:,} input days",
              file=sys.stderr)
    print(f"{len(visits):,} visits, {int(visits['is_call'].sum()):,} calls, "
          f"{len(daily):,} port-days -> {args.out / 'port_daily_calls.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
