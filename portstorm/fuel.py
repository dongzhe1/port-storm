"""Run the GHG4 fuel model over daily AIS files, one parquet of message-level fuel per day.

    python -m portstorm fuel --src ais/ --out fuel/ --fleet fleet_registry.csv \
        --start 2021-01-01 --end 2021-12-31
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import ghg4
from .ais import CANONICAL, find_day, list_days, read_polars
from .parallel import imap

LOWER = {v: k for k, v in CANONICAL.items() if k != "transceiverclass"}
_FLEET: dict = {}


def read_ais(path: Path):
    df = read_polars(path)
    return df.rename({c: LOWER[c] for c in df.columns if c in LOWER})


def run_day(task: tuple) -> tuple[str, str, float]:
    date, src, out, fleet = task
    t0 = time.time()
    dest = Path(out) / f"ais_fuel_messages_v2_{date}.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return date, "cached", 0.0
    try:
        path = find_day(Path(src), date)
        if path is None:
            return date, "FAILED: no archive for this date", 0.0
        if fleet not in _FLEET:
            _FLEET[fleet] = ghg4.load_fleet(fleet)
        df = ghg4.fuel_messages(read_ais(path), _FLEET[fleet])
        tmp = dest.with_suffix(".parquet.part")
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)
        return date, "ok", time.time() - t0
    except Exception as exc:
        return date, f"FAILED: {type(exc).__name__}: {exc}", time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(description="GHG4 fuel per AIS message, one file per day")
    p.add_argument("--src", type=Path, required=True, help="directory of daily AIS files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fleet", type=Path, required=True, help="fleet registry CSV")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not args.fleet.is_file():
        p.error(f"--fleet not found: {args.fleet}")
    dates = [d for d in list_days(args.src)
             if (args.start is None or d >= args.start) and (args.end is None or d <= args.end)]
    if not dates:
        p.error(f"no AIS days in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(d, str(args.src), str(args.out), str(args.fleet)) for d in dates]
    done = failed = cached = 0
    for date, status, _ in imap(run_day, tasks, args.workers):
        done += 1
        if status == "cached":
            cached += 1
        elif status != "ok":
            failed += 1
            print(f"  {date}  {status}", flush=True)
    print(f"days {done} | cached {cached} | failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
