"""Download the public inputs.

    ais        NOAA MarineCadastre daily AIS, 2015 onward (~300 MB per day)
    reference  USACE Principal Ports polygons and the IBTrACS North Atlantic tracks
    cyport     CyPort's released tables (for `portstorm cyport`)

Downloads resume from partial files, and complete files are skipped.

    python -m portstorm download reference --out data/
    python -m portstorm download ais --start 2021-08-01 --end 2021-09-30 --out data/ais
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

AIS_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"
REFERENCE = {
    "usace_principal_ports.geojson": "https://geospatial-usace.opendata.arcgis.com/api/"
    "download/v1/items/16d570a7aa054943aabf52b0032a3b57/geojson?layers=0",
    "ibtracs.NA.csv": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv",
}
CYPORT_URL = "https://raw.githubusercontent.com/ChenchenMobility/Maritime-Data-CyPort/main/"
CYPORT = {
    "interaction_data.csv": "Tropical_cyclone_interaction/interaction_data.csv",
    "port_vessel_count.csv": "All_time_data/port_vessel_count.csv",
    "port&resources.csv": "All_time_data/port%26resources.csv",
}
RETRIES = 4


def fetch(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return "exists"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(RETRIES):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "portstorm/0.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            with requests.get(url, headers=headers, stream=True, timeout=(15, 180)) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with part.open(mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if dest.suffix == ".zip":
                with zipfile.ZipFile(part) as z:
                    z.namelist()
            part.replace(dest)
            return "ok"
        except (requests.RequestException, zipfile.BadZipFile):
            if part.exists() and attempt:
                part.unlink()
            time.sleep(3 * 2 ** attempt)
    return "failed"


def report(results: dict) -> int:
    for name, status in results.items():
        if status not in ("ok", "exists"):
            print(f"  {name}: {status}")
    ok = sum(s in ("ok", "exists") for s in results.values())
    print(f"{ok}/{len(results)} files present")
    return 0 if ok == len(results) else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Download public inputs")
    sub = p.add_subparsers(dest="what", required=True)
    a = sub.add_parser("ais")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--threads", type=int, default=4)
    for name in ("reference", "cyport"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.what == "ais":
        d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
        days = [d0 + timedelta(n) for n in range((d1 - d0).days + 1)]
        jobs = {f"{d}": (AIS_URL.format(y=d.year, m=d.month, d=d.day),
                         args.out / str(d.year) / f"AIS_{d:%Y_%m_%d}.zip") for d in days}
    elif args.what == "reference":
        jobs = {n: (u, args.out / n) for n, u in REFERENCE.items()}
    else:
        jobs = {n: (CYPORT_URL + u, args.out / n) for n, u in CYPORT.items()}
    with ThreadPoolExecutor(max_workers=getattr(args, "threads", 3)) as pool:
        status = dict(zip(jobs, pool.map(lambda j: fetch(*j), jobs.values())))
    return report(status)


if __name__ == "__main__":
    sys.exit(main())
