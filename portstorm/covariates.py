"""Observed storm conditions at each port from NOAA CO-OPS gauges and ASOS stations.

For every port-cyclone interaction: peak surge (highest water level during exposure minus the
median over REFERENCE_DAYS either side) and highest sustained wind at the nearest tide gauge,
and total and peak-hour rainfall at the nearest ASOS station. Requests are rate-limited
globally and cached on disk.

    python -m portstorm covariates --exposure storm_exposure.csv --ports ports.geojson \
        --out out/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .storms import haversine_km, load_usace_ports

REFERENCE_DAYS = 7
MIN_INTERVAL = 0.6
WORKERS = 3
MAX_RETRIES = 4
MAX_HOURLY_RAIN_IN = 5.0
BACKOFF_BASE = 5.0
TIMEOUT = 60

COOPS_MD = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
COOPS_DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

ASOS_STATIONS = "https://mesonet.agron.iastate.edu/sites/networks.php?network=_ALL_&format=csv&nohtml=on"

US_STATES = set("""AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI""".split())


_rate_lock = threading.Lock()
_last_call = [0.0]
_cache_lock = threading.Lock()
_CACHE_DIR = [None]


def _throttle() -> None:
    with _rate_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _cache_path(url: str, params: dict) -> Path | None:
    if _CACHE_DIR[0] is None:
        return None
    key = hashlib.sha1(f"{url}?{urllib.parse.urlencode(sorted(params.items()))}"
                       .encode()).hexdigest()
    return _CACHE_DIR[0] / f"{key}.txt"


def _get(url: str, params: dict) -> str:
    cp = _cache_path(url, params)
    if cp is not None and cp.is_file():
        return cp.read_text(encoding="utf-8")

    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "portstorm/0.1"})
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out = r.read().decode("utf-8", errors="replace")
            if cp is not None:
                with _cache_lock:
                    cp.write_text(out, encoding="utf-8")
            return out
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (403, 429, 500, 502, 503, 504):
                raise
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
        except Exception as exc:
            last = exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
    raise last if last else RuntimeError("request failed")


def asos_stations(cache: Path) -> pd.DataFrame:
    if not cache.is_file() or cache.stat().st_size == 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
                urllib.request.Request(ASOS_STATIONS,
                                       headers={"User-Agent": "portstorm/0.1"}),
                timeout=300) as r:
            cache.write_bytes(r.read())
    df = pd.read_csv(cache)
    net = df["iem_network"].astype(str)
    keep = net.str.match(r"^[A-Z]{2}_ASOS$") & net.str[:2].isin(US_STATES)
    return df[keep].dropna(subset=["lat", "lon"])[["stid", "station_name", "lat", "lon"]]


def coops_stations() -> pd.DataFrame:
    d = json.loads(_get(COOPS_MD, {"type": "waterlevels"}))
    return pd.DataFrame([{"id": s["id"], "name": s["name"],
                          "lat": float(s["lat"]), "lon": float(s["lng"])}
                         for s in d["stations"]])


def coops_series(station: str, start, end, product: str) -> pd.Series:
    params = {"product": product, "application": "research",
              "begin_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d"),
              "station": station, "time_zone": "gmt", "units": "metric", "format": "json"}
    if product == "water_level":
        params["datum"] = "MSL"
    try:
        d = json.loads(_get(COOPS_DATA, params))
    except Exception:
        return pd.Series(dtype=float)
    rows = d.get("data") or d.get("predictions") or []
    if not rows:
        return pd.Series(dtype=float)
    key = "s" if product == "wind" else "v"
    out = {}
    for r in rows:
        try:
            out[pd.Timestamp(r["t"])] = float(r[key])
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(out).sort_index()


def asos_rain_mm(station: str, start, end) -> tuple[float | None, float | None]:
    params = {"station": station, "data": "p01i", "tz": "UTC",
              "year1": start.year, "month1": start.month, "day1": start.day,
              "year2": end.year, "month2": end.month, "day2": end.day,
              "format": "onlycomma", "missing": "empty"}
    try:
        text = _get(ASOS, params)
    except Exception:
        return None, None
    times, vals = [], []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3 and parts[2].strip():
            try:
                vals.append(float(parts[2]))
                times.append(pd.Timestamp(parts[1]))
            except (ValueError, TypeError):
                pass
    if not vals:
        return None, None
    hourly = pd.Series(vals, index=pd.DatetimeIndex(times)).resample("h").max().dropna()
    hourly = hourly[hourly <= MAX_HOURLY_RAIN_IN]
    if hourly.empty:
        return None, None
    return round(float(hourly.sum()) * 25.4, 2), round(float(hourly.max()) * 25.4, 2)


def main() -> int:
    p = argparse.ArgumentParser(description="Observed storm conditions at each port")
    p.add_argument("--exposure", type=Path, required=True)
    p.add_argument("--ports", type=Path, required=True, help="USACE Principal Ports GeoJSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--skip-rain", action="store_true")
    p.add_argument("--asos-cache", type=Path, default=Path("cache/asos_stations.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache/noaa"))
    p.add_argument("--workers", type=int, default=WORKERS)
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR[0] = args.cache_dir
    exp = pd.read_csv(args.exposure, parse_dates=["exposure_start", "exposure_end"])
    if args.season:
        exp = exp[exp["season"] == args.season]
    if args.name:
        exp = exp[exp["name"].astype(str).str.upper() == args.name.upper()]
    if exp.empty:
        p.error("no interactions match the filters")
    args.out.mkdir(parents=True, exist_ok=True)

    ports = load_usace_ports(args.ports)
    st = coops_stations()
    asos = asos_stations(args.asos_cache)
    nearest = {}
    for port, (la, lo) in ports.items():
        dc = haversine_km(la, lo, st["lat"].to_numpy(), st["lon"].to_numpy())
        da = haversine_km(la, lo, asos["lat"].to_numpy(), asos["lon"].to_numpy())
        ic, ia = int(np.argmin(dc)), int(np.argmin(da))
        nearest[port] = (st.iloc[ic]["id"], st.iloc[ic]["name"], round(float(dc.min()), 1),
                         asos.iloc[ia]["stid"], round(float(da.min()), 1))

    def one(rec_in):
        _, e = rec_in
        port = e["port"]
        if port not in nearest:
            return None
        sid_station, _, sdist, asos_id, adist = nearest[port]
        s0 = e["exposure_start"].normalize()
        s1 = e["exposure_end"].normalize() + pd.Timedelta(days=1)
        r0 = s0 - pd.Timedelta(days=REFERENCE_DAYS)
        r1 = s1 + pd.Timedelta(days=REFERENCE_DAYS)
        rec = {"SID": e["SID"], "name": e["name"], "season": e["season"], "port": port,
               "coops_station": sid_station, "coops_dist_km": sdist,
               "asos_station": asos_id, "asos_dist_km": adist}
        wl = coops_series(sid_station, r0, r1, "water_level")
        rec["surge_height_m"] = None
        if len(wl):
            win = wl[(wl.index >= s0) & (wl.index <= s1)]
            calm = wl[(wl.index < s0) | (wl.index > s1)]
            typical = float(calm.median() if len(calm) else wl.median())
            rec["surge_height_m"] = round(float(win.max()) - typical, 3) if len(win) else None
        wd = coops_series(sid_station, s0, s1, "wind")
        rec["max_wind_ms"] = round(float(wd.max()), 2) if len(wd) else None
        rec["rainfall_mm"] = rec["peak_hourly_rain_mm"] = None
        if not args.skip_rain:
            try:
                rec["rainfall_mm"], rec["peak_hourly_rain_mm"] = asos_rain_mm(asos_id, s0, s1)
            except Exception:
                pass
        return rec

    rows, errors = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one, r) for r in exp.iterrows()]):
            try:
                rec = fut.result()
            except Exception:
                errors += 1
                rec = None
            if rec:
                rows.append(rec)
    cov = pd.DataFrame(rows)
    cov.to_csv(args.out / "storm_covariates.csv", index=False)
    print(f"{len(cov):,} rows, {errors} failed -> {args.out / 'storm_covariates.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
