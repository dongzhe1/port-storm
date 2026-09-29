"""Generate a small synthetic dataset that runs the whole pipeline and the analysis.

Writes, in the layout run_pipeline.sh reads: eleven ports at real Gulf and Atlantic locations
(usace_principal_ports.geojson), thirteen 2021 storms in IBTrACS layout (ibtracs.NA.csv), four years
(2018-2021) of two-hourly AIS for cargo ships, tankers, cruise ships and tugs (ais/<year>/), a
fleet registry in the Clarksons layout (fleet_registry.csv), scrubber install dates, and storm
covariates in the layout `portstorm covariates` writes. Storms cut arrivals, lengthen waits
(offshore when the storm is close, and for ten days after) and shorten stays in proportion to how
close and strong they are; tugs ignore them. 2018-2020 are storm-free baseline years.

    python -m portstorm fake --out demo/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START, END, STORM_YEAR, STEP_H, HALF = "2018-01-01", "2021-12-31", 2021, 2, 0.08
PORTS = [("Houston Port Authority, TX", 29.72, -95.15, (-0.6, 0.8), (1, 1, 6, 0), 6, 3),
         ("Corpus Christi, TX", 27.81, -97.40, (-0.3, 0.95), (2, 2, 1, 1), 5, 2),
         ("Port Freeport, TX", 28.94, -95.31, (-0.8, 0.6), (1, 1, 1, 0), 3, 14),
         ("Port Arthur, TX", 29.87, -93.93, (-1, 0), (0, 1, 5, 0), 4, 2),
         ("Lake Charles Harbor District, LA", 30.22, -93.25, (-1, 0), (2, 2, 1, 1), 5, 2),
         ("New Orleans, LA", 29.93, -90.06, (-0.7, 0.7), (2, 2, 1, 1), 6, 3),
         ("Port of Pascagoula, MS", 30.35, -88.56, (-1, 0), (1, 1, 1, 0), 3, 14),
         ("Mobile, AL", 30.70, -88.04, (-1, 0), (3, 4, 0, 0), 5, 2),
         ("Tampa Port Authority, FL", 27.93, -82.43, (-0.7, -0.7), (3, 3, 0, 0), 5, 2),
         ("Jacksonville, FL", 30.40, -81.55, (0, 1), (4, 2, 0, 1), 4, 2),
         ("Port of Savannah, GA", 32.12, -81.14, (-0.3, 0.95), (6, 1, 1, 0), 6, 2)]
MERCHANT = [("Fully Cellular Container", 70, 40_000, 80_000), ("Bulk Carrier", 70, 30_000, 80_000),
            ("Tanker", 80, 40_000, 120_000), ("Cruise Ship", 60, 5_000, 12_000)]
STORMS = [("ANA", "06-10", 25.0, -94.5, 28.9, -95.3, 45), ("BILL", "06-28", 26.0, -88.0, 30.3, -88.4, 58),
          ("CLAUDETTE", "07-05", 27.0, -77.5, 30.3, -81.3, 95), ("DANNY", "07-20", 24.0, -95.0, 27.9, -97.2, 100),
          ("ELSA", "08-05", 25.0, -92.5, 29.9, -93.6, 125), ("FRED", "08-20", 28.0, -78.0, 32.0, -80.9, 90),
          ("GRACE", "08-30", 25.5, -94.0, 29.2, -95.1, 60), ("HENRI", "09-08", 24.5, -84.5, 27.9, -82.9, 55),
          ("IDA", "09-15", 25.0, -89.0, 29.4, -89.9, 120), ("JULIAN", "09-28", 25.5, -95.5, 29.4, -94.7, 42),
          ("KATE", "10-02", 25.5, -92.5, 29.8, -93.0, 48), ("LARRY", "06-18", 23.5, -96.0, 27.6, -97.2, 52),
          ("MINDY", "07-28", 24.0, -85.0, 27.8, -82.7, 85)]
AIS_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselName",
            "IMO", "CallSign", "VesselType", "Status", "Length", "Width", "Draft", "Cargo",
            "TransceiverClass"]


def sshs(wind: float) -> int:
    return next((c for c, lo in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64), (0, 34))
                 if wind >= lo), -1)


def km(lat1, lon1, lat2, lon2):
    p = np.radians
    a = (np.sin(p(lat2 - lat1) / 2) ** 2
         + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def storm_tracks() -> pd.DataFrame:
    rows = []
    for name, md, la0, lo0, la1, lo1, peak in STORMS:
        t0 = pd.Timestamp(f"{STORM_YEAR}-{md}")
        sid = f"{STORM_YEAR}{t0.dayofyear:03d}N{int(la0):02d}{int(360 + lo0):03d}"
        for i in range(40):
            f = i / 39
            wind = (30 + (peak - 30) * np.sin(np.pi / 2 * f / 0.8) if f <= 0.8
                    else peak - (peak - 25) * (f - 0.8) / 0.2)
            lat, lon = la0 + (la1 - la0) * f / 0.8, lo0 + (lo1 - lo0) * f / 0.8 + 0.3 * np.sin(4 * f)
            land = f > 0.8
            rows.append({"SID": sid, "SEASON": STORM_YEAR, "NAME": name,
                         "ISO_TIME": t0 + pd.Timedelta(hours=3 * i), "NATURE": "TS",
                         "LAT": round(lat, 2), "LON": round(lon, 2), "USA_WIND": round(wind),
                         "USA_PRES": round(1010 - 0.8 * wind), "USA_SSHS": sshs(wind),
                         "LANDFALL": 0 if land else 50, "DIST2LAND": 0 if land else 60})
    return pd.DataFrame(rows)


def storm_pressure(tracks: pd.DataFrame, lat: float, lon: float, days: pd.DatetimeIndex) -> np.ndarray:
    d = km(lat, lon, tracks["LAT"], tracks["LON"])
    hit = np.clip(1 - d / 300, 0, 1) * (0.4 + 0.12 * tracks["USA_SSHS"].clip(lower=0))
    by_day = pd.Series(hit.to_numpy(), index=tracks["ISO_TIME"].dt.normalize()).groupby(level=0).max()
    return by_day.reindex(days).fillna(0.0).rolling(3, center=True, min_periods=1).max().to_numpy()


def covariates(tracks: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (sid, name), g in tracks.groupby(["SID", "NAME"]):
        for i, (port, lat, lon, *_) in enumerate(PORTS):
            d = km(lat, lon, g["LAT"], g["LON"])
            j = int(np.argmin(d))
            if d.iloc[j] > 500:
                continue
            w = g["USA_WIND"].iloc[j] * np.clip(1 - d.iloc[j] / 400, 0.05, 1)
            rows.append({"SID": sid, "name": name, "season": STORM_YEAR, "port": port,
                         "coops_station": 8_770_000 + i, "coops_dist_km": 5.0,
                         "asos_station": f"K{i:03d}", "asos_dist_km": 8.0,
                         "surge_height_m": round(0.02 * w + rng.uniform(0, 0.2), 3),
                         "max_wind_ms": round(0.514 * w * 0.6, 2),
                         "rainfall_mm": round(1.5 * w * rng.uniform(0.5, 1.5), 1),
                         "peak_hourly_rain_mm": round(0.2 * w * rng.uniform(0.5, 1.5), 1)})
    return pd.DataFrame(rows)


def ports_geojson() -> dict:
    feats = [{"type": "Feature", "properties": {"PORTNAME": n, "TYPE": "Coastal"},
              "geometry": {"type": "Polygon", "coordinates": [[
                  [lo - HALF, la - HALF], [lo + HALF, la - HALF], [lo + HALF, la + HALF],
                  [lo - HALF, la + HALF], [lo - HALF, la - HALF]]]}} for n, la, lo, *_ in PORTS]
    return {"type": "FeatureCollection", "features": feats}


def simulate(rng: np.random.Generator, tracks: pd.DataFrame):
    days = pd.date_range(START, END)
    msgs, fleet, imo = [], [], 9_100_000
    for lat0, lon0, (sy, sx), mix, rate, n_tugs in (p[1:] for p in PORTS):
        at = lambda r, j=0.0: (lat0 + sy * r + rng.uniform(-j, j), lon0 + sx * r + rng.uniform(-j, j))
        berths = [(lat0 + rng.uniform(-0.05, 0.05), lon0 + rng.uniform(-0.05, 0.05)) for _ in range(8)]
        anchors = [at(0.15, 0.02) for _ in range(3)]
        hold = at(0.45, 0.03)
        base = (lat0 + 0.03, lon0 + 0.03)
        pool = []
        for t in np.repeat(np.arange(4), np.round(30 * np.array(mix) / sum(mix)).astype(int)):
            typ, vt, lo, hi = MERCHANT[t]
            imo += 1
            tech = rng.choice(["No Scrubber", "Open Loop", "Hybrid", "Closed Loop"],
                              p=[0.7, 0.15, 0.1, 0.05])
            fleet.append({"IMO Number": imo, "Type": typ, "Dwt": int(rng.uniform(lo, hi)),
                          "Built Date": f"01-Jul-{rng.integers(1995, 2021)}",
                          "SOx Scrubber Technology Type": tech,
                          "install_date": None if tech == "No Scrubber" else
                          f"{rng.integers(2016, 2022)}-{rng.integers(1, 13):02d}-01"})
            pool.append([imo, vt, 0.0])
        s = storm_pressure(tracks, lat0, lon0, days)
        queue = pd.Series(s).shift(1).rolling(10, min_periods=1).max().fillna(0.0).to_numpy()
        for d in range(len(days)):
            for _ in range(rng.poisson(rate * (1 - 0.85 * s[d]))):
                free = [v for v in pool if v[2] <= 24 * d]
                if not free:
                    break
                v = free[rng.integers(len(free))]
                t = 24 * d + rng.uniform(0, 24)
                wait = rng.uniform(0, 10) * (1 + 2 * s[d] + 3 * queue[d])
                stay = rng.lognormal(np.log(20), 0.5) * (1 - 0.5 * s[d])
                a = hold if rng.random() < 0.3 + s[d] + 0.3 * queue[d] else anchors[rng.integers(3)]
                b, out = berths[rng.integers(8)], at(0.5)
                legs = [(wait, a, a, 0.1, 1), (max(1.5, 6 * np.hypot(a[0] - b[0], a[1] - b[1])), a, b, 8.0, 0),
                        (stay, b, b, 0.2, 5), (3.0, b, out, 10.0, 0)]
                for hours, x, y, sog, status in legs:
                    n = max(int(hours / STEP_H), 1)
                    for k in range(n):
                        f = (k + 1) / n
                        msgs.append((v[0], v[1], t, x[0] + f * (y[0] - x[0]), x[1] + f * (y[1] - x[1]),
                                     sog, status))
                        t += hours / n
                v[2] = t
        for _ in range(n_tugs):
            imo += 1
            fleet.append({"IMO Number": imo, "Type": "Other Vessel", "Dwt": int(rng.uniform(300, 600)),
                          "Built Date": f"01-Jul-{rng.integers(2000, 2020)}",
                          "SOx Scrubber Technology Type": "No Scrubber", "install_date": None})
            for d in range(len(days)):
                for h in range(0, 24, STEP_H):
                    busy = 6 <= h < 20
                    x = (berths[rng.integers(8)] if busy else base)
                    msgs.append((imo, 52, 24 * d + h + rng.uniform(0, 0.5),
                                 x[0] + rng.uniform(-0.002, 0.002), x[1] + rng.uniform(-0.002, 0.002),
                                 rng.uniform(4, 7) if busy else 0.1, 0 if busy else 5))
    ais = pd.DataFrame(msgs, columns=["IMO_n", "VesselType", "hours", "LAT", "LON", "SOG", "Status"])
    ais["BaseDateTime"] = pd.Timestamp(START) + pd.to_timedelta(ais.pop("hours"), unit="h")
    return ais, pd.DataFrame(fleet)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic demo dataset")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "usace_principal_ports.geojson").write_text(json.dumps(ports_geojson()))
    tracks = storm_tracks()
    units = pd.DataFrame([{c: "" for c in tracks.columns}])
    pd.concat([units, tracks.astype(str)]).to_csv(args.out / "ibtracs.NA.csv", index=False)
    covariates(tracks, rng).to_csv(args.out / "storm_covariates.csv", index=False)

    ais, fleet = simulate(rng, tracks)
    fleet.drop(columns="install_date").to_csv(args.out / "fleet_registry.csv", index=False)
    fleet.dropna(subset=["install_date"]).rename(columns={"IMO Number": "imo"})[
        ["imo", "install_date"]].to_csv(args.out / "scrubber_install_dates.csv", index=False)

    ais = ais[ais["BaseDateTime"].between(START, f"{END} 23:59:59")].sort_values(
        ["BaseDateTime", "IMO_n"])
    ais["MMSI"] = 300_000_000 + ais["IMO_n"] - 9_100_000
    ais["IMO"] = "IMO" + ais["IMO_n"].astype(str)
    ais["BaseDateTime"] = ais["BaseDateTime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for c in ("COG", "Heading", "Length", "Width", "Draft", "Cargo"):
        ais[c] = 0
    ais["VesselName"] = "SHIP" + ais["MMSI"].astype(str)
    ais["CallSign"] = ""
    ais["TransceiverClass"] = "A"
    ais["LAT"], ais["LON"], ais["SOG"] = ais["LAT"].round(5), ais["LON"].round(5), ais["SOG"].round(1)
    for day, g in ais.groupby(ais["BaseDateTime"].str[:10]):
        folder = args.out / "ais" / day[:4]
        folder.mkdir(parents=True, exist_ok=True)
        g[AIS_COLS].to_csv(folder / f"AIS_{day.replace('-', '_')}.csv", index=False)
    print(f"{len(ais):,} AIS messages, {len(fleet)} ships, {len(STORMS)} storms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
