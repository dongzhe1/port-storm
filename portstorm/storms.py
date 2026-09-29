"""Port-cyclone interactions from IBTrACS best tracks (CyPort's definition).

A port interacts with a cyclone when the eye passes within BUFFER_KM; the exposure period runs
from the first to the last track point inside the buffer.

    python -m portstorm storms --ibtracs ibtracs.NA.csv --ports usace_principal_ports.geojson \
        --out out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .ports import webmercator_to_wgs84

BUFFER_KM = 500.0
EARTH_R_KM = 6371.0

def polygon_centroid(rings: list) -> tuple[float, float]:
    ring = max(rings, key=len)
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring))
    a *= 0.5
    return (cy / (6 * a), cx / (6 * a))


def load_usace_ports(path: Path, drop_states: tuple[str, ...] = ("AK",),
                     types: tuple[str, ...] | None = None) -> dict[str, tuple[float, float]]:
    with path.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    ports: dict[str, tuple[float, float]] = {}
    for feat in gj["features"]:
        prop = feat["properties"]
        name = str(prop.get("PORTNAME", "")).strip()
        if not name or any(name.endswith(f", {s}") for s in drop_states):
            continue
        if types and prop.get("TYPE") not in types:
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Polygon":
            rings = geom["coordinates"]
        elif geom.get("type") == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        else:
            continue
        cy, cx = polygon_centroid(rings)
        if abs(cy) > 90.0 or abs(cx) > 180.0:
            cy, cx = webmercator_to_wgs84(cx, cy)
        ports[name] = (cy, cx)
    if ports:
        lats = [v[0] for v in ports.values()]
        lons = [v[1] for v in ports.values()]
        if not (-90 <= min(lats) and max(lats) <= 90 and -180 <= min(lons) and max(lons) <= 180):
            raise ValueError("port coordinates are not geographic after conversion; "
                             f"lat {min(lats):.1f}..{max(lats):.1f} "
                             f"lon {min(lons):.1f}..{max(lons):.1f}")
    return ports


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_ibtracs(path: Path, season_min: int, season_max: int) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1], low_memory=False,
                     usecols=["SID", "SEASON", "NAME", "ISO_TIME", "NATURE", "LAT", "LON",
                              "USA_WIND", "USA_PRES", "USA_SSHS", "LANDFALL", "DIST2LAND"])
    df = df[~df["SID"].astype(str).str.startswith("SID")]
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df[df["SEASON"].between(season_min, season_max)]
    for c in ("LAT", "LON", "USA_WIND", "USA_PRES", "USA_SSHS", "LANDFALL", "DIST2LAND"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df.dropna(subset=["LAT", "LON", "ISO_TIME"]).sort_values(["SID", "ISO_TIME"])


def main() -> int:
    p = argparse.ArgumentParser(description="Port-cyclone exposure records from IBTrACS")
    p.add_argument("--ibtracs", type=Path, required=True, help="IBTrACS basin CSV")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--season-min", type=int, default=2015)
    p.add_argument("--season-max", type=int, default=2025)
    p.add_argument("--name", default=None, help="restrict to one storm name, e.g. IDA")
    p.add_argument("--season", type=int, default=None, help="restrict to one season")
    p.add_argument("--buffer-km", type=float, default=BUFFER_KM,
                   help=f"impact-zone radius (default {BUFFER_KM:.0f})")
    p.add_argument("--ports", type=Path, required=True, help="USACE Principal Ports GeoJSON")
    p.add_argument("--port-types", default=None,
                   help="comma-separated USACE TYPE filter, e.g. Coastal or Coastal,Internal")
    args = p.parse_args()

    if not args.ports.is_file():
        p.error(f"--ports not found: {args.ports}")
    types = tuple(t.strip() for t in args.port_types.split(",")) if args.port_types else None
    ports = load_usace_ports(args.ports, types=types)
    if not ports:
        p.error("no ports survived the filters")

    if not args.ibtracs.is_file():
        p.error(f"--ibtracs not found: {args.ibtracs}")
    args.out.mkdir(parents=True, exist_ok=True)

    trk = load_ibtracs(args.ibtracs, args.season_min, args.season_max)
    if args.season:
        trk = trk[trk["SEASON"] == args.season]
    if args.name:
        trk = trk[trk["NAME"].astype(str).str.upper() == args.name.upper()]
    if trk.empty:
        p.error("no track points match the filters")

    print(f"IBTrACS : {args.ibtracs.name}")
    print(f"Storms  : {trk['SID'].nunique()} across seasons "
          f"{int(trk['SEASON'].min())}-{int(trk['SEASON'].max())}")
    print(f"Ports   : {len(ports)}   buffer {args.buffer_km:.0f} km\n")

    prox_rows, inter_rows = [], []
    for sid, g in trk.groupby("SID", sort=True):
        name = str(g["NAME"].iloc[0])
        season = int(g["SEASON"].iloc[0])
        storm_peak_sshs = pd.to_numeric(g["USA_SSHS"], errors="coerce").max()
        storm_peak_sshs = -9 if pd.isna(storm_peak_sshs) else storm_peak_sshs
        tlat, tlon = g["LAT"].to_numpy(), g["LON"].to_numpy()
        for port, (plat, plon) in ports.items():
            d = pd.Series(haversine_km(plat, plon, tlat, tlon), index=g.index)
            inside = d <= args.buffer_km
            if not inside.any():
                continue
            sub = g[inside]
            prox_rows.append(pd.DataFrame({
                "SID": sid, "name": name, "season": season, "port": port,
                "time": sub["ISO_TIME"].values, "dist_km": d[inside].round(1).values,
                "wind_kt": sub["USA_WIND"].values, "sshs": sub["USA_SSHS"].values,
            }))
            i = int(d.values.argmin())
            row = g.iloc[i]
            inter_rows.append({
                "SID": sid, "name": name, "season": season, "port": port,
                "exposure_start": sub["ISO_TIME"].min(), "exposure_end": sub["ISO_TIME"].max(),
                "exposure_hours": (sub["ISO_TIME"].max() - sub["ISO_TIME"].min())
                                   .total_seconds() / 3600,
                "min_dist_km": round(float(d.min()), 1),
                "closest_time": row["ISO_TIME"],
                "wind_at_closest_kt": row["USA_WIND"],
                "pres_at_closest_mb": row["USA_PRES"],
                "sshs_at_closest": row["USA_SSHS"],
                "sshs_max_in_window": sub["USA_SSHS"].max(),
                "sshs_storm_lifetime": int(storm_peak_sshs),
                "wind_max_in_window_kt": sub["USA_WIND"].max(),
                "near_land_in_window": int((sub["LANDFALL"].fillna(999) == 0).any()),
                "landfall_in_window": int((sub["DIST2LAND"].fillna(999) == 0).any()),
            })

    if not inter_rows:
        print("\n! No port fell within the buffer of any storm in this range.")
        print(f"  {len(ports)} port(s), {trk['SID'].nunique()} storm(s), "
              f"buffer {args.buffer_km:.0f} km")
        pd.DataFrame(columns=["SID", "name", "season", "port"]).to_csv(
            args.out / "storm_exposure.csv", index=False)
        return 1
    inter = pd.DataFrame(inter_rows).sort_values(["season", "SID", "port"])
    prox = pd.concat(prox_rows, ignore_index=True) if prox_rows else pd.DataFrame()
    ip, pp = args.out / "storm_exposure.csv", args.out / "storm_track_6h.csv"
    inter.to_csv(ip, index=False)
    prox.to_csv(pp, index=False)

    print(f"  {len(inter):,} port-cyclone interactions -> {ip}")
    print(f"  {len(prox):,} proximity records          -> {pp}")
    if len(inter):
        print("\nStorms touching the most ports:")
        top = (inter.groupby(["season", "name"]).agg(
            ports=("port", "nunique"), closest=("min_dist_km", "min"),
            max_cat=("sshs_max_in_window", "max")).sort_values("ports", ascending=False))
        print(top.head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
