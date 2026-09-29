"""Assign AIS positions to USACE Principal Port polygons.

Polygons are reprojected from Web Mercator on load and indexed on a coarse grid, so a
position in open water resolves in one dictionary lookup.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CELL_DEG = 0.5
WEB_MERCATOR_R = 6378137.0


def webmercator_to_wgs84(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / WEB_MERCATOR_R)
    lat = math.degrees(2 * math.atan(math.exp(y / WEB_MERCATOR_R)) - math.pi / 2)
    return lat, lon


def load_polygons(path: Path, drop_states=("AK",)) -> list[tuple[str, np.ndarray, tuple]]:
    with path.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    out = []
    for feat in gj["features"]:
        name = str(feat["properties"].get("PORTNAME", "")).strip()
        if not name or any(name.endswith(f", {s}") for s in drop_states):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Polygon":
            rings = geom["coordinates"]
        elif geom.get("type") == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        else:
            continue
        for ring in rings:
            pts = np.asarray(ring, dtype=float)
            if pts.shape[0] < 4:
                continue
            if np.abs(pts).max() > 400.0:
                lat, lon = zip(*(webmercator_to_wgs84(x, y) for x, y in pts))
                pts = np.column_stack([lon, lat])
            out.append((name, pts,
                        (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())))
    return out


def grid_index(polys: list) -> dict[tuple[int, int], list[int]]:
    idx: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, (x0, x1, y0, y1)) in enumerate(polys):
        for cx in range(int(np.floor(x0 / CELL_DEG)), int(np.floor(x1 / CELL_DEG)) + 1):
            for cy in range(int(np.floor(y0 / CELL_DEG)), int(np.floor(y1 / CELL_DEG)) + 1):
                idx.setdefault((cx, cy), []).append(i)
    return idx


def _in_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    inside = np.zeros(lon.shape, dtype=bool)
    for a in range(len(x1)):
        cond = (y1[a] > lat) != (y2[a] > lat)
        if not cond.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2[a] - x1[a]) * (lat - y1[a]) / (y2[a] - y1[a]) + x1[a]
        inside ^= cond & (lon < xint)
    return inside


def assign_ports(lon: np.ndarray, lat: np.ndarray, polys: list, idx: dict) -> np.ndarray:
    out = np.full(lon.shape, "", dtype=object)
    if len(lon) == 0:
        return out
    cx = np.floor(lon / CELL_DEG).astype(np.int64)
    cy = np.floor(lat / CELL_DEG).astype(np.int64)
    order = np.lexsort((cy, cx))
    scx, scy = cx[order], cy[order]
    bounds = np.flatnonzero(np.r_[True, (scx[1:] != scx[:-1]) | (scy[1:] != scy[:-1]), True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        cand = idx.get((int(scx[s]), int(scy[s])))
        if not cand:
            continue
        sel = order[s:e]
        plon, plat = lon[sel], lat[sel]
        todo = np.ones(sel.shape, dtype=bool)
        for i in cand:
            name, ring, (x0, x1, y0, y1) = polys[i]
            m = todo & (plon >= x0) & (plon <= x1) & (plat >= y0) & (plat <= y1)
            if not m.any():
                continue
            hit = np.zeros(sel.shape, dtype=bool)
            hit[m] = _in_ring(plon[m], plat[m], ring)
            out[sel[hit]] = name
            todo &= ~hit
            if not todo.any():
                break
    return out


