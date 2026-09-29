"""Step 10 - port reference points and cyclone positions for the map (Supplementary Fig. 1).
The upstream tables give the distance from the eye to each port within 500 km every 3 hours (storm_track_6h.csv),
but no coordinates. We start from public port locations, place each cyclone fix by least squares on its distances to
those ports (penalising positions within 500 km of a port not listed at that time), choose among candidate solutions
by track continuity, re-estimate the port points from the fixes, and iterate.
Writes port_locations.csv and storm_fixes.csv.
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import panel

OUT = panel.OUT
CORE = panel.CORE
PORTS0 = {
    "Albany Port District, NY": (42.63, -73.76), "Baltimore, MD": (39.26, -76.58), "Beaumont, TX": (30.08, -94.09),
    "Boston, MA": (42.35, -71.04), "Bridgeport, CT": (41.17, -73.18), "Brownsville, TX": (25.95, -97.40),
    "Calhoun Port Authority, TX": (28.65, -96.56), "Canaveral Port District, FL": (28.41, -80.61),
    "Corpus Christi, TX": (27.81, -97.40), "Galveston, TX": (29.31, -94.79), "Greater Lafourche Port, LA": (29.11, -90.20),
    "Green Bay, WI": (44.53, -88.01), "Guayama, PR": (17.93, -66.16), "Guaynabo, PR": (18.43, -66.12),
    "Houston Port Authority, TX": (29.72, -95.15), "Illinois International Port, IL": (41.72, -87.54),
    "Jacksonville, FL": (30.40, -81.55), "Lake Charles Harbor District, LA": (30.22, -93.25),
    "Manatee County Port, FL": (27.64, -82.56), "Marquette, MI": (46.54, -87.39), "Milwaukee, WI": (43.02, -87.90),
    "Mobile, AL": (30.70, -88.04), "Morehead City, NC": (34.72, -76.70), "Mueller Township, MI": (45.96, -85.87),
    "New Haven, CT": (41.29, -72.91), "New Orleans, LA": (29.93, -90.06), "New York, NY & NJ": (40.66, -74.08),
    "Orange County Nav District, TX": (30.09, -93.73), "Panama City Port Authority, FL": (30.18, -85.73),
    "Philadelphia Regional Port, PA": (39.90, -75.14), "Plaquemines Port District, LA": (29.50, -89.75),
    "Port Arthur, TX": (29.87, -93.93), "Port Everglades, FL": (26.09, -80.12), "Port Freeport, TX": (28.94, -95.31),
    "Port Jefferson, NY": (40.95, -73.07), "Port of Brunswick, GA": (31.15, -81.50), "Port of Charleston, SC": (32.80, -79.93),
    "Port of Greater Baton Rouge, LA": (30.44, -91.20), "Port of Iberia District, LA": (29.98, -91.84),
    "Port of Palm Beach District, FL": (26.77, -80.05), "Port of Pascagoula, MS": (30.35, -88.56),
    "Port of Providence, RI": (41.80, -71.39), "Port of Savannah, GA": (32.12, -81.14), "PortMiami, FL": (25.77, -80.17),
    "Portland, ME": (43.65, -70.25), "Sabine Pass Port Authority, TX": (29.73, -93.87), "San Juan, PR": (18.45, -66.10),
    "Searsport, ME": (44.45, -68.92), "South Jersey Port Corp, NJ": (39.90, -75.18),
    "South Louisiana, LA, Port of": (30.05, -90.55), "Tampa Port Authority, FL": (27.93, -82.43),
    "Terrebonne Parish Port, LA": (29.58, -90.72), "Texas City, TX": (29.37, -94.89),
    "Virgin Islands - St. Croix, VI": (17.70, -64.75), "Virginia, VA, Port of": (36.90, -76.33), "Wilmington, NC": (34.20, -77.95)}
R_EARTH = 6371.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fit_point(plat, plon, d, start, far_lat=None, far_lon=None):
    def res(x):
        r = hav(x[0], x[1], plat, plon) - d
        if far_lat is not None and len(far_lat):
            r = np.concatenate([r, 0.5 * np.clip(500 - hav(x[0], x[1], far_lat, far_lon), 0, None)])
        return r
    s = least_squares(res, start, method="lm" if len(d) >= 2 else "trf")
    return s.x, float(np.sqrt(np.mean(res(s.x)[:len(d)] ** 2))), float(np.sum(res(s.x) ** 2))


def locate_storms(T, P):
    """one fix per (SID, time) with >= 3 known ports; candidates from a grid of starts, then continuity."""
    names = np.array(list(P)); plat = np.array([P[n][0] for n in names]); plon = np.array([P[n][1] for n in names])
    rows = []
    for sid, g in T.groupby("SID"):
        cands = []
        times = sorted(g.time.unique())
        for t in times:
            x = g[g.time == t]
            m = np.isin(names, x.port.values)
            if m.sum() < 3:
                cands.append([]); continue
            d = x.set_index("port").dist_km.reindex(names[m]).values
            far = ~m
            c0 = np.array([plat[m].mean(), plon[m].mean()])
            sols = []
            for dy in (-6, -3, 0, 3, 6):
                for dx in (-6, -3, 0, 3, 6):
                    xy, rms, cost = fit_point(plat[m], plon[m], d, c0 + [dy, dx], plat[far], plon[far])
                    if not any(hav(xy[0], xy[1], s[0], s[1]) < 30 for s in sols):
                        sols.append((xy[0], xy[1], rms, cost, int(m.sum())))
            sols = sorted(sols, key=lambda s: s[3])[:3]
            cands.append(sols)
        # dynamic programming: cost = fit cost + jump penalty (km beyond 150 per 3 h)
        best, prev = [], None
        for i, sols in enumerate(cands):
            if not sols:
                best.append(None); prev = None; continue
            if prev is None:
                pick = min(sols, key=lambda s: s[3])
            else:
                pick = min(sols, key=lambda s: s[3] + 50 * max(hav(s[0], s[1], prev[0], prev[1]) - 150, 0))
            best.append(pick); prev = pick
        for t, b in zip(times, best):
            if b is not None:
                rows.append(dict(SID=sid, time=t, lat=b[0], lon=b[1], rms_km=b[2], n_ports=b[4]))
    return pd.DataFrame(rows)


def refine_ports(T, S, P):
    M = T.merge(S, on=["SID", "time"])
    out = {}
    for port, x in M.groupby("port"):
        if port not in P or len(x) < 5:
            continue
        good = x[x.rms_km < 40]
        if len(good) < 5:
            out[port] = P[port]; continue
        xy, rms, _ = fit_point(good.lat.values, good.lon.values, good.dist_km.values, np.array(P[port]))
        out[port] = (float(xy[0]), float(xy[1])) if hav(xy[0], xy[1], *P[port]) < 150 else P[port]
    return {**P, **out}


if __name__ == "__main__":
    T = pd.read_csv(CORE + "storm_track_6h.csv")
    T = T[T.dist_km < 500]
    P = dict(PORTS0)
    for it in range(3):
        S = locate_storms(T[T.port.isin(P)], P)
        P = refine_ports(T[T.port.isin(P)], S, P)
        M = T[T.port.isin(P)].merge(S, on=["SID", "time"])
        M["pred"] = hav(M.lat.values, M.lon.values, np.array([P[p][0] for p in M.port]), np.array([P[p][1] for p in M.port]))
        M["err"] = M.pred - M.dist_km
        print(f"iteration {it + 1}: fixes {len(S)}, median |error| {M.err.abs().median():.1f} km, 90th pct {M.err.abs().quantile(.9):.1f} km", flush=True)
    S = locate_storms(T[T.port.isin(P)], P)
    M = T[T.port.isin(P)].merge(S, on=["SID", "time"])
    M["pred"] = hav(M.lat.values, M.lon.values, np.array([P[p][0] for p in M.port]), np.array([P[p][1] for p in M.port]))
    M["err"] = M.pred - M.dist_km
    L = pd.DataFrame([dict(port=p, lat=v[0], lon=v[1], lat0=PORTS0[p][0], lon0=PORTS0[p][1],
                           moved_km=hav(v[0], v[1], *PORTS0[p])) for p, v in P.items()])
    e = M.groupby("port").err.agg(rms_km=lambda s: float(np.sqrt(np.mean(s ** 2))), n_fixes="count")
    L = L.merge(e, left_on="port", right_index=True, how="left")
    L.to_csv(os.path.join(OUT, "port_locations.csv"), index=False)
    S = S.merge(T.groupby("SID")[["name", "season"]].first(), left_on="SID", right_index=True)
    S.to_csv(os.path.join(OUT, "storm_fixes.csv"), index=False)
    pd.set_option("display.width", 200)
    print(L.sort_values("moved_km", ascending=False).round(2).to_string(index=False))
