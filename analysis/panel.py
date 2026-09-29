"""Core of the national storm panel: every US port complex x tropical cyclone pair, 2015-2023,
scored against a seasonal normal built from storm-free baseline years.

Imported by every analysis script; 01_score_events.py runs the main scoring and 06_robustness.py the placebo dates
and alternative normals.

Paths (environment variables):
  HURR_DATA  pipeline output folder, with the tier panels in tiers/ (default: current folder)
  HURR_OUT   output folder for intermediate tables (default: ./out); figures go to HURR_OUT/figures

Method (default options, DEFAULT below)
  Events    storm_exposure.csv pairs with eye <= 200 km of the port and >= 34 kt while exposed.
  Windows   storm = exposure period in whole UTC days; prep = 2 days before; after = days 1..10 after.
  Baseline  other years 2015-2023 with no storm within 250 km of that port from 60 days before to
            45 days after the event's calendar window (needs >= 3 years).
  Normal    each year indexed to its own mean over [start-45, start-11]; seasonal index = median
            over baseline years of days d-3..d+3; expected = event-year level x index.
  Series    tier (port=berth, channel, anchorage, offshore_hold, offshore=open water, total)
            x vessel class (tanker, cargo, other, all) x {vessel-hours, CO2, fuel main/aux/boiler};
            tier sulfur (SO2 to air, to wash water), distinct vessels in cell;
            port polygon: port calls, vessels present (from port_daily_calls.csv).
  Scored    complexes emitting >= 50 t CO2/day in the reference window.
"""
import os, pickle, time
import numpy as np, pandas as pd

ROOT = os.environ.get("HURR_DATA", ".")
OUT = os.environ.get("HURR_OUT", "out")
os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
CORE = os.path.join(ROOT, "")
TIER = os.path.join(ROOT, "tiers/")
IDX = pd.date_range("2015-01-01", "2023-12-31")
MEAS = ["hours", "co2_ton", "fc_main_ton", "fc_AE_ton", "fc_boiler_ton"]
TIERS = ["port", "channel", "anchorage", "offshore_hold", "offshore"]
DEFAULT = dict(max_dist=200, min_wind=34, min_co2=50, min_base=3, excl_km=250,
               ref=(45, 11), pool=3, stat="median", pre=60, post=45, normal="seasonal")
# Study sample: Gulf and Atlantic port areas. Caribbean and Great Lakes ports are excluded (10 events in the full panel,
# all tropical storms or remnants, too few to estimate anything for these regions).
EXCLUDE_PORTS = {
    "Guayama, PR", "Guaynabo, PR", "San Juan, PR", "Virgin Islands - St. Croix, VI",                              # Caribbean
    "Green Bay, WI", "Illinois International Port, IL", "Marquette, MI", "Milwaukee, WI", "Mueller Township, MI"}  # Great Lakes
INT_BINS, INT_LABELS = [0, 49, 63, 200], ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]


def load(cache=True):
    """read package files once; cached as a pickle in HURR_OUT."""
    f = os.path.join(OUT, "_panel_inputs.pkl")
    if cache and os.path.exists(f):
        return pickle.load(open(f, "rb"))
    t0 = time.time()
    ex = pd.read_csv(CORE + "storm_exposure.csv", parse_dates=["exposure_start", "exposure_end"])
    cov = pd.read_csv(CORE + "storm_covariates.csv")
    ports = ex.port.unique()
    z = pd.read_csv(TIER + "emissions_by_zone.csv",
                    usecols=["date", "complex", "tier", "vclass"] + MEAS,
                    dtype={"complex": "category", "tier": "category", "vclass": "category"}, parse_dates=["date"])
    z = z[z.complex.isin(ports) & (z.tier != "outside")]
    s = pd.read_csv(TIER + "tier_daily_sulfur.csv",
                    usecols=["date", "complex", "tier", "n_vessels_in_cell", "so2_air_ton", "so2_washwater_ton"],
                    dtype={"complex": "category", "tier": "category"}, parse_dates=["date"])
    s = s[s.complex.isin(ports) & (s.tier != "outside")]
    calls = pd.read_csv(CORE + "port_daily_calls.csv", parse_dates=["date"])
    calls = calls[calls.port.isin(ports)]
    D = dict(ex=ex, cov=cov, z=z, s=s, calls=calls)
    if cache:
        pickle.dump(D, open(f, "wb"))
    print(f"inputs loaded in {time.time() - t0:.0f} s")
    return D


def complex_frame(c, D):
    q = D["z"][D["z"].complex == c]
    a = q.pivot_table(index="date", columns=["vclass", "tier"], values=MEAS, aggfunc="sum", observed=True)
    a.columns = [f"{v}|{t}|{m}" for m, v, t in a.columns]
    a = a.reindex(IDX, fill_value=0.0).fillna(0.0)
    cols = {}
    for col in a.columns:
        v, t, m = col.split("|")
        for key in {f"{v}|{t}|{m}", f"{v}|total|{m}", f"all|{t}|{m}", f"all|total|{m}"}:
            cols[key] = cols.get(key, 0.0) + a[col]
    out = pd.DataFrame(cols)
    q = D["s"][D["s"].complex == c]
    b = q.pivot_table(index="date", columns="tier", values=["n_vessels_in_cell", "so2_air_ton", "so2_washwater_ton"],
                      aggfunc="sum", observed=True)
    b.columns = [f"all|{t}|{m}" for m, t in b.columns]
    b = b.reindex(IDX, fill_value=0.0).fillna(0.0)
    for m in ["so2_air_ton", "so2_washwater_ton"]:
        b[f"all|total|{m}"] = b[[x for x in b.columns if x.endswith("|" + m)]].sum(axis=1)
    k = D["calls"][D["calls"].port == c].set_index("date")
    cl = pd.DataFrame({"all|port|port_calls": k.port_calls, "all|port|vessels_present": k.vessels_present}).reindex(IDX).fillna(0.0)
    return pd.concat([out, b, cl], axis=1)


def events(D, o):
    ex, cov = D["ex"], D["cov"]
    ev = ex[(ex.min_dist_km <= o["max_dist"]) & (ex.wind_max_in_window_kt >= o["min_wind"])].copy()
    ev = ev.merge(cov[["SID", "port", "surge_height_m", "max_wind_ms", "rainfall_mm"]], on=["SID", "port"], how="left")
    ev["start"] = ev.exposure_start.dt.normalize(); ev["end"] = ev.exposure_end.dt.normalize()
    return ev[~ev.port.isin(EXCLUDE_PORTS)]


def md_shift(ts, y):
    try:
        return ts.replace(year=y)
    except ValueError:
        return None


def baseline_years(D, port, lo_d, hi_d, Y, excl_km):
    other = D["ex"][(D["ex"].port == port) & (D["ex"].min_dist_km <= excl_km)]
    base = []
    for y in range(2015, 2024):
        if y == Y:
            continue
        a, b = md_shift(lo_d, y), md_shift(hi_d, y)
        if a is None or b is None:
            continue
        if other[(other.exposure_end >= a) & (other.exposure_start <= b)].empty:
            base.append(y)
    return base


def score(F, start, end, Y, base, o, cols=None, daily_cols=None):
    """score one (possibly placebo) event. start/end are dates in year Y. Returns (rows, daily frame)."""
    F = F if cols is None else F[[c for c in cols if c in F.columns]]
    lo_d, hi_d = start - pd.Timedelta(days=o["pre"]), end + pd.Timedelta(days=o["post"])
    ref_a, ref_b = start - pd.Timedelta(days=o["ref"][0]), start - pd.Timedelta(days=o["ref"][1])
    days = pd.date_range(lo_d, hi_d); nd = len(days); names = F.columns

    def block(y):
        a = md_shift(lo_d, y); ra, rb = md_shift(ref_a, y), md_shift(ref_b, y)
        v = F.loc[a:a + pd.Timedelta(days=nd - 1)].values
        lvl = F.loc[ra:rb].values.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return v / lvl, lvl

    obs = F.loc[days[0]:days[-1]].values
    if len(obs) != nd:
        return [], None
    arr = np.stack([block(y)[0] for y in base])                       # years x days x cols
    _, lvlY = block(Y)
    p = o["pool"]
    pad = np.pad(arr, ((0, 0), (p, p), (0, 0)), mode="edge")
    win = np.stack([pad[:, i:i + nd, :] for i in range(2 * p + 1)], axis=1).reshape(-1, nd, len(names))
    win = np.where(np.isfinite(win), win, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(win, axis=0) if o["stat"] == "median" else np.nanmean(win, axis=0)
    exp = lvlY * med
    if o["normal"] == "same_year":        # package-style: flat level from same-year blocks before and after
        post_a, post_b = end + pd.Timedelta(days=o["ref"][1]), end + pd.Timedelta(days=o["ref"][0])
        lv = np.concatenate([F.loc[ref_a:ref_b].values, F.loc[post_a:post_b].values]).mean(axis=0)
        exp = np.tile(lv, (nd, 1)); med = np.ones_like(exp)
    ph = {"prep": (start - pd.Timedelta(days=2), start - pd.Timedelta(days=1)), "storm": (start, end),
          "after": (end + pd.Timedelta(days=1), end + pd.Timedelta(days=10))}
    rows = []
    for name, (a, b) in ph.items():
        i0, i1 = days.get_loc(a), days.get_loc(b)
        ob = obs[i0:i1 + 1].sum(axis=0); x = exp[i0:i1 + 1].sum(axis=0)
        with np.errstate(all="ignore"):
            ratio = ob / x
            others = arr[:, i0:i1 + 1, :].mean(axis=1) / med[i0:i1 + 1].mean(axis=0)
        outside = (ratio < np.nanmin(others, axis=0)) | (ratio > np.nanmax(others, axis=0))
        for j, col in enumerate(names):
            if not np.isfinite(x[j]) or x[j] <= 0:
                continue
            v, t, m = col.split("|")
            rows.append(dict(phase=name, vclass=v, tier=t, measure=m, obs=ob[j], expected=x[j], delta=ob[j] - x[j],
                             ratio=ratio[j], outside=bool(outside[j]), level=lvlY[j]))
    daily = None
    if daily_cols is not None:
        keep = [j for j, c in enumerate(names) if c in daily_cols]
        rel = (days - start).days
        sel = (rel >= -2) & (rel <= (end - start).days + o["post"])
        daily = pd.concat([pd.DataFrame(obs[sel][:, keep], columns=["obs:" + names[j] for j in keep]),
                           pd.DataFrame(exp[sel][:, keep], columns=["exp:" + names[j] for j in keep])], axis=1)
        daily["rel_day"] = rel[sel]
    return rows, daily


def run(D, opts=None, cols=None, daily_cols=None, placebo=False, verbose=True):
    """score all pairs. placebo=True: each baseline year in turn plays the event year (same calendar
    dates), scored against the remaining baseline years."""
    o = DEFAULT | (opts or {})
    ev = events(D, o)
    frames, rows, daily, t0 = {}, [], [], time.time()
    for _, e in ev.iterrows():
        c, Y = e.port, e.start.year
        if e.start.month == 1 or (e.end.month == 12 and e.end.day > 10):
            continue
        if c not in frames:
            frames[c] = complex_frame(c, D)
        F = frames[c]
        lo_d, hi_d = e.start - pd.Timedelta(days=o["pre"]), e.end + pd.Timedelta(days=o["post"])
        base = baseline_years(D, c, lo_d, hi_d, Y, o["excl_km"])
        ref_a, ref_b = e.start - pd.Timedelta(days=o["ref"][0]), e.start - pd.Timedelta(days=o["ref"][1])
        if "all|total|co2_ton" not in F or F["all|total|co2_ton"][ref_a:ref_b].mean() < o["min_co2"] or len(base) < o["min_base"]:
            continue
        meta = dict(SID=e.SID, storm=e["name"], season=Y, port=c, min_dist_km=e.min_dist_km,
                    sshs=e.sshs_max_in_window, wind_at_closest_kt=e.wind_at_closest_kt, landfall=e.landfall_in_window,
                    surge_m=e.surge_height_m, rain_mm=e.rainfall_mm, k_days=(e.end - e.start).days + 1, n_base=len(base))
        todo = [(Y, e.start, e.end, base, None)]
        if placebo:
            todo = []
            for y in base:
                s_, e_ = md_shift(e.start, y), md_shift(e.end, y)
                rest = [b for b in base if b != y]
                if s_ is not None and e_ is not None and len(rest) >= o["min_base"]:
                    todo.append((y, s_, e_, rest, y))
        for (yy, s_, e_, bb, fake) in todo:
            r, d = score(F, s_, e_, yy, bb, o, cols=cols, daily_cols=daily_cols)
            m2 = meta | dict(placebo_year=fake, n_base=len(bb))
            rows += [m2 | x for x in r]
            if d is not None:
                d["SID"] = e.SID; d["port"] = c; d["k_days"] = meta["k_days"]; d["placebo_year"] = fake
                daily.append(d)
    R = pd.DataFrame(rows)
    if len(R):
        R["intensity"] = pd.cut(R.wind_at_closest_kt, INT_BINS, labels=INT_LABELS)
    if verbose:
        n = R[["SID", "port", "placebo_year"]].drop_duplicates().shape[0] if len(R) else 0
        print(f"scored {n} pair-years in {time.time() - t0:.0f} s  opts={opts or {}}  placebo={placebo}")
    return R, (pd.concat(daily, ignore_index=True) if daily else None)


def boot_ratio(df, num, den, n=1000, seed=1, by="SID"):
    """storm-clustered bootstrap 95% CI for sum(num)/sum(den)."""
    rng = np.random.default_rng(seed)
    g = df.groupby(by)[[num, den]].sum().values
    if len(g) < 2:
        return np.nan, np.nan
    s = g[rng.integers(0, len(g), size=(n, len(g)))].sum(axis=1)
    with np.errstate(all="ignore"):
        return tuple(np.nanpercentile(s[:, 0] / s[:, 1], [2.5, 97.5]))


# ---------------------------------------------------------------- reconciling class normals (step 01b)
CLS4 = ["tanker", "cargo", "passenger", "other"]
REC_BAND = (1.0, 1.0)      # (1, 1) = rescale every cell (full anchoring to the all-ship zone normal)


def class_factors(R):
    """Factors per pair-year x zone that rescale the four class normals to the all-ship normal, over the
    two days before to ten days after (prep + storm + after): 'factor' = all-ship normal CO2 / sum of class
    normal CO2 (applied to class CO2 and fuel), 'factor_h' = the same for ship-hours (applied to class hours).
    Each class is scored against its own baseline, and in thinly used zones a class can have a near-zero
    baseline level, which distorts its normal (e.g. tankers in Mobile's offshore holding area around Zeta,
    2020); summed over classes the normals also run a few percent low. With REC_BAND = (1, 1) every cell is
    rescaled, so the class split only sets the mix (by class and engine) and the level always comes from the
    all-ship normal of that zone."""
    k = ["SID", "port", "placebo_year", "tier"]
    out = []
    for meas, name in [("co2_ton", "factor"), ("hours", "factor_h")]:
        c = R[(R.measure == meas)].copy()
        c["placebo_year"] = c.placebo_year.fillna(0).astype(int)
        a = c[c.vclass == "all"].groupby(k).expected.sum()
        s_ = c[c.vclass.isin(CLS4)].groupby(k).expected.sum()
        f = (a / s_).replace([np.inf, -np.inf], np.nan)
        out.append(f.where((f < REC_BAND[0]) | (f > REC_BAND[1])).rename(name))
    F = pd.concat(out, axis=1).reset_index()
    return F[F[["factor", "factor_h"]].notna().any(axis=1)]


def _fac(F, keys, col):
    return F.set_index(["SID", "port", "placebo_year", "tier"])[col].reindex(keys).fillna(1.0).values


def apply_factors_results(R, F):
    """scale class-level expected values (and ratio, delta) in a window-level results frame."""
    R = R.copy(); py = R.placebo_year.fillna(0).astype(int)
    key = pd.MultiIndex.from_arrays([R.SID, R.port, py, R.tier])
    fac = np.where(R.measure.values == "hours", _fac(F, key, "factor_h"), _fac(F, key, "factor"))
    fac = np.where(R.vclass.isin(CLS4).values & R.measure.isin(["hours", "co2_ton", "fc_main_ton", "fc_AE_ton", "fc_boiler_ton"]).values, fac, 1.0)
    R["expected"] = R.expected * fac
    R["delta"] = R.obs - R.expected
    with np.errstate(all="ignore"):
        R["ratio"] = R.obs / R.expected
    return R


def apply_factors_daily(D, F):
    """scale class-level exp: columns of a daily frame by the pair-year x zone factors."""
    D = D.copy(); py = D.placebo_year.fillna(0).astype(int)
    key = D.SID.astype(str) + "|" + D.port.astype(str) + "|" + py.astype(str)
    for (sid, port, y), g in F.groupby(["SID", "port", "placebo_year"]):
        m = (key == f"{sid}|{port}|{y}").values
        if not m.any():
            continue
        for _, r in g.iterrows():
            for c in [c for c in D.columns if c.startswith("exp:") and c.split(":")[1].split("|")[0] in CLS4 and c.split("|")[1] == r.tier]:
                meas = c.split("|")[2]
                f = r.factor_h if meas == "hours" else (r.factor if meas in ("co2_ton", "fc_main_ton", "fc_AE_ton", "fc_boiler_ton") else np.nan)
                if np.isfinite(f):
                    D.loc[m, c] = D.loc[m, c] * f
    return D


# ---------------------------------------------------------------- window sums with anchored pollutants
EF_POLL = {"nox": {"main_big": 91.9, "main_small": 62.9, "aux": 52.6, "boiler": 6.6},        # g per kg fuel
           "pm25": {"main_big": 0.995, "main_small": 0.898, "aux": 0.856, "boiler": 0.302}}
ENG = [("fc_main_ton", "main"), ("fc_AE_ton", "aux"), ("fc_boiler_ton", "boiler")]
WINDOWS = {"prep": lambda k: (-2, -1), "storm": lambda k: (0, k - 1), "after": lambda k: (k, k + 9),
           "event30": lambda k: (-2, k + 29)}


def class_zone(x, kind, tiers):
    """sums over classes, engines and zones of a daily slice: CO2 and fuel-based NOx, PM2.5 (t)."""
    out = {"co2cz": 0.0, "nox": 0.0, "pm25": 0.0}
    for c in CLS4:
        big = c in ("tanker", "cargo")
        for t in tiers:
            col = f"{kind}:{c}|{t}|co2_ton"
            if col in x:
                out["co2cz"] += x[col].fillna(0).sum()
            for m, e in ENG:
                col = f"{kind}:{c}|{t}|{m}"
                if col not in x:
                    continue
                v = x[col].fillna(0).sum()
                ek = ("main_big" if big else "main_small") if e == "main" else e
                for p in EF_POLL:
                    out[p] += v * EF_POLL[p][ek] / 1000
    return out


def window_sums(D, windows=None):
    """D: daily frame -> one row per pair-year x window x zone (each zone and the port-area total) with
    observed and normal ship-hours, CO2, NOx and PM2.5. Ship-hours and CO2 use the all-ship series of that
    zone (or of the port area). The pollutant normal is anchored to it: normal NOx = normal CO2 x (normal
    NOx / normal CO2 from the class-by-engine normals), so all four measures share one baseline and the
    pollutant ratio differs from the CO2 ratio only through the change in NOx (PM2.5) per tonne of CO2."""
    windows = windows or WINDOWS
    rows = []
    D = D.copy(); D["placebo_year"] = D.placebo_year.fillna(0).astype(int)
    for (sid, port, py), d in D.groupby(["SID", "port", "placebo_year"]):
        k = int(d.k_days.iloc[0]); d = d.set_index("rel_day")
        for w, f in windows.items():
            a, b = f(k); x = d.loc[a:b]
            for z in TIERS + ["total"]:
                zz = TIERS if z == "total" else [z]
                r = dict(SID=sid, port=port, placebo_year=py, window=w, zone=z)
                for kind in ("obs", "exp"):
                    r[f"{kind}_hours"] = x.get(f"{kind}:all|{z}|hours", pd.Series(0.0)).sum()
                    r[f"{kind}_co2"] = x.get(f"{kind}:all|{z}|co2_ton", pd.Series(0.0)).sum()
                    cz = class_zone(x, kind, zz)
                    r[f"{kind}_co2cz"] = cz["co2cz"]
                    for p in EF_POLL:
                        r[f"{kind}_{p}_cz"] = cz[p]
                for p in EF_POLL:
                    r[f"obs_{p}"] = r[f"obs_{p}_cz"]
                    r[f"exp_{p}"] = r["exp_co2"] * r[f"exp_{p}_cz"] / r["exp_co2cz"] if r["exp_co2cz"] > 0 else np.nan
                rows.append(r)
    return pd.DataFrame(rows)
