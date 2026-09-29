"""Step 8 - event-level classification (Methods, 'Event-level classification and determinants'; Fig. 3a,b).
For every event, window (storm window; ten days after; whole event) and zone:
  lh = ln(observed / normal ship-hours); lc = ln(observed / normal CO2) (also NOx);
  g = lc - lh, the divergence gap D.
Noise bands come from placebo dates and are scaled to the zone's normal size and the window length by quantile
regression. Classes (code names in brackets): no detectable change (no response); aligned response (coupled);
CO2-dominant divergence (relative (amplified)); traffic-dominant divergence (relative (damped)); the absolute class
is defined but did not occur during the storm window.
Writes decoupling_events.csv, decoupling_placebo.csv, decoupling_placebo_classified.csv, decoupling_thresholds.csv,
decoupling_counts.csv, decoupling_zone_counts.csv and pollutant_intensity_test.csv (NOx and PM2.5 per tonne of CO2).
"""
import os
import numpy as np, pandas as pd
import panel

OUT = panel.OUT
CLS = ["tanker", "cargo", "passenger", "other"]
TIERS = panel.TIERS
ZONES = TIERS + ["total"]
ENG = [("fc_main_ton", "main"), ("fc_AE_ton", "aux"), ("fc_boiler_ton", "boiler")]
EF_NOX = {"main_big": 91.9, "main_small": 62.9, "aux": 52.6, "boiler": 6.6}   # g/kg fuel, as 4.8
WINDOWS = ["storm", "after", "event30"]


def window_sums(D):
    """panel.window_sums (anchored pollutants) plus the log responses and gaps."""
    S = panel.window_sums(D, {w: panel.WINDOWS[w] for w in WINDOWS})
    with np.errstate(all="ignore"):
        for m in ("hours", "co2", "nox"):
            S["l" + m[0] if m != "nox" else "ln"] = np.log(S[f"obs_{m}"] / S[f"exp_{m}"])
    S["g"] = S.lc - S.lh
    S["g_nox"] = S.ln - S.lh
    return S.replace([np.inf, -np.inf], np.nan)


def add_days(S, P):
    S = S.merge(P[["SID", "port", "k_days"]], on=["SID", "port"], how="left")
    S["days"] = np.select([S.window == "storm", S.window == "after", S.window == "prep"], [S.k_days, 10, 2], S.k_days + 32)
    with np.errstate(all="ignore"):
        S["size"] = np.log(S.exp_co2 / S.days)                 # normal CO2 per day in that zone (log t)
        S["ld"] = np.log(S.days)
        S["hours_day"] = S.exp_hours / S.days
    return S


MEASURES = {"lh": "tau_h", "lc": "tau_c", "ln": "tau_n", "g": "tau_g", "g_nox": "tau_gn"}
MIN_HOURS_DAY = 24          # a zone needs >= 1 ship on average in the normal to be classified


def fit_bands(PL, q=0.95):
    """noise band = q-th quantile of |x| over placebo dates, scaled to zone size and window length
    (quantile regression |x| ~ log size + log days, per window x zone x measure)."""
    import statsmodels.formula.api as smf
    fits = {}
    for (w, z), d in PL[PL.hours_day >= MIN_HOURS_DAY].groupby(["window", "zone"]):
        for m in MEASURES:
            x = d.assign(a=d[m].abs())[["a", "size", "ld"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 50:
                continue
            fits[(w, z, m)] = smf.quantreg("a ~ size + ld", x).fit(q=q).params
    return fits


def apply_bands(S, fits):
    S = S.copy()
    for m, tcol in MEASURES.items():
        S[tcol] = np.nan
        for (w, z, mm), p in fits.items():
            if mm != m:
                continue
            i = (S.window == w) & (S.zone == z)
            S.loc[i, tcol] = np.maximum(p["Intercept"] + p["size"] * S.loc[i, "size"] + p["ld"] * S.loc[i, "ld"], 0.02)
    return S


def classify(S, env="lc", gap="g"):
    tc, tg = ("tau_c", "tau_g") if env == "lc" else ("tau_n", "tau_gn")
    th, tcv, tgv = S.tau_h.values, S[tc].values, S[tg].values
    lh, le, g = S.lh.values, S[env].values, S[gap].values
    out = np.full(len(S), "no data", dtype=object)
    ok = np.isfinite(lh) & np.isfinite(le) & np.isfinite(th) & (S.hours_day.values >= MIN_HOURS_DAY)
    nores = ok & (np.abs(lh) <= th) & (np.abs(le) <= tcv)
    coup = ok & ~nores & (np.abs(g) <= tgv)
    dec = ok & ~nores & ~coup
    absd = dec & (np.abs(lh) > th) & (np.abs(le) > tcv) & (np.sign(lh) != np.sign(le))
    rel = dec & ~absd
    out[nores] = "no response"; out[coup] = "coupled"
    out[rel & (np.abs(le) > np.abs(lh))] = "relative (amplified)"
    out[rel & (np.abs(le) <= np.abs(lh))] = "relative (damped)"
    out[absd] = "absolute"
    return out


ORDER = ["no response", "coupled", "relative (amplified)", "relative (damped)", "absolute", "no data"]

if __name__ == "__main__":
    P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
    fe = os.path.join(OUT, "decoupling_events_raw.csv")
    if os.path.exists(fe):
        E = pd.read_csv(fe)
    else:
        Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz"))
        E = window_sums(Dd[Dd.placebo_year.isna()]); E.to_csv(fe, index=False)
    fp = os.path.join(OUT, "decoupling_placebo.csv")
    if os.path.exists(fp):
        PL = pd.read_csv(fp)
    else:
        cols = {f"{v}|{t}|{m}" for v in CLS + ["all"] for t in ZONES for m in panel.MEAS}
        ND = panel.load()
        Rp, Dp = panel.run(ND, cols=cols, daily_cols=cols, placebo=True)
        Dp = panel.apply_factors_daily(Dp, panel.class_factors(Rp))     # same reconciliation as 01b
        PL = window_sums(Dp); PL.to_csv(fp, index=False)
    E, PL = add_days(E, P), add_days(PL, P)
    fits = fit_bands(PL)
    pd.DataFrame([dict(window=w, zone=z, measure=m, **p.to_dict()) for (w, z, m), p in fits.items()]).to_csv(
        os.path.join(OUT, "decoupling_thresholds.csv"), index=False)
    E, PL = apply_bands(E, fits), apply_bands(PL, fits)
    E["cls_co2"] = classify(E); E["cls_nox"] = classify(E, env="ln", gap="g_nox")
    PL["cls_co2"] = classify(PL)
    E = E.merge(P.drop(columns=["k_days"]), on=["SID", "port"], how="left")
    E.to_csv(os.path.join(OUT, "decoupling_events.csv"), index=False)
    PL.to_csv(os.path.join(OUT, "decoupling_placebo_classified.csv"), index=False)

    # ---- pollutant intensity (NOx or PM2.5 per tonne of CO2) on storm events vs placebo dates, port area
    rng = np.random.default_rng(5)

    def inten(d, p, n=2000):
        g = d.groupby("SID")[[f"obs_{p}", "obs_co2", f"exp_{p}", "exp_co2"]].sum()
        f = lambda s: (s[..., 0] / s[..., 1]) / (s[..., 2] / s[..., 3])
        est = float(f(g.values.sum(0)))
        b = f(g.values[rng.integers(0, len(g), (n, len(g)))].sum(1))
        return est, b

    IT = []
    Et, Pt = E[E.zone == "total"], PL[PL.zone == "total"].merge(P[["SID", "port", "intensity"]], on=["SID", "port"], how="left")
    for w in WINDOWS:
        for g in ["All events"] + list(Et.intensity.dropna().unique()):
            de = Et[Et.window == w] if g == "All events" else Et[(Et.window == w) & (Et.intensity == g)]
            dp = Pt[Pt.window == w] if g == "All events" else Pt[(Pt.window == w) & (Pt.intensity == g)]
            for p in ("nox", "pm25"):
                se, sb = inten(de, p); pe, pb = inten(dp, p)
                IT.append(dict(window=w, intensity=g, pollutant=p, storm=se, storm_lo=np.percentile(sb, 2.5), storm_hi=np.percentile(sb, 97.5),
                               placebo=pe, placebo_lo=np.percentile(pb, 2.5), placebo_hi=np.percentile(pb, 97.5),
                               diff=se / pe, diff_lo=np.percentile(sb / pb, 2.5), diff_hi=np.percentile(sb / pb, 97.5),
                               n_events=len(de), n_placebo=len(dp)))
    IT = pd.DataFrame(IT); IT.to_csv(os.path.join(OUT, "pollutant_intensity_test.csv"), index=False)
    print(IT.round(3).to_string(index=False))

    C = []
    for env in ("cls_co2", "cls_nox"):
        for (w, z), d in E.groupby(["window", "zone"]):
            for g, dd in [("all", d)] + list(d.groupby("intensity")):
                vc = dd[env].value_counts()
                C.append(dict(env=env[4:], window=w, zone=z, intensity=g, n=len(dd), **{o: int(vc.get(o, 0)) for o in ORDER}))
    C = pd.DataFrame(C); C.to_csv(os.path.join(OUT, "decoupling_counts.csv"), index=False)
    pd.set_option("display.width", 220)
    pc = PL.groupby(["window", "zone"]).cls_co2.value_counts(normalize=True).unstack().round(3)
    print("placebo class shares (false-positive check):\n", pc)
    print(C[(C.intensity == "all") & (C.env == "co2")].to_string(index=False))
    print(C[(C.zone == "total") & (C.env == "co2")].to_string(index=False))
    Z = E[E.zone != "total"].groupby(["SID", "port", "window", "intensity"]).cls_co2.apply(lambda s: (s == "absolute").any()).rename("any_abs").reset_index()
    ZC = Z.groupby(["window", "intensity"]).any_abs.agg(["sum", "count"]).reset_index()
    ZC = pd.concat([ZC, Z.groupby("window").any_abs.agg(["sum", "count"]).reset_index().assign(intensity="all")])
    ZC.to_csv(os.path.join(OUT, "decoupling_zone_counts.csv"), index=False)
    print(ZC.to_string(index=False))
