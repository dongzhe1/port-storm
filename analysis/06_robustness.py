"""Step 6 - robustness (Methods, 'Uncertainty and robustness'; Extended Data Fig. 1).
 a) Placebo dates: every cyclone-free baseline year in turn plays the event year (same calendar window, same port),
    scored against the remaining baseline years; permutation test with 2,000 draws of one placebo year per event.
 b) Leave one port or one cyclone out.
 c) Nine alternative normals (reference window, pooling width, median or mean, exclusion radius, minimum number of
    baseline years, flat same-year normal).
 d) Dose-response: event-level ln(observed / normal) on wind at closest approach, distance, landfall, surge and rain,
    with cyclone-clustered standard errors, plus 10-knot wind bins.
Writes robust_placebo.csv, robust_permutation.csv, robust_loo.csv, robust_normals.csv, robust_dose_response.csv,
robust_dose_bins.csv, robust_jit_placebo.csv, placebo_results.csv.gz.
"""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from panel import load, run, OUT, boot_ratio

D = load()
R = pd.read_csv(os.path.join(OUT, "panel_results.csv.gz"))
HEAD = [  # (label, vclass, tier, measure, phase)
    ("Total CO2, storm window", "all", "total", "co2_ton", "storm"),
    ("Total CO2, after window", "all", "total", "co2_ton", "after"),
    ("Total vessel-hours, storm window", "all", "total", "hours", "storm"),
    ("Berth CO2, storm window", "all", "port", "co2_ton", "storm"),
    ("Anchorage vessel-hours, after window", "all", "anchorage", "hours", "after"),
    ("Offshore-hold vessel-hours, after window", "all", "offshore_hold", "hours", "after"),
    ("Open-water vessel-hours, storm window", "all", "offshore", "hours", "storm"),
    ("Port calls, storm window", "all", "port", "port_calls", "storm"),
]
COLS = sorted({f"{v}|{t}|{m}" for _, v, t, m, _ in HEAD})


def pooled(df, key):
    _, v, t, m, ph = key
    d = df[(df.vclass == v) & (df.tier == t) & (df.measure == m) & (df.phase == ph)]
    return d


# ---------------------------------------------------------------- a) placebo
PL, _ = run(D, cols=COLS, placebo=True)
PL.to_csv(os.path.join(OUT, "placebo_results.csv.gz"), index=False)
rows, perm = [], []
rng = np.random.default_rng(7)
for key in HEAD:
    for g in ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt", "All"]:
        real = pooled(R, key); pl = pooled(PL, key)
        if g != "All":
            real, pl = real[real.intensity == g], pl[pl.intensity == g]
        if len(pl) < 5:
            continue
        lo, hi = boot_ratio(pl, "obs", "expected")
        rr = real.obs.sum() / real.expected.sum(); pr = pl.obs.sum() / pl.expected.sum()
        rows.append(dict(outcome=key[0], intensity=g, n_real=len(real), n_placebo=len(pl), real_ratio=rr,
                         placebo_ratio=pr, placebo_lo=lo, placebo_hi=hi, placebo_share_outside=pl.outside.mean(),
                         real_share_outside=real.outside.mean()))
        # permutation: one placebo year per pair
        grp = pl.groupby(["SID", "port"])
        arrs = [(x.obs.values, x.expected.values) for _, x in grp]
        draws = np.empty(2000)
        for b in range(2000):
            o = e = 0.0
            for ob, ex in arrs:
                j = rng.integers(0, len(ob)); o += ob[j]; e += ex[j]
            draws[b] = o / e
        p = (np.sum(np.abs(np.log(draws)) >= abs(np.log(rr))) + 1) / (len(draws) + 1)
        perm.append(dict(outcome=key[0], intensity=g, real_ratio=rr, null_median=np.median(draws),
                         null_2_5=np.percentile(draws, 2.5), null_97_5=np.percentile(draws, 97.5), p_perm=p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "robust_placebo.csv"), index=False)
pd.DataFrame(perm).to_csv(os.path.join(OUT, "robust_permutation.csv"), index=False)

# ---------------------------------------------------------------- b) leave-one-out
loo = []
for key in HEAD:
    d = pooled(R, key); d = d[d.intensity == "Hurricane ≥64 kt"]
    full = d.obs.sum() / d.expected.sum()
    for by in ["port", "SID"]:
        vals = {u: d[d[by] != u].obs.sum() / d[d[by] != u].expected.sum() for u in d[by].unique()}
        s = pd.Series(vals)
        loo.append(dict(outcome=key[0], intensity="Hurricane ≥64 kt", drop=by, full=full, min=s.min(), max=s.max(),
                        most_influential=s.sub(full).abs().idxmax(), n_units=len(s)))
pd.DataFrame(loo).to_csv(os.path.join(OUT, "robust_loo.csv"), index=False)

# ---------------------------------------------------------------- c) alternative normals
VARIANTS = {"default": {}, "reference [-60,-11]": dict(ref=(60, 11)), "reference [-30,-8]": dict(ref=(30, 8)),
            "no day pooling": dict(pool=0), "pool ±7 days": dict(pool=7), "mean not median": dict(stat="mean"),
            "exclusion 150 km": dict(excl_km=150), "exclusion 400 km": dict(excl_km=400),
            "≥4 baseline years": dict(min_base=4), "same-year flat (package style)": dict(normal="same_year")}
nr = []
for name, opt in VARIANTS.items():
    Rv, _ = run(D, opts=opt, cols=COLS)
    for key in HEAD:
        for g in ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]:
            d = pooled(Rv, key); d = d[d.intensity == g]
            if len(d) < 5:
                continue
            lo, hi = boot_ratio(d, "obs", "expected")
            nr.append(dict(variant=name, outcome=key[0], intensity=g, n=len(d), ratio=d.obs.sum() / d.expected.sum(), lo=lo, hi=hi))
pd.DataFrame(nr).to_csv(os.path.join(OUT, "robust_normals.csv"), index=False)

# ---------------------------------------------------------------- d) dose-response
dr, bins = [], []
for key in [HEAD[0], HEAD[1], HEAD[4], HEAD[6]]:
    d = pooled(R, key).copy()
    d = d[(d.obs > 0) & (d.expected > 0)]
    d["y"] = np.log(d.obs / d.expected)
    d["wind10"] = d.wind_at_closest_kt / 10; d["dist100"] = d.min_dist_km / 100
    d["landfall"] = d.landfall.astype(float); d["logexp"] = np.log(d.expected)
    for name, f in {"wind + distance": "y ~ wind10 + dist100",
                    "+ landfall + size": "y ~ wind10 + dist100 + landfall + logexp",
                    "+ surge + rain (subset)": "y ~ wind10 + dist100 + landfall + logexp + surge_m + I(rain_mm/100)"}.items():
        dd = d.dropna(subset=["surge_m", "rain_mm"]) if "surge" in f else d
        m = smf.wls(f, dd, weights=np.sqrt(dd.expected)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.SID)[0]})
        for p in m.params.index:
            dr.append(dict(outcome=key[0], model=name, term=p, coef=m.params[p], se=m.bse[p], p=m.pvalues[p],
                           n=int(m.nobs), n_storms=dd.SID.nunique()))
    d["wbin"] = pd.cut(d.wind_at_closest_kt, [0, 30, 40, 50, 64, 83, 96, 200],
                       labels=["<30", "30–39", "40–49", "50–63", "64–82 (Cat1)", "83–95 (Cat2)", "≥96 (Cat3+)"])
    for b, x in d.groupby("wbin", observed=True):
        lo, hi = boot_ratio(x, "obs", "expected") if len(x) >= 3 else (np.nan, np.nan)
        bins.append(dict(outcome=key[0], wind_bin=b, n=len(x), ratio=x.obs.sum() / x.expected.sum(), lo=lo, hi=hi))
pd.DataFrame(dr).to_csv(os.path.join(OUT, "robust_dose_response.csv"), index=False)
pd.DataFrame(bins).to_csv(os.path.join(OUT, "robust_dose_bins.csv"), index=False)

pd.set_option("display.width", 220)
print(pd.DataFrame(rows)[lambda x: x.intensity.isin(["Hurricane ≥64 kt", "All"])].round(3).to_string(index=False))
print(pd.DataFrame(perm)[lambda x: x.intensity == "Hurricane ≥64 kt"].round(3).to_string(index=False))
print(pd.DataFrame(loo).round(3).to_string(index=False))
print(pd.DataFrame(nr)[lambda x: x.intensity == "Hurricane ≥64 kt"].pivot_table(index="variant", columns="outcome", values="ratio").round(2).to_string())
print(pd.DataFrame(dr)[lambda x: x.term.isin(["wind10", "dist100"])].round(3).to_string(index=False))
print(pd.DataFrame(bins).round(2).to_string(index=False))

# ---------------------------------------------------------------- e) placebo check for the JIT queue quantity
# net excess waiting hours (anchorage + offshore hold, all classes, after window) per pair: real vs placebo
q = []
for name, df in [("real", R), ("placebo", PL)]:
    d = df[(df.vclass == "all") & df.tier.isin(["anchorage", "offshore_hold"]) & (df.measure == "hours") & (df.phase == "after")]
    for g, x in d.groupby("intensity", observed=True):
        npy = x[["SID", "port", "placebo_year"]].drop_duplicates().shape[0] if "placebo_year" in x else x[["SID", "port"]].drop_duplicates().shape[0]
        q.append(dict(sample=name, intensity=g, pair_years=npy, net_excess_h=x.delta.sum(),
                      net_excess_h_per_pair_year=x.delta.sum() / npy))
Q = pd.DataFrame(q); Q.to_csv(os.path.join(OUT, "robust_jit_placebo.csv"), index=False)
print(Q.round(0).to_string(index=False))
