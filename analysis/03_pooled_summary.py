"""Step 3 - pooled changes from normal with 95% intervals from resampling cyclones (bootstrap, 1,000 draws).
Writes pooled_ratios.csv (cyclone class or distance class x zone x period x measure), recovery_pairs.csv,
recovery_pooled.csv and engine_intensity.csv (fuel per ship-hour by engine during the storm window).
"""
import os
import numpy as np, pandas as pd
from panel import OUT, TIERS, boot_ratio

R = pd.read_csv(os.path.join(OUT, "panel_results.csv.gz"))
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz"))
R["dist"] = pd.cut(R.min_dist_km, [0, 50, 100, 200], labels=["≤50 km", "50–100 km", "100–200 km"])
ALL = TIERS + ["total"]

# 1) pooled ratios
out = []
for m in ["hours", "co2_ton", "n_vessels_in_cell", "so2_air_ton", "so2_washwater_ton", "port_calls", "vessels_present"]:
    for grp in ["intensity", "dist"]:
        for g, d0 in R[(R.vclass == "all") & (R.measure == m)].groupby(grp, observed=True):
            for t in d0.tier.unique():
                for ph in ["prep", "storm", "after"]:
                    d = d0[(d0.tier == t) & (d0.phase == ph)]
                    if len(d) < 5:
                        continue
                    lo, hi = boot_ratio(d, "obs", "expected")
                    out.append(dict(group=grp, level=g, measure=m, tier=t, phase=ph, n=len(d),
                                    ratio=d.obs.sum() / d.expected.sum(), lo=lo, hi=hi, delta=d.delta.sum(),
                                    share_outside=d.outside.mean()))
A = pd.DataFrame(out); A.to_csv(os.path.join(OUT, "pooled_ratios.csv"), index=False)

# 2) removed / moved / delayed
rec = []
for (sid, port), d in Dd.groupby(["SID", "port"]):
    k = int(d.k_days.iloc[0]); d = d.set_index("rel_day")
    for v in ["tanker", "cargo", "other", "all"]:
        for m in ["hours", "co2_ton"]:
            c = f"{v}|total|{m}"
            if "obs:" + c not in d:
                continue
            x = d["obs:" + c] - d["exp:" + c]
            tiers = [f"{v}|{t}|{m}" for t in TIERS if "obs:" + f"{v}|{t}|{m}" in d]
            ev = pd.Series({t: (d["obs:" + t] - d["exp:" + t]).loc[-2:k - 1].sum() for t in tiers})
            r = dict(SID=sid, port=port, vclass=v, measure=m, loss=-x.loc[-2:k - 1].sum(),
                     gross=-ev[ev < 0].sum(), moved=ev[ev > 0].sum())
            for n in [10, 20, 30, 45]:
                r[f"rec{n}"] = x.loc[k:k + n - 1].sum()
            rec.append(r)
REC = pd.DataFrame(rec).merge(P[["SID", "port", "intensity", "min_dist_km"]], on=["SID", "port"])
REC.to_csv(os.path.join(OUT, "recovery_pairs.csv"), index=False)
rows = []
for (v, m, g), d0 in REC.groupby(["vclass", "measure", "intensity"]):
    d = d0[d0.loss > 0]
    if len(d) < 5:
        continue
    r = dict(vclass=v, measure=m, intensity=g, n=len(d), loss=d.loss.sum(), moved_pct=100 * d.moved.sum() / d.gross.sum())
    for n in [10, 20, 30, 45]:
        r[f"rec{n}_pct"] = 100 * d[f"rec{n}"].sum() / d.loss.sum()
        lo, hi = boot_ratio(d, f"rec{n}", "loss"); r[f"rec{n}_lo"], r[f"rec{n}_hi"] = 100 * lo, 100 * hi
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "recovery_pooled.csv"), index=False)

# 3) engine intensity in the storm window
e = R[(R.phase == "storm") & R.measure.isin(["hours", "fc_main_ton", "fc_AE_ton", "fc_boiler_ton"])]
pv = e.pivot_table(index=["SID", "port", "vclass", "tier", "intensity"], columns="measure", values=["obs", "expected"],
                   aggfunc="sum", observed=True)
eng = []
for (v, t, g), d in pv.groupby(level=["vclass", "tier", "intensity"]):
    if len(d) < 5:
        continue
    ho, he = d[("obs", "hours")].sum(), d[("expected", "hours")].sum()
    r = dict(vclass=v, tier=t, intensity=g, n=len(d), hours_ratio=ho / he)
    for f, lab in [("fc_main_ton", "main"), ("fc_AE_ton", "aux"), ("fc_boiler_ton", "boiler")]:
        if ("obs", f) in d:
            r[f"{lab}_per_hour_ratio"] = (d[("obs", f)].sum() / ho) / (d[("expected", f)].sum() / he)
    eng.append(r)
pd.DataFrame(eng).to_csv(os.path.join(OUT, "engine_intensity.csv"), index=False)
print(A[(A.group == "intensity") & (A.measure == "co2_ton") & (A.tier == "total")][["level", "phase", "ratio", "lo", "hi"]].round(2).to_string(index=False))
