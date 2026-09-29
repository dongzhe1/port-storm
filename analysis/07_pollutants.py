"""Step 7 - NOx and PM2.5 from fuel by engine, on the same baseline as CO2 (Supplementary Fig. 3).
NOx and PM2.5 = fuel by engine (main, auxiliary, boiler) x fleet-average emission factors. Ship-hours and CO2 use the
all-ship series of each zone; the pollutant normal is the CO2 normal times the class-by-engine NOx (PM2.5) per tonne
of CO2 (panel.window_sums), so all measures share one baseline. SO2 to air comes from the upstream sulfur model.
Writes pollutant_tier_pairs.csv (zone x window), pollutant_event30_pairs.csv (study region, whole event),
port_window_pairs.csv (study region, all windows), pollutant_pooled.csv, divergence_pooled.csv, window_sums_events.csv.
"""
import os
import numpy as np, pandas as pd
import panel
from panel import OUT, TIERS

rng = np.random.default_rng(1)
R = pd.read_csv(os.path.join(OUT, "panel_results.csv.gz")); P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
D = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz")); D = D[D.placebo_year.isna()]
MEAS = ["hours", "co2", "nox", "pm25"]


def boot(d, a, b, n=1000):
    g = d.groupby("SID")[[a, b]].sum().values
    s = g[rng.integers(0, len(g), (n, len(g)))].sum(1)
    return np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5])


W = panel.window_sums(D).merge(P[["SID", "port", "intensity"]], on=["SID", "port"])
W.to_csv(os.path.join(OUT, "window_sums_events.csv"), index=False)

# ---- zone level, windows prep / storm / after
Z = W[(W.zone != "total") & W.window.isin(["prep", "storm", "after"])].rename(columns={"window": "phase", "zone": "tier"})
S = Z[["SID", "port", "intensity", "phase", "tier"]].copy()
for m in MEAS:
    S["o_" + m] = Z["obs_" + m].values; S["e_" + m] = Z["exp_" + m].fillna(0).values
x = R[(R.measure == "so2_air_ton") & (R.vclass == "all") & (R.tier != "total")][["SID", "port", "phase", "tier", "obs", "expected"]]
S = S.merge(x.rename(columns={"obs": "o_so2", "expected": "e_so2"}), on=["SID", "port", "phase", "tier"], how="left").fillna(0)
S.to_csv(os.path.join(OUT, "pollutant_tier_pairs.csv"), index=False)

rows = []
for (g, ph), d in S.groupby(["intensity", "phase"]):
    for p in ["co2", "nox", "pm25", "so2"]:
        lo, hi = boot(d, "o_" + p, "e_" + p)
        dd = d.groupby("tier")[["o_" + p, "e_" + p]].sum(); delta = dd["o_" + p] - dd["e_" + p]
        anc = dd.loc["anchorage"]
        rows.append(dict(intensity=g, phase=ph, pollutant=p, ratio=d["o_" + p].sum() / d["e_" + p].sum(), lo=lo, hi=hi,
                         share_outside=delta.drop(["port", "channel"], errors="ignore").sum() / delta.sum(),
                         anch_ratio=anc["o_" + p] / anc["e_" + p]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "pollutant_pooled.csv"), index=False)

# ---- port area (all-ship total series), every window
T = W[W.zone == "total"].copy()
T.to_csv(os.path.join(OUT, "port_window_pairs.csv"), index=False)
E = T[T.window == "event30"][["SID", "port", "intensity"] + [f"{k}_{m}" for m in MEAS for k in ("obs", "exp")]]
E.to_csv(os.path.join(OUT, "pollutant_event30_pairs.csv"), index=False)

dv = []
for (g, w), d in T.groupby(["intensity", "window"]):
    for m in MEAS:
        lo, hi = boot(d, "obs_" + m, "exp_" + m)
        dv.append(dict(intensity=g, window=w, measure=m, ratio=d["obs_" + m].sum() / d["exp_" + m].sum(), lo=lo, hi=hi))
for w, d in T.groupby("window"):
    for m in MEAS:
        lo, hi = boot(d, "obs_" + m, "exp_" + m)
        dv.append(dict(intensity="All events", window=w, measure=m, ratio=d["obs_" + m].sum() / d["exp_" + m].sum(), lo=lo, hi=hi))
DV = pd.DataFrame(dv); DV.to_csv(os.path.join(OUT, "divergence_pooled.csv"), index=False)
print(DV.pivot_table(index=["intensity", "measure"], columns="window", values="ratio").round(3).to_string())
