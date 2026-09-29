"""Step 5 - recovery arrival management (Methods, 'Recovery arrival management'; Fig. 4, Supplementary Table 4).
If queued tankers and cargo ships had been given later arrival times after reopening and slowed down on the
approach instead of waiting in anchorages and offshore holding areas, how much fuel, CO2 and air pollution
would have been avoided?
Net excess waiting hours = observed minus normal ship-hours in anchorages and offshore holding areas, summed over
the window (queue = ten days after the storm window; event30 = whole event) and pooled over the events of a
cyclone class before flooring at zero; 95% intervals from resampling cyclones.
A ship on an approach of T hours that would wait W hours instead covers the same distance in T + W hours. Main-engine
fuel per distance scales with speed squared (propeller law), so main-engine fuel on the approach falls by
1 - (T/(T+W))^2, with a speed floor of 60% of service speed. Generator and boiler fuel is burned per hour and is
unchanged, only moved from the anchorage to the approach. Main-engine fuel per hour is the event's normal
open-water value for that ship class (national class median if missing).
Scenarios: T in {24, 72, 120} h, W in {12, 24, 48} h, fuel USD {450, 650, 900} per t; CO2 = fuel x 3.114.
Writes jit_pairs.csv, jit_scenarios.csv.
"""
import os
import numpy as np, pandas as pd
from panel import OUT

Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz"))
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
CF = 3.114
SPEED_FLOOR = 0.6
WIN = {"queue": lambda k: (k, k + 9), "event30": lambda k: (-2, k + 29)}
rows = []
for (sid, port), d in Dd.groupby(["SID", "port"]):
    k = int(d.k_days.iloc[0]); d = d.set_index("rel_day")
    for w, f in WIN.items():
        a, b = f(k)
        for v in ["tanker", "cargo"]:
            s = lambda kind, t, m: d[f"{kind}:{v}|{t}|{m}"].loc[a:b].sum() if f"{kind}:{v}|{t}|{m}" in d else 0.0
            H_obs = s("obs", "anchorage", "hours") + s("obs", "offshore_hold", "hours")
            H_exp = s("exp", "anchorage", "hours") + s("exp", "offshore_hold", "hours")
            fuel_wait_obs = sum(s("obs", t, m) for t in ["anchorage", "offshore_hold"] for m in ["fc_AE_ton", "fc_boiler_ton", "fc_main_ton"])
            fuel_wait_exp = sum(s("exp", t, m) for t in ["anchorage", "offshore_hold"] for m in ["fc_AE_ton", "fc_boiler_ton", "fc_main_ton"])
            he = d[f"exp:{v}|offshore|hours"].loc[a:b].sum() if f"exp:{v}|offshore|hours" in d else 0
            fm = d[f"exp:{v}|offshore|fc_main_ton"].loc[a:b].sum() / he if he > 0 else np.nan
            rows.append(dict(SID=sid, port=port, window=w, vclass=v, net_excess_wait_h=H_obs - H_exp,
                             wait_h_obs=H_obs, wait_h_exp=H_exp, net_excess_wait_fuel_t=fuel_wait_obs - fuel_wait_exp,
                             f_main_t_per_h=fm))
J = pd.DataFrame(rows).merge(P[["SID", "storm", "season", "port", "intensity"]], on=["SID", "port"])
med = J.groupby("vclass").f_main_t_per_h.median()
J["f_main_t_per_h"] = J.f_main_t_per_h.fillna(J.vclass.map(med))
J.to_csv(os.path.join(OUT, "jit_pairs.csv"), index=False)


def save_per_hour(T, W):
    r = max(T / (T + W), SPEED_FLOOR)
    return T * (1 - r ** 2) / W


sc = []
rng = np.random.default_rng(1)
groups = list(J.groupby(["window", "intensity"])) + [((w, "All storms"), J[J.window == w]) for w in WIN]
for (w, g), d in groups:
    d = d.assign(fh=d.net_excess_wait_h * d.f_main_t_per_h)      # t main fuel "at stake" per unit per_h
    G = d.groupby("SID")[["net_excess_wait_h", "fh", "net_excess_wait_fuel_t"]].sum()
    B = G.values[rng.integers(0, len(G), size=(1000, len(G)))].sum(axis=1)
    H, FH = max(0.0, G.net_excess_wait_h.sum()), max(0.0, G.fh.sum())
    FH_lo, FH_hi = np.maximum(0, np.percentile(B[:, 1], [2.5, 97.5]))
    H_lo, H_hi = np.percentile(B[:, 0], [2.5, 97.5])
    for T in [24, 72, 120]:
        for W in [12, 24, 48, "upper"]:
            per_h = 2.0 if W == "upper" else save_per_hour(T, W)
            fuel, lo, hi = FH * per_h, FH_lo * per_h, FH_hi * per_h
            for price in [450, 650, 900]:
                sc.append(dict(window=w, intensity=g, n_pairs=d[["SID", "port"]].drop_duplicates().shape[0],
                               net_excess_wait_h=H, wait_h_lo=H_lo, wait_h_hi=H_hi,
                               relocated_hotel_fuel_t=max(0.0, G.net_excess_wait_fuel_t.sum()),
                               T_h=T, W_h=W, fuel_saved_t=fuel, co2_saved_t=fuel * CF, co2_lo=lo * CF, co2_hi=hi * CF,
                               fuel_price=price, fuel_cost_saved_usd=fuel * price,
                               scc51_usd=fuel * CF * 51, scc190_usd=fuel * CF * 190))
S = pd.DataFrame(sc)
S.to_csv(os.path.join(OUT, "jit_scenarios.csv"), index=False)
pd.set_option("display.width", 200)
base = S[(S.T_h == 72) & (S.fuel_price == 650)]
print(base.pivot_table(index=["window", "intensity"], columns="W_h", values="co2_saved_t", aggfunc="first").round(0).to_string())
print(base[base.W_h == 24][["window", "intensity", "n_pairs", "net_excess_wait_h", "wait_h_lo", "wait_h_hi", "relocated_hotel_fuel_t", "co2_saved_t", "co2_lo", "co2_hi",
                            "fuel_cost_saved_usd"]].round(0).to_string(index=False))
