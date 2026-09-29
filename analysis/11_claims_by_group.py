"""Step 11 - do the headline results hold across kinds of events? (Supplementary Table 1)
Each headline result is recomputed within subgroups of the 344 events (cyclone class, port type, port layout, coast,
period 2015-19 vs 2020-23) and leaving out one cyclone at a time, as pooled ratios with 95% intervals from resampling
cyclones:
  C1 storm-window CO2 change; C2 share of the storm-window CO2 change outside the port area; C3 whole-event CO2;
  C4 whole-event CO2 per port call; C5 CO2 change minus ship-hour change (storm window, ten days after);
  C6 whole-event ship-hours, NOx and PM2.5; C7 arrival-management saving after hurricanes in days of normal CO2.
Also the landfall split quoted in Results (hurricane storm-window CO2 by whether the cyclone made landfall).
Writes claims_by_group.csv, claims_loo.csv, claims_landfall.csv.
"""
import os
import numpy as np, pandas as pd
import panel

OUT = panel.OUT
rng = np.random.default_rng(11)
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]
E = pd.read_csv(os.path.join(OUT, "decoupling_events.csv")); E = E[E.zone == "total"]
PT = pd.read_csv(os.path.join(OUT, "het_port_traits.csv"), index_col=0)
TP = pd.read_csv(os.path.join(OUT, "pollutant_tier_pairs.csv"))
EV = pd.read_csv(os.path.join(OUT, "pollutant_event30_pairs.csv"))
EF = pd.read_csv(os.path.join(OUT, "efficiency_pairs.csv")); EF = EF[EF.window == "event30"]
J = pd.read_csv(os.path.join(OUT, "jit_pairs.csv")); J = J[J.window == "queue"]

# one row per event with everything needed
W = {w: E[E.window == w].set_index(["SID", "port"]) for w in ["storm", "after", "event30"]}
B = W["storm"][["storm", "season", "intensity"]].copy()
for w, d in W.items():
    for m in ["hours", "co2", "nox"]:
        B[f"{w}_o_{m}"] = d[f"obs_{m}"]; B[f"{w}_e_{m}"] = d[f"exp_{m}"]
tp = TP.groupby(["SID", "port", "phase"])[["o_pm25", "e_pm25"]].sum().unstack("phase")
for ph in ["storm", "after"]:
    B[f"{ph}_o_pm25"] = tp[("o_pm25", ph)]; B[f"{ph}_e_pm25"] = tp[("e_pm25", ph)]
ev = EV.set_index(["SID", "port"]); B["event30_o_pm25"] = ev.obs_pm25; B["event30_e_pm25"] = ev.exp_pm25
st = TP[TP.phase == "storm"].assign(d=lambda x: x.o_co2 - x.e_co2)
st["outside"] = st.tier.isin(["anchorage", "offshore_hold", "offshore"])
dd = st.groupby(["SID", "port", "outside"]).d.sum().unstack(fill_value=0)
B["d_out"] = dd[True]; B["d_all"] = dd[True] + dd[False]
ef = EF.set_index(["SID", "port"])
B["o_calls"] = ef.obs_calls; B["e_calls"] = ef.exp_calls; B["o_co2c"] = ef.obs_all_total_co2; B["e_co2c"] = ef.exp_all_total_co2
PER_H = 72 * (1 - (72 / 96) ** 2) / 24                        # t main fuel saved per (waiting h x t/h), T=72, W=24
j = J.assign(s=J.net_excess_wait_h * J.f_main_t_per_h * PER_H * 3.114).groupby(["SID", "port"]).s.sum()
B["jit_co2"] = j
B = B.reset_index()
kd = pd.read_csv(os.path.join(OUT, "panel_pairs.csv")).set_index(["SID", "port"]).k_days
B["norm_co2_day"] = B.event30_e_co2 / (B.set_index(["SID", "port"]).index.map(kd).values + 32)
B = B.merge(PT[["port_type", "layout", "region"]], left_on="port", right_index=True, how="left")
B["period"] = np.where(B.season <= 2019, "2015–2019", "2020–2023")
B["region2"] = np.where(B.region.isin(["Gulf", "Atlantic"]), B.region, "Caribbean / Great Lakes")


def pooled(d, num, den, n=1000, fn=None):
    """ratio of sums (or fn of several sums) with storm-clustered bootstrap."""
    cols = num + den if isinstance(num, list) else [num, den]
    g = d.groupby("SID")[cols].sum()
    f = fn or (lambda s: s[:, 0] / s[:, 1])
    est = float(f(g.values.sum(axis=0, keepdims=True))[0])
    if len(g) < 3:
        return est, np.nan, np.nan
    s = g.values[rng.integers(0, len(g), size=(n, len(g)))].sum(axis=1)
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(f(s), [2.5, 97.5])
    return est, float(lo), float(hi)


def claims(d):
    r = {"n_events": len(d), "n_storms": d.SID.nunique(), "n_ports": d.port.nunique()}
    for g, lab in [(INTS[0], "TS"), (INTS[2], "hur")]:
        x = d[d.intensity == g]
        r[f"n_{lab}"] = len(x)
        if len(x) >= 3:
            r[f"C1_co2_storm_{lab}"], r[f"C1_lo_{lab}"], r[f"C1_hi_{lab}"] = pooled(x, "storm_o_co2", "storm_e_co2")
    x = d[d.intensity == INTS[1]]
    if len(x) >= 3:
        r["C1_co2_storm_sTS"] = pooled(x, "storm_o_co2", "storm_e_co2")[0]
    r["C2_share_outside"], r["C2_lo"], r["C2_hi"] = pooled(d, "d_out", "d_all")
    r["C3_co2_event"], r["C3_lo"], r["C3_hi"] = pooled(d, "event30_o_co2", "event30_e_co2")
    y = d.dropna(subset=["o_calls"]); y = y[y.e_calls > 0]
    r["C4_co2_per_call"], r["C4_lo"], r["C4_hi"] = pooled(y, ["o_co2c", "o_calls"], ["e_co2c", "e_calls"],
                                                          fn=lambda s: (s[:, 0] / s[:, 1]) / (s[:, 2] / s[:, 3]))
    gap = lambda s: s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    r["C5_storm_co2_minus_hours"], r["C5s_lo"], r["C5s_hi"] = pooled(d, ["storm_o_co2", "storm_e_co2"], ["storm_o_hours", "storm_e_hours"], fn=gap)
    r["C5_after_co2_minus_hours"], r["C5a_lo"], r["C5a_hi"] = pooled(d, ["after_o_co2", "after_e_co2"], ["after_o_hours", "after_e_hours"], fn=gap)
    r["C6_hours_event"] = pooled(d, "event30_o_hours", "event30_e_hours")[0]
    r["C6_nox_event"], r["C6n_lo"], r["C6n_hi"] = pooled(d, "event30_o_nox", "event30_e_nox")
    r["C6_pm25_event"], r["C6p_lo"], r["C6p_hi"] = pooled(d, "event30_o_pm25", "event30_e_pm25")
    h = d[(d.intensity == INTS[2])].dropna(subset=["jit_co2"])
    if len(h) >= 3:
        g = h.groupby("SID")[["jit_co2", "norm_co2_day"]].sum()
        r["C7_jit_days_per_hur_event"] = max(g.jit_co2.sum(), 0) / g.norm_co2_day.sum()
        r["C7_share_hur_events_with_queue"] = float((h.jit_co2 > 0).mean())
        top = h.jit_co2.clip(lower=0).sort_values(ascending=False)
        r["C7_top3_share"] = float(top.head(3).sum() / top.sum()) if top.sum() > 0 else np.nan
    return r


rows = [dict(grouping="all", level="All events", **claims(B))]
for grp in ["intensity", "port_type", "layout", "region2", "period"]:
    for lev, d in B.groupby(grp):
        rows.append(dict(grouping=grp, level=lev, **claims(d)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "claims_by_group.csv"), index=False)

# leave one storm out: hurricane drop, whole-event CO2, JIT days
L = []
for sid in B.SID.unique():
    d = B[B.SID != sid]; c = claims(d)
    L.append(dict(left_out=sid, storm=B[B.SID == sid].storm.iloc[0], season=int(B[B.SID == sid].season.iloc[0]),
                  C1_hur=c.get("C1_co2_storm_hur"), C1_TS=c.get("C1_co2_storm_TS"), C2=c["C2_share_outside"],
                  C3=c["C3_co2_event"], C4=c["C4_co2_per_call"], C6_nox=c["C6_nox_event"], C7=c.get("C7_jit_days_per_hur_event")))
L = pd.DataFrame(L); L.to_csv(os.path.join(OUT, "claims_loo.csv"), index=False)

# landfall split (Results): storm-window CO2 by whether the cyclone made landfall; own random stream
LF = B.merge(pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))[["SID", "port", "landfall"]], on=["SID", "port"])
rng = np.random.default_rng(1)
lrows = []
for lf, d in LF.groupby("landfall"):
    lrows.append(dict(landfall=int(lf), n_events=len(d), n_cyclones=d.SID.nunique(), group="all classes"))
    h = d[d.intensity == INTS[2]]
    est, lo, hi = pooled(h, "storm_o_co2", "storm_e_co2")
    lrows.append(dict(landfall=int(lf), n_events=len(h), n_cyclones=h.SID.nunique(), group="hurricanes",
                      co2_storm_ratio=est, lo=lo, hi=hi))
pd.DataFrame(lrows).to_csv(os.path.join(OUT, "claims_landfall.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
show = ["grouping", "level", "n_events", "n_TS", "n_hur", "C1_co2_storm_TS", "C1_co2_storm_sTS", "C1_co2_storm_hur", "C2_share_outside",
        "C3_co2_event", "C3_lo", "C3_hi", "C4_co2_per_call", "C4_lo", "C4_hi"]
print(R[show].round(3).to_string(index=False))
show2 = ["grouping", "level", "C5_storm_co2_minus_hours", "C5s_lo", "C5s_hi", "C5_after_co2_minus_hours", "C5a_lo", "C5a_hi",
         "C6_hours_event", "C3_co2_event", "C6_nox_event", "C6n_lo", "C6n_hi", "C6_pm25_event", "C7_jit_days_per_hur_event",
         "C7_share_hur_events_with_queue", "C7_top3_share"]
print(R[show2].round(3).to_string(index=False))
print("leave-one-storm-out ranges:\n", L[["C1_hur", "C1_TS", "C2", "C3", "C4", "C6_nox", "C7"]].agg(["min", "max"]).round(3).to_string())
print(L.sort_values("C7").head(3)[["storm", "season", "C7", "C1_hur", "C3"]].round(3).to_string(index=False))
print(L.sort_values("C7").tail(3)[["storm", "season", "C7", "C1_hur", "C3"]].round(3).to_string(index=False))
