"""Step 9 - which events respond, and why (Fig. 3, Extended Data Figs 4 and 6).
  (1) whole-event net change per event against placebo noise bands (Extended Data Fig. 4);
  (2) port type from the normal share of CO2 by ship type: energy (tankers >= 55%), service-craft (tugs, supply boats,
      fishing >= 40%), cargo (cargo ships >= 50%), otherwise mixed;
  (3) cyclone traits: closest distance, duration, surge, rain, landfall, wind;
  (4) port layout from the normal zone mix: river / inland, coastal with anchorage, open coast;
  (5) where the arrival-management potential sits (net excess waiting, ten days after, from jit_pairs.csv).
Regressions on event-level log ratios with standardised predictors and cyclone-clustered standard errors
(Extended Data Fig. 6). Writes het_*.csv.
"""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panel

OUT = panel.OUT
rng = np.random.default_rng(7)
E = pd.read_csv(os.path.join(OUT, "decoupling_events.csv"))
PL = pd.read_csv(os.path.join(OUT, "decoupling_placebo_classified.csv"))
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz")); Dd = Dd[Dd.placebo_year.isna()]
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]

# ------------------------------------------------------------------ port traits from the normal
rows = []
for (sid, port), d in Dd.groupby(["SID", "port"]):
    k = int(d.k_days.iloc[0]); x = d.set_index("rel_day").loc[-2:k + 29]
    r = dict(SID=sid, port=port, days=len(x))
    for v in ["tanker", "cargo", "passenger", "other", "all"]:
        r[v] = x.get(f"exp:{v}|total|co2_ton", pd.Series(0.0)).sum()
    for t in panel.TIERS:
        r["z_" + t] = x.get(f"exp:all|{t}|co2_ton", pd.Series(0.0)).sum()
    rows.append(r)
TR = pd.DataFrame(rows)
PT = TR.groupby("port").sum(numeric_only=True)
for v in ["tanker", "cargo", "passenger", "other"]:
    PT[v + "_sh"] = PT[v] / PT["all"]
PT["inner_sh"] = (PT.z_port + PT.z_channel) / PT["all"]
PT["wait_sh"] = (PT.z_anchorage + PT.z_offshore_hold) / PT["all"]
PT["open_sh"] = PT.z_offshore / PT["all"]
PT["size"] = np.log(PT["all"] / PT.days)
PT["port_type"] = np.select([PT.tanker_sh >= .55, PT.other_sh >= .40, PT.cargo_sh >= .50], ["Energy", "Service", "Cargo"], "Mixed")
PT["layout"] = np.select([PT.open_sh < .05, PT.wait_sh >= .15], ["River / inland", "Coastal, anchorage"], "Open coast")
ST = {"TX": "Gulf", "LA": "Gulf", "MS": "Gulf", "AL": "Gulf", "PR": "Caribbean", "VI": "Caribbean",
      "MI": "Great Lakes", "WI": "Great Lakes", "IL": "Great Lakes"}
GULF_FL = ["Tampa", "Manatee", "Panama City"]
def region(p):
    s = p.replace(", Port of", "").split(",")[-1].strip()[:2]
    if p.startswith("Virgin Islands"):
        return "Caribbean"
    if s == "FL":
        return "Gulf" if any(g in p for g in GULF_FL) else "Atlantic"
    return ST.get(s, "Atlantic")
PT["region"] = [region(p) for p in PT.index]
PT.to_csv(os.path.join(OUT, "het_port_traits.csv"))

T = E[E.zone == "total"].merge(PT[["tanker_sh", "cargo_sh", "passenger_sh", "other_sh", "inner_sh", "wait_sh", "open_sh",
                                    "port_type", "layout", "region"]].rename(columns={"size": "port_size"}),
                               left_on="port", right_index=True, how="left")
T["dist_class"] = pd.cut(T.min_dist_km, [-1, 50, 100, 201], labels=["<50 km", "50–100 km", "100–200 km"])
T["dur_class"] = pd.cut(T.k_days, [0, 2, 4, 99], labels=["1–2 days", "3–4 days", "≥5 days"])
W = {w: T[T.window == w].set_index(["SID", "port"]) for w in ["storm", "after", "event30"]}


def boot(df, num, den, n=1000):
    g = df.groupby("SID")[[num, den]].sum().values
    if len(g) < 3:
        return np.nan, np.nan
    s = g[rng.integers(0, len(g), size=(n, len(g)))].sum(axis=1)
    with np.errstate(all="ignore"):
        return tuple(np.nanpercentile(s[:, 0] / s[:, 1], [2.5, 97.5]))


# ------------------------------------------------------------------ (1) winners and losers
w1 = []
for w in ["storm", "after", "event30"]:
    d = W[w]; pl = PL[(PL.zone == "total") & (PL.window == w)]
    for env, tau in [("lc", "tau_c"), ("ln", "tau_n"), ("lh", "tau_h")]:
        up = (d[env] > d[tau]).sum(); dn = (d[env] < -d[tau]).sum()
        pup = (pl[env] > pl[tau]).mean(); pdn = (pl[env] < -pl[tau]).mean()
        w1.append(dict(window=w, measure={"lc": "CO2", "ln": "NOx", "lh": "ship-hours"}[env], n=len(d),
                       sig_up=int(up), sig_down=int(dn), expected_up_by_chance=round(pup * len(d), 1),
                       expected_down_by_chance=round(pdn * len(d), 1), share_point_above_1=round((d[env] > 0).mean(), 3)))
W1 = pd.DataFrame(w1); W1.to_csv(os.path.join(OUT, "het_winners_losers.csv"), index=False)
d = W["event30"].reset_index()
d["dco2"] = d.obs_co2 - d.exp_co2
pos = d[d.dco2 > 0].sort_values("dco2", ascending=False)
top10_share_pos = pos.dco2.head(10).sum() / pos.dco2.sum()
net_share = d.dco2.sum() / d.exp_co2.sum()
UP = d[d.lc > d.tau_c][["storm", "season", "port", "intensity", "min_dist_km", "k_days", "lh", "lc", "ln", "port_type", "layout"]]
DN = d[d.lc < -d.tau_c][["storm", "season", "port", "intensity", "min_dist_km", "k_days", "lh", "lc", "ln", "port_type", "layout"]]
pd.concat([UP.assign(direction="net increase"), DN.assign(direction="net decrease")]).to_csv(os.path.join(OUT, "het_event30_significant.csv"), index=False)

# ------------------------------------------------------------------ (2)-(4) grouped pooled ratios
G = []
for grp in ["port_type", "layout", "region", "dist_class", "dur_class", "intensity"]:
    for w in ["storm", "after", "event30"]:
        dd = W[w].reset_index()
        for lev, x in dd.groupby(grp, observed=True):
            r = dict(grouping=grp, level=str(lev), window=w, n_events=len(x), n_ports=x.port.nunique())
            for m in ["hours", "co2", "nox"]:
                r[m] = x[f"obs_{m}"].sum() / x[f"exp_{m}"].sum()
                r[m + "_lo"], r[m + "_hi"] = boot(x, f"obs_{m}", f"exp_{m}")
            r["share_responding"] = (x.cls_co2.isin(["coupled", "relative (amplified)", "relative (damped)", "absolute"])).mean()
            r["share_rel_decoupled"] = x.cls_co2.str.startswith("relative").mean()
            G.append(r)
G = pd.DataFrame(G); G.to_csv(os.path.join(OUT, "het_grouped.csv"), index=False)
CC = []
for grp in ["intensity", "port_type", "layout"]:
    for w in ["storm", "after", "event30"]:
        for lev, x in W[w].reset_index().groupby(grp, observed=True):
            vc = x.cls_co2.value_counts()
            CC.append(dict(grouping=grp, level=str(lev), window=w, n=len(x), **{c: int(vc.get(c, 0)) for c in
                      ["no response", "coupled", "relative (amplified)", "relative (damped)", "absolute", "no data"]}))
CC = pd.DataFrame(CC); CC.to_csv(os.path.join(OUT, "het_class_counts.csv"), index=False)

# same, within hurricanes and strong TS only (to separate layout/type from storm strength)
G2 = []
for grp in ["port_type", "layout"]:
    for w in ["storm", "after", "event30"]:
        dd = W[w].reset_index(); dd = dd[dd.intensity != INTS[0]]
        for lev, x in dd.groupby(grp, observed=True):
            r = dict(grouping=grp, level=str(lev), window=w, n_events=len(x), n_ports=x.port.nunique())
            for m in ["hours", "co2", "nox"]:
                r[m] = x[f"obs_{m}"].sum() / x[f"exp_{m}"].sum()
                r[m + "_lo"], r[m + "_hi"] = boot(x, f"obs_{m}", f"exp_{m}")
            G2.append(r)
G2 = pd.DataFrame(G2); G2.to_csv(os.path.join(OUT, "het_grouped_strong_storms.csv"), index=False)

# ------------------------------------------------------------------ determinants (regressions)
X_CONT = {"wind_at_closest_kt": "Wind at closest approach", "log_dist": "Distance (log)", "k_days": "Duration",
          "surge_m": "Storm surge", "rain_mm": "Rainfall", "port_size": "Port size (log normal CO2)",
          "tanker_sh": "Tanker share", "other_sh": "Service-craft share", "passenger_sh": "Passenger share",
          "wait_sh": "Waiting-zone share", "open_sh": "Open-water share", "season": "Year"}
X_BIN = {"landfall": "Landfall", "reg_Atlantic": "Atlantic (vs Gulf)", "reg_Caribbean": "Caribbean", "reg_Great Lakes": "Great Lakes",
         "met_missing": "Surge/rain not observed"}
OUTC = [("storm", "lc", "CO2, storm window"), ("storm", "lh", "Ship-hours, storm window"), ("storm", "g", "Decoupling gap, storm window"),
        ("after", "lc", "CO2, 10 days after"), ("after", "g", "Decoupling gap, 10 days after"),
        ("event30", "lc", "CO2, whole event"), ("event30", "ln", "NOx, whole event")]
Rg = []
for w, y, lab in OUTC:
    d = W[w].reset_index().copy()
    d["log_dist"] = np.log(d.min_dist_km.clip(lower=5))
    d["port_size"] = d["size"]
    for r_ in ["Atlantic", "Caribbean", "Great Lakes"]:
        d["reg_" + r_] = (d.region == r_).astype(float)
    d["met_missing"] = (d.surge_m.isna() | d.rain_mm.isna()).astype(float)     # no CO-OPS gauge / ASOS match
    d["surge_m"] = d.surge_m.fillna(0.0); d["rain_mm"] = d.rain_mm.fillna(0.0)
    tau = {"lc": "tau_c", "lh": "tau_h", "g": "tau_g", "ln": "tau_n"}[y]
    cols = list(X_CONT) + list(X_BIN)
    d = d[[y, tau, "SID"] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    Z = d[cols].copy()
    for c in X_CONT:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
    Z = Z.loc[:, Z.std() > 0]                       # drop regions absent from the sample
    Z = sm.add_constant(Z)
    grp = pd.factorize(d.SID)[0]
    for spec, fit in [("OLS", sm.OLS(d[y], Z).fit(cov_type="cluster", cov_kwds={"groups": grp})),
                      ("WLS", sm.WLS(d[y], Z, weights=1 / d[tau] ** 2).fit(cov_type="cluster", cov_kwds={"groups": grp}))]:
        ci = fit.conf_int()
        for c in [c for c in cols if c in Z.columns]:
            Rg.append(dict(outcome=lab, window=w, y=y, spec=spec, term=c, label={**X_CONT, **X_BIN}[c], coef=fit.params[c],
                           lo=ci.loc[c, 0], hi=ci.loc[c, 1], p=fit.pvalues[c], n=int(fit.nobs), r2=fit.rsquared))
Rg = pd.DataFrame(Rg); Rg.to_csv(os.path.join(OUT, "het_determinants.csv"), index=False)

# ------------------------------------------------------------------ (5) just-in-time potential
J = pd.read_csv(os.path.join(OUT, "jit_pairs.csv"))
J = J[J.window == "queue"].groupby(["SID", "port", "intensity"]).apply(
    lambda x: pd.Series(dict(H=x.net_excess_wait_h.sum(), S=(2 * x.f_main_t_per_h * x.net_excess_wait_h).sum())), include_groups=False).reset_index()
J = J.merge(PT[["port_type", "layout", "region"]], left_on="port", right_index=True)
J5 = []
for grp in ["intensity", "port_type", "layout", "region"]:
    tot = J.S.sum()
    for lev, x in J.groupby(grp):
        J5.append(dict(grouping=grp, level=lev, n_events=len(x), net_excess_wait_h=x.H.sum(), fuel_saving_index_t=x.S.sum(),
                       share_of_national=x.S.sum() / tot))
J5 = pd.DataFrame(J5)
byport = J.groupby("port").S.sum().sort_values(ascending=False)
hur = J[J.intensity == INTS[2]].groupby("port").S.sum().sort_values(ascending=False)
J5.to_csv(os.path.join(OUT, "het_jit_concentration.csv"), index=False)
byport.rename("fuel_saving_index_t").to_csv(os.path.join(OUT, "het_jit_by_port.csv"))

# ------------------------------------------------------------------ print summary
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
print("== (1) winners/losers\n", W1.to_string(index=False))
print(f"whole event: net CO2 change {net_share:+.2%} of normal; top 10 of {len(pos)} events with a point increase carry {top10_share_pos:.0%} of all increases")
print("significant whole-event CO2 changes:\n", pd.concat([UP.assign(dir='up'), DN.assign(dir='down')]).round(2).to_string(index=False))
print("== port types\n", PT.groupby("port_type").size(), "\n", PT.groupby("layout").size(), "\n", PT.groupby("region").size())
print(PT.groupby(["port_type"]).apply(lambda x: ", ".join(x.index), include_groups=False).to_string())
print(PT.groupby(["layout"]).apply(lambda x: ", ".join(x.index), include_groups=False).to_string())
print("== grouped (all events)\n", G[["grouping", "level", "window", "n_events", "n_ports", "hours", "co2", "co2_lo", "co2_hi", "nox", "share_responding", "share_rel_decoupled"]].round(3).to_string(index=False))
print("== grouped (strong TS + hurricanes)\n", G2[["grouping", "level", "window", "n_events", "hours", "co2", "co2_lo", "co2_hi", "nox"]].round(3).to_string(index=False))
sig = Rg[(Rg.spec == "OLS")]
print("== determinants (OLS, standardized; * p<.05)\n", sig.assign(s=np.where(sig.p < .05, "*", "")).pivot_table(index="label", columns="outcome", values="coef").round(3).to_string())
print(sig[sig.p < .05][["outcome", "label", "coef", "lo", "hi", "p"]].round(3).to_string(index=False))
w_ = Rg[(Rg.spec == "WLS") & (Rg.p < .05)]
print("WLS significant:\n", w_[["outcome", "label", "coef", "p"]].round(3).to_string(index=False))
print("== class counts\n", CC[CC.window == "storm"].to_string(index=False))
print("== (5) JIT\n", J5.round(3).to_string(index=False))
print("top ports (all storms):\n", (byport / byport.sum()).head(8).round(3).to_string())
print("top ports (hurricanes):\n", (hur / hur.sum()).head(8).round(3).to_string())
