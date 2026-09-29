"""Step 4 - carbon efficiency and port calls (Methods, 'Carbon efficiency'; Extended Data Fig. 7).
Windows, relative to the start of the storm window (k = its length in days):
  storm = days 0..k-1; queue = days k..k+9 (the ten days after); event30 = whole event, -2..k+29; event45 = -2..k+44.
Pooled over events (observed vs normal), 95% intervals from resampling cyclones:
  CO2 per ship-hour by zone and ship class; fuel per ship-hour by engine; waiting hours (anchorages + offshore
  holding areas, tankers and cargo ships) per port call; CO2 per port call.
Port calls come from port_daily_calls.csv (berth episodes of all ship types) and are used as ratios only.
Writes efficiency_pairs.csv, efficiency_pooled.csv, efficiency_engine_queue.csv.
"""
import os
import numpy as np, pandas as pd
from panel import OUT, TIERS, boot_ratio

Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz"))
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
WIN = {"storm": lambda k: (0, k - 1), "queue": lambda k: (k, k + 9),
       "event30": lambda k: (-2, k + 29), "event45": lambda k: (-2, k + 44)}
CL = ["all", "tanker", "cargo"]

rows = []
for (sid, port), d in Dd.groupby(["SID", "port"]):
    k = int(d.k_days.iloc[0]); d = d.set_index("rel_day")
    g = lambda kind, col, a, b: d[f"{kind}:{col}"].loc[a:b].sum() if f"{kind}:{col}" in d else np.nan
    for w, f in WIN.items():
        a, b = f(k)
        r = dict(SID=sid, port=port, window=w)
        for kind in ["obs", "exp"]:
            r[f"{kind}_calls"] = g(kind, "all|port|port_calls", a, b)
            for v in CL:
                for t in TIERS + ["total"]:
                    r[f"{kind}_{v}_{t}_co2"] = g(kind, f"{v}|{t}|co2_ton", a, b)
                    r[f"{kind}_{v}_{t}_hours"] = g(kind, f"{v}|{t}|hours", a, b)
                    for fu in ["fc_main_ton", "fc_AE_ton", "fc_boiler_ton"]:
                        r[f"{kind}_{v}_{t}_{fu}"] = g(kind, f"{v}|{t}|{fu}", a, b)
            r[f"{kind}_wait_hours"] = sum(g(kind, f"{v}|{t}|hours", a, b) for v in ["tanker", "cargo"]
                                         for t in ["anchorage", "offshore_hold"])
        rows.append(r)
E = pd.DataFrame(rows).merge(P[["SID", "port", "intensity", "wind_at_closest_kt"]], on=["SID", "port"])
E = E.fillna(0.0)
E["penalty_co2"] = E.obs_calls * (E.obs_all_total_co2 / E.obs_calls.replace(0, np.nan)
                                  - E.exp_all_total_co2 / E.exp_calls.replace(0, np.nan))
E.to_csv(os.path.join(OUT, "efficiency_pairs.csv"), index=False)


def ratio_of_ratios(d, n1, d1, n2, d2, B=1000, seed=1):
    """pooled (sum n1/sum d1)/(sum n2/sum d2) with storm-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    G = d.groupby("SID")[[n1, d1, n2, d2]].sum().values
    est = (G[:, 0].sum() / G[:, 1].sum()) / (G[:, 2].sum() / G[:, 3].sum())
    S = G[rng.integers(0, len(G), size=(B, len(G)))].sum(axis=1)
    with np.errstate(all="ignore"):
        bs = (S[:, 0] / S[:, 1]) / (S[:, 2] / S[:, 3])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi


out = []
for (w, g), d in E.groupby(["window", "intensity"]):
    d = d[(d.exp_calls > 0) & (d.exp_all_total_co2 > 0)]
    # 1 CO2 per vessel-hour
    for v in CL:
        for t in TIERS + ["total"]:
            dd = d[d[f"exp_{v}_{t}_hours"] > 0]
            if len(dd) < 5 or dd[f"obs_{v}_{t}_hours"].sum() == 0:
                continue
            est, lo, hi = ratio_of_ratios(dd, f"obs_{v}_{t}_co2", f"obs_{v}_{t}_hours", f"exp_{v}_{t}_co2", f"exp_{v}_{t}_hours")
            out.append(dict(window=w, intensity=g, metric="CO2 per vessel-hour", vclass=v, tier=t, n=len(dd), ratio=est, lo=lo, hi=hi))
    # 3 waiting hours per call, 4 CO2 per call, activity (calls) and total
    for name, (n1, d1, n2, d2) in {"waiting hours per call": ("obs_wait_hours", "obs_calls", "exp_wait_hours", "exp_calls"),
                                   "CO2 per port call": ("obs_all_total_co2", "obs_calls", "exp_all_total_co2", "exp_calls")}.items():
        est, lo, hi = ratio_of_ratios(d, n1, d1, n2, d2)
        out.append(dict(window=w, intensity=g, metric=name, vclass="all", tier="total", n=len(d), ratio=est, lo=lo, hi=hi))
    for name, (n, dn) in {"port calls": ("obs_calls", "exp_calls"), "CO2": ("obs_all_total_co2", "exp_all_total_co2")}.items():
        lo, hi = boot_ratio(d, n, dn)
        out.append(dict(window=w, intensity=g, metric=name, vclass="all", tier="total", n=len(d),
                        ratio=d[n].sum() / d[dn].sum(), lo=lo, hi=hi))
    # pooled penalty: CO2 beyond what the observed calls would have needed at the normal pooled CO2 per call
    G = d.groupby("SID")[["obs_all_total_co2", "obs_calls", "exp_all_total_co2", "exp_calls"]].sum().values
    pen = lambda S: S[..., 0] - S[..., 1] * S[..., 2] / S[..., 3]
    S = G[np.random.default_rng(1).integers(0, len(G), size=(1000, len(G)))].sum(axis=1)
    lo, hi = np.nanpercentile(pen(S), [2.5, 97.5])
    out.append(dict(window=w, intensity=g, metric="disruption penalty (t CO2, pooled)", vclass="all", tier="total", n=len(d),
                    ratio=pen(G.sum(axis=0)), lo=lo, hi=hi))
O = pd.DataFrame(out)
# 5 decomposition shares
dec = []
for (w, g), d in O[O.metric.isin(["port calls", "CO2 per port call", "CO2"])].groupby(["window", "intensity"]):
    r = d.set_index("metric").ratio
    tot = np.log(r["CO2"]); act = np.log(r["port calls"]); inten = np.log(r["CO2 per port call"])
    dec.append(dict(window=w, intensity=g, metric="decomposition: activity share of ln change", vclass="all", tier="total",
                    ratio=act / tot if tot != 0 else np.nan))
    dec.append(dict(window=w, intensity=g, metric="decomposition: efficiency share of ln change", vclass="all", tier="total",
                    ratio=inten / tot if tot != 0 else np.nan))
O = pd.concat([O, pd.DataFrame(dec)], ignore_index=True)
O.to_csv(os.path.join(OUT, "efficiency_pooled.csv"), index=False)

# 2 engine use in the queue window: fuel per vessel-hour ratio by engine
eng = []
q = E[E.window == "queue"]
for g, d in q.groupby("intensity"):
    for v in ["tanker", "cargo"]:
        for t in ["anchorage", "offshore_hold", "offshore", "port"]:
            dd = d[d[f"exp_{v}_{t}_hours"] > 0]
            if len(dd) < 5:
                continue
            r = dict(intensity=g, vclass=v, tier=t, n=len(dd),
                     hours_ratio=dd[f"obs_{v}_{t}_hours"].sum() / dd[f"exp_{v}_{t}_hours"].sum())
            for fu, lab in [("fc_main_ton", "main"), ("fc_AE_ton", "aux"), ("fc_boiler_ton", "boiler")]:
                est, lo, hi = ratio_of_ratios(dd, f"obs_{v}_{t}_{fu}", f"obs_{v}_{t}_hours", f"exp_{v}_{t}_{fu}", f"exp_{v}_{t}_hours")
                r[f"{lab}_per_hour"], r[f"{lab}_lo"], r[f"{lab}_hi"] = est, lo, hi
                r[f"{lab}_share_of_extra_fuel"] = (dd[f"obs_{v}_{t}_{fu}"].sum() - dd[f"exp_{v}_{t}_{fu}"].sum())
            tot = sum(r[f"{l}_share_of_extra_fuel"] for l in ["main", "aux", "boiler"])
            for l in ["main", "aux", "boiler"]:
                r[f"{l}_share_of_extra_fuel"] = r[f"{l}_share_of_extra_fuel"] / tot if tot else np.nan
            eng.append(r)
pd.DataFrame(eng).to_csv(os.path.join(OUT, "efficiency_engine_queue.csv"), index=False)

pd.set_option("display.width", 200)
k = O[O.metric.isin(["port calls", "CO2", "CO2 per port call", "waiting hours per call", "disruption penalty (t CO2, pooled)",
                     "decomposition: efficiency share of ln change"])]
k = k.assign(txt=k.apply(lambda r: f"{r.ratio:.3g} [{r.lo:.3g}, {r.hi:.3g}]" if pd.notna(r.lo) else f"{r.ratio:.3g}", axis=1))
print(k.pivot_table(index=["intensity", "metric"], columns="window", values="txt", aggfunc="first")[["storm", "queue", "event30", "event45"]].to_string())
print(O[(O.metric == "CO2 per vessel-hour") & (O.vclass == "all") & (O.intensity.str.startswith("Hurr"))]
      .pivot_table(index="tier", columns="window", values="ratio").round(2).to_string())
