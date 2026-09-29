"""Step 12 - divergence against placebo dates, the waiting-zone check, and targeting (Extended Data Fig. 3; Fig. 4b).
(1) D = dln(CO2) - dln(ship-hours) for every event against the same D on placebo dates, by port type
    (Extended Data Fig. 3; Supplementary Note 1).
(2) Waiting-zone share and whole-event CO2 with controls, clustering and leave-one-port-out (rev_anchorage_robustness.csv;
    not shown in the manuscript figures).
(3) Targeting recovery arrival management: in the ten days after a hurricane, the net excess waiting emissions of
    tankers and cargo ships at each port, the avoided CO2 (T = 72 h, W = 24 h), and the share covered by targeting the
    top ports against the port-days of intervention needed (Fig. 4b, Supplementary Fig. 1b).
Writes rev_D_distributions.csv, rev_anchorage_robustness.csv, rev_intervention_*.csv and figures/ed3_divergence_placebo.png.
"""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
import panel

OUT = panel.OUT
FIG = panel.FIG
rng = np.random.default_rng(3)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]
TYPES = ["Cargo", "Mixed", "Energy", "Service"]
TCOL = {"Energy": "#d55e00", "Cargo": "#2a78d6", "Mixed": "#e69f00", "Service": "#56b4e9"}
PLAB = {"Energy": "Energy", "Cargo": "Cargo", "Mixed": "Mixed", "Service": "Service-craft"}
PT = pd.read_csv(os.path.join(OUT, "het_port_traits.csv"), index_col=0)
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
E = pd.read_csv(os.path.join(OUT, "decoupling_events.csv")); E = E[E.zone == "total"]
PL = pd.read_csv(os.path.join(OUT, "decoupling_placebo_classified.csv")); PL = PL[PL.zone == "total"]
E = E.merge(PT[["port_type", "layout", "region"]], left_on="port", right_index=True)
PL = PL.merge(PT[["port_type", "layout", "region"]], left_on="port", right_index=True).merge(
    P[["SID", "port", "intensity", "wind_at_closest_kt", "min_dist_km", "surge_m", "rain_mm", "landfall", "season"]], on=["SID", "port"])


def boot_mean_diff(a, ga, b, gb, n=2000):
    """difference in means, storm events minus placebo, resampling storms (clusters) in both."""
    A = pd.DataFrame({"v": a, "g": ga}).groupby("g").v.agg(["sum", "count"]).values
    B = pd.DataFrame({"v": b, "g": gb}).groupby("g").v.agg(["sum", "count"]).values
    ia = rng.integers(0, len(A), (n, len(A))); ib = rng.integers(0, len(B), (n, len(B)))
    ma = A[ia, 0].sum(1) / A[ia, 1].sum(1); mb = B[ib, 0].sum(1) / B[ib, 1].sum(1)
    return np.percentile(ma - mb, [2.5, 97.5])


# ------------------------------------------------------------------ (1) D distributions
D = []
for w in ["storm", "after", "event30"]:
    for t in ["All"] + TYPES:
        for g in ["All"] + INTS:
            s = E[(E.window == w) & ((E.port_type == t) | (t == "All")) & ((E.intensity == g) | (g == "All"))].dropna(subset=["g"])
            p = PL[(PL.window == w) & ((PL.port_type == t) | (t == "All")) & ((PL.intensity == g) | (g == "All"))].dropna(subset=["g"])
            if len(s) < 5:
                continue
            lo, hi = boot_mean_diff(s.g.values, s.SID.values, p.g.values, p.SID.values)
            D.append(dict(window=w, port_type=t, intensity=g, n_events=len(s), n_placebo=len(p),
                          mean_storm=s.g.mean(), median_storm=s.g.median(), sd_storm=s.g.std(),
                          mean_placebo=p.g.mean(), median_placebo=p.g.median(), sd_placebo=p.g.std(),
                          iqr_placebo=p.g.quantile(.75) - p.g.quantile(.25),
                          diff=s.g.mean() - p.g.mean(), diff_lo=lo, diff_hi=hi,
                          ks_p=stats.ks_2samp(s.g, p.g).pvalue,
                          share_storm_beyond_band=float((s.g.abs() > s.tau_g).mean()),
                          share_placebo_beyond_band=float((p.g.abs() > p.tau_g).mean())))
D = pd.DataFrame(D); D.to_csv(os.path.join(OUT, "rev_D_distributions.csv"), index=False)

# ------------------------------------------------------------------ (2) anchorage / waiting-zone robustness
def design(d):
    d = d.copy()
    d["log_dist"] = np.log(d.min_dist_km.clip(lower=5))
    d["met_missing"] = (d.surge_m.isna() | d.rain_mm.isna()).astype(float)
    d["surge_m"] = d.surge_m.fillna(0); d["rain_mm"] = d.rain_mm.fillna(0)
    for g in INTS[1:]:
        d["i_" + g[:3]] = (d.intensity == g).astype(float)
    d["gulf"] = (d.region == "Gulf").astype(float)
    d["river"] = (d.layout == "River / inland").astype(float)
    return d


TR = PT.copy()
TR["log_co2_day"] = np.log(TR["all"] / TR.days)
# baseline traffic: normal ship-hours per day over each port's events (from event-level normals)
hrs = E[E.window == "event30"].assign(hd=lambda x: x.exp_hours / x.days).groupby("port").hd.mean()
wait_h = pd.read_csv(os.path.join(OUT, "decoupling_events.csv"))
wait_h = wait_h[(wait_h.window == "event30") & wait_h.zone.isin(["anchorage", "offshore_hold"])].assign(hd=lambda x: x.exp_hours / x.days)
wait_h = wait_h.groupby(["port", "SID"]).hd.sum().groupby("port").mean()
TR["log_hours_day"] = np.log(hrs.reindex(TR.index))
TR["log_wait_hours_day"] = np.log1p(wait_h.reindex(TR.index).fillna(0))
PORTX = ["wait_sh", "log_co2_day", "log_hours_day", "log_wait_hours_day", "tanker_sh", "passenger_sh", "other_sh", "open_sh"]
STORM = ["wind_at_closest_kt", "i_Str", "i_Hur", "log_dist", "k_days", "surge_m", "rain_mm", "landfall", "met_missing"]
SPECS = [("1 waiting share only", ["wait_sh"]),
         ("2 + storm (wind, class, distance, duration, surge, rain, landfall)", ["wait_sh"] + STORM),
         ("3 + port size and baseline traffic", ["wait_sh"] + STORM + ["log_co2_day", "log_hours_day"]),
         ("4 + fleet mix (tanker, passenger, service-craft shares)", ["wait_sh"] + STORM + ["log_co2_day", "log_hours_day", "tanker_sh", "passenger_sh", "other_sh"]),
         ("5 + Gulf", ["wait_sh"] + STORM + ["log_co2_day", "log_hours_day", "tanker_sh", "passenger_sh", "other_sh", "gulf"]),
         ("6 + geography (open-water share, river/inland)", ["wait_sh"] + STORM + ["log_co2_day", "log_hours_day", "tanker_sh", "passenger_sh", "other_sh", "gulf", "open_sh", "river"]),
         ("7 + year", ["wait_sh"] + STORM + ["log_co2_day", "log_hours_day", "tanker_sh", "passenger_sh", "other_sh", "gulf", "open_sh", "river", "season"])]
OUTS = [("event30", "lc", "Whole-event CO2"), ("event30", "ln", "Whole-event NOx"), ("after", "lc", "CO2, 10 days after")]


def fit(d, y, xs, cluster="storm"):
    d = d[[y, "SID", "port"] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    X = d[xs].copy()
    sd = {c: X[c].std() for c in xs}
    for c in xs:
        if c in PORTX + ["wind_at_closest_kt", "log_dist", "k_days", "surge_m", "rain_mm", "season"]:
            X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = sm.add_constant(X, has_constant="add")
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    gs, gp = pd.factorize(d.SID)[0], pd.factorize(d.port)[0]
    groups = {"storm": gs, "port": gp, "two-way": np.column_stack([gs, gp])}[cluster]
    r = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return r, len(d), sd


A = []
for w, y, lab in OUTS:
    d = design(E[E.window == w].merge(TR[[c for c in PORTX if c not in ("wait_sh", "tanker_sh", "passenger_sh", "other_sh", "open_sh")]],
                                      left_on="port", right_index=True).merge(PT[["wait_sh", "tanker_sh", "passenger_sh", "other_sh", "open_sh"]], left_on="port", right_index=True))
    for spec, xs in SPECS:
        for cl in ["storm", "port", "two-way"]:
            r, n, _ = fit(d, y, xs, cl)
            ci = r.conf_int().loc["wait_sh"]
            A.append(dict(outcome=lab, spec=spec, cluster=cl, coef=r.params["wait_sh"], lo=ci[0], hi=ci[1], p=r.pvalues["wait_sh"], n=n))
    # alternative measure: absolute waiting capacity with size control
    xs = ["log_wait_hours_day"] + STORM + ["log_co2_day", "log_hours_day", "tanker_sh", "passenger_sh", "other_sh", "gulf", "open_sh", "river", "season"]
    for cl in ["storm", "port", "two-way"]:
        r, n, _ = fit(d, y, xs, cl); ci = r.conf_int().loc["log_wait_hours_day"]
        A.append(dict(outcome=lab, spec="8 absolute waiting capacity (log ship-hours/day) instead of share, full controls", cluster=cl,
                      coef=r.params["log_wait_hours_day"], lo=ci[0], hi=ci[1], p=r.pvalues["log_wait_hours_day"], n=n))
    # leave one port out (full model, two-way)
    for port in PT.sort_values("wait_sh", ascending=False).index[:8]:
        r, n, _ = fit(d[d.port != port], y, SPECS[-1][1], "two-way"); ci = r.conf_int().loc["wait_sh"]
        A.append(dict(outcome=lab, spec=f"9 without {port}", cluster="two-way", coef=r.params["wait_sh"], lo=ci[0], hi=ci[1], p=r.pvalues["wait_sh"], n=n))
    # placebo dates: same full model on storm-free years
    pdd = design(PL[PL.window == w].merge(TR[["log_co2_day", "log_hours_day", "log_wait_hours_day"]], left_on="port", right_index=True)
                 .merge(PT[["wait_sh", "tanker_sh", "passenger_sh", "other_sh", "open_sh"]], left_on="port", right_index=True))
    r, n, _ = fit(pdd, y, SPECS[-1][1], "two-way"); ci = r.conf_int().loc["wait_sh"]
    A.append(dict(outcome=lab, spec="10 placebo dates (cyclone-free years), full model", cluster="two-way", coef=r.params["wait_sh"], lo=ci[0], hi=ci[1], p=r.pvalues["wait_sh"], n=n))
A = pd.DataFrame(A); A.to_csv(os.path.join(OUT, "rev_anchorage_robustness.csv"), index=False)

# ------------------------------------------------------------------ (3) recovery arrival management
Dd = pd.read_csv(os.path.join(OUT, "panel_daily.csv.gz")); Dd = Dd[Dd.placebo_year.isna()]
EFN = {"main_big": 91.9, "aux": 52.6, "boiler": 6.6}          # tankers and cargo ships: slow-speed main engines
ENG = [("fc_main_ton", "main_big"), ("fc_AE_ton", "aux"), ("fc_boiler_ton", "boiler")]
J = pd.read_csv(os.path.join(OUT, "jit_pairs.csv")); J = J[J.window == "queue"]
PER_H = 72 * (1 - (72 / 96) ** 2) / 24
rows = []
for (sid, port), d in Dd.groupby(["SID", "port"]):
    k = int(d.k_days.iloc[0]); x = d.set_index("rel_day")
    a, b = k, k + 9
    r = dict(SID=sid, port=port, k=k)
    for kind in ("obs", "exp"):
        wz = {"co2": 0.0, "nox": 0.0, "hotel_nox": 0.0}
        for v in ("tanker", "cargo"):
            for t in ("anchorage", "offshore_hold"):
                c = f"{kind}:{v}|{t}|co2_ton"
                wz["co2"] += x[c].loc[a:b].sum() if c in x else 0.0
                for m, e in ENG:
                    c = f"{kind}:{v}|{t}|{m}"
                    val = x[c].loc[a:b].sum() * EFN[e] / 1000 if c in x else 0.0
                    wz["nox"] += val
                    if m != "fc_main_ton":
                        wz["hotel_nox"] += val
        r.update({f"{kind}_wait_{k_}": v_ for k_, v_ in wz.items()})
        # all zones, all vessels: the post-storm anomaly the intervention is measured against
        r[f"{kind}_all_co2"] = x[f"{kind}:all|total|co2_ton"].loc[a:b].sum()
        nox = 0.0
        for v in ("tanker", "cargo", "passenger", "other"):
            for t in panel.TIERS:
                for m, e in ENG:
                    c = f"{kind}:{v}|{t}|{m}"
                    if c in x:
                        ee = e if (v in ("tanker", "cargo") or m != "fc_main_ton") else "main_small"
                        nox += x[c].loc[a:b].sum() * {"main_small": 62.9, **EFN}[ee] / 1000
        r[f"{kind}_all_nox"] = nox
    r["normal_co2_day"] = x["exp:all|total|co2_ton"].loc[-2:k + 29].sum() / (k + 32)
    rows.append(r)
I = pd.DataFrame(rows).merge(P[["SID", "port", "storm", "season", "intensity"]], on=["SID", "port"])
js = J.assign(s=J.net_excess_wait_h * J.f_main_t_per_h * PER_H).groupby(["SID", "port"]).s.sum().rename("main_fuel_saved_t")
I = I.merge(js, left_on=["SID", "port"], right_index=True, how="left")
for m in ("co2", "nox", "hotel_nox"):
    I[f"x_wait_{m}"] = I[f"obs_wait_{m}"] - I[f"exp_wait_{m}"]
I["x_all_co2"] = I.obs_all_co2 - I.exp_all_co2; I["x_all_nox"] = I.obs_all_nox - I.exp_all_nox
I["co2_avoided_t"] = I.main_fuel_saved_t * 3.114; I["nox_avoided_t"] = I.main_fuel_saved_t * EFN["main_big"] / 1000
I.to_csv(os.path.join(OUT, "rev_intervention_events.csv"), index=False)
H = I[I.intensity == INTS[2]]
N_PORTDAYS = PT.shape[0] * len(pd.date_range("2015-01-01", "2023-12-31"))
EVENT_PORTDAYS = (P.k_days + 32).sum()
byport = H.groupby("port")[["x_wait_co2", "x_wait_nox", "x_all_co2", "x_all_nox", "co2_avoided_t", "nox_avoided_t", "x_wait_hotel_nox"]].sum()
byport["events"] = H.groupby("port").size()
byport = byport.sort_values("x_wait_co2", ascending=False)
pos = byport[byport.x_wait_co2 > 0]
tot = dict(x_all_co2=H.x_all_co2.sum(), x_all_nox=H.x_all_nox.sum(), x_wait_co2=H.x_wait_co2.sum(), x_wait_nox=H.x_wait_nox.sum(),
           x_wait_co2_pos_ports=pos.x_wait_co2.sum(), co2_avoided=max(H.co2_avoided_t.sum(), 0), nox_avoided=max(H.nox_avoided_t.sum(), 0),
           hotel_nox_moved=max(H.x_wait_hotel_nox.sum(), 0), normal_co2_day_sum=H.normal_co2_day.sum(), n_events=len(H))
C = []
for kport in [1, 2, 3, 5, 8, len(pos)]:
    sel = pos.head(kport)
    C.append(dict(ports=kport, port_list="; ".join(sel.index), intervention_port_days=int(sel.events.sum() * 10),
                  share_all_port_days=sel.events.sum() * 10 / N_PORTDAYS, share_storm_event_port_days=sel.events.sum() * 10 / EVENT_PORTDAYS,
                  share_of_excess_waiting_co2=sel.x_wait_co2.sum() / pos.x_wait_co2.sum(),
                  share_of_post_storm_excess_co2=sel.x_wait_co2.sum() / tot["x_all_co2"],
                  share_of_post_storm_excess_nox=sel.x_wait_nox.sum() / tot["x_all_nox"],
                  co2_avoided_t=max(sel.co2_avoided_t.sum(), 0), nox_avoided_t=max(sel.nox_avoided_t.sum(), 0)))
C = pd.DataFrame(C); C.to_csv(os.path.join(OUT, "rev_intervention_concentration.csv"), index=False)
pd.Series(tot).to_csv(os.path.join(OUT, "rev_intervention_totals.csv"))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(15, 4.4))
gs = fig.add_gridspec(1, 4, wspace=.36)
xs_ = np.linspace(-1.6, 1.0, 300)
for j, t in enumerate(TYPES):
    ax = fig.add_subplot(gs[0, j])
    s = E[(E.window == "storm") & (E.port_type == t)].g.dropna(); p = PL[(PL.window == "storm") & (PL.port_type == t)].g.dropna()
    kp, ks = stats.gaussian_kde(p.clip(-2, 1.5)), stats.gaussian_kde(s.clip(-2, 1.5))
    ax.fill_between(xs_, kp(xs_), color="#c3c2b9", alpha=.8, lw=0, label=f"placebo dates, no cyclone (n={len(p)})")
    ax.plot(xs_, ks(xs_), color=TCOL[t], lw=2, label=f"cyclone events (n={len(s)})")
    ax.plot(s.clip(-1.6, 1), np.full(len(s), -0.06 * ax.get_ylim()[1]), "|", color=TCOL[t], ms=6, alpha=.6)
    ax.axvline(0, color="#52514e", lw=.7); ax.axvline(s.mean(), color=TCOL[t], lw=1.2, ls="--")
    r = D[(D.window == "storm") & (D.port_type == t) & (D.intensity == "All")].iloc[0]
    ax.text(.03, .97, f"{len(s)} cyclone events, {len(p)} placebo dates\nmean shift {r['diff']:+.2f} [{r.diff_lo:+.2f} to {r.diff_hi:+.2f}]\nbeyond noise band: {r.share_storm_beyond_band:.0%} of cyclone events\nvs {r.share_placebo_beyond_band:.0%} of placebo dates",
            transform=ax.transAxes, va="top", fontsize=7, color="#0b0b0b")
    ax.set_xlabel("D = Δln CO2 − Δln ship-hours\n(storm window)"); ax.set_yticks([]); ax.set_xlim(-1.6, 1.0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)          # headroom for the text block
    ax.set_title(f"{'abcd'[j]}  {PLAB[t]} ports", loc="left", fontweight="bold")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(color="#c3c2b9", label="placebo dates in cyclone-free years"),
                    Line2D([], [], color="#52514e", lw=2, label="cyclone events (colour = port type; ticks = single events; dashed = mean)")],
           frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2)
# panel on the waiting-zone effect moved to Extended Data Fig. 6b (16_main_figures.py)
# figure title removed (caption carries it)
fig.savefig(os.path.join(FIG, "ed3_divergence_placebo.png"), dpi=200, bbox_inches="tight")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(D[(D.intensity == "All")].round(3).to_string(index=False))
print(D[(D.window == "storm") & (D.port_type == "All")].round(3).to_string(index=False))
print(A.round(3).to_string(index=False))
print(pd.Series(tot).round(1).to_string()); print(C.round(4).to_string(index=False))
