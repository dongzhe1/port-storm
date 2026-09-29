"""Main Figs 1-4, Extended Data Figs 4-6 and Supplementary Fig. 3.
Fig. 1 response before, during and after the storm window, by cyclone class      figures/fig1_response.png
Fig. 2 changes by operational zone, before / during / after                      figures/fig2_zones.png, fig2_zone_ratios.csv
Fig. 3 cyclone and port characteristics behind traffic-emissions divergence       figures/fig3_ports.png, fig3_port_coast.csv
Fig. 4 recovery arrival management: size and targeting                           figures/fig4_arrival_management.png
Extended Data Fig. 4 whole-event net change, event by event                      figures/ed4_net_change.png
Extended Data Fig. 5 zone changes as heat maps (same numbers as Fig. 2)          figures/ed5_zones_heatmap.png
Extended Data Fig. 6 what drives the response (regression)                       figures/ed6_drivers.png
Supplementary Fig. 3 air pollutants on the CO2 baseline                          figures/supp3_pollutants.png
Needs the outputs of steps 7-12.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy import stats
import panel

OUT = panel.OUT
FIG = panel.FIG
CORE = panel.CORE
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]
ILAB = ["Tropical\nstorm", "Strong trop.\nstorm", "Hurricane"]
M = [("hours", "Ship-hours (operational)", "#8a8984"), ("co2", "CO2 (climate)", "#2a78d6"), ("nox", "NOx (air pollutant)", "#d55e00"),
     ("pm25", "PM2.5 (air pollutant)", "#cc79a7")]
MC = {m: c for m, _, c in M}
TCOL = {"Energy": "#d55e00", "Cargo": "#2a78d6", "Mixed": "#e69f00", "Service": "#56b4e9"}   # CVD-validated (no red-green pair)
PLAB = {"Energy": "Energy", "Cargo": "Cargo", "Mixed": "Mixed", "Service": "Service-craft"}
WCOL = [(0, 50, "#9cc3ee"), (50, 64, "#4f8fd9"), (64, 999, "#b8431f")]
DV = pd.read_csv(os.path.join(OUT, "divergence_pooled.csv"))
PT = pd.read_csv(os.path.join(OUT, "het_port_traits.csv"), index_col=0)
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
note = lambda fig, y, s: fig.text(.01, y, s, fontsize=7.2, color="#52514e")
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
PCT = FuncFormatter(lambda v, _: "0%" if abs(v - 1) < 1e-9 else f"{(v - 1) * 100:+.0f}%")
M2 = [m for m in M if m[0] in ("hours", "co2")]                  # main figures: operational vs climate response


def pct_axis(ax, ticks=None, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(PCT)


def dots(ax, x, win, g, ms=5.5, measures=M, spread=.14):
    for i, (m, lab, c) in enumerate(measures):
        r = DV[(DV.window == win) & (DV.measure == m) & (DV.intensity == g)]
        if r.empty:
            continue
        r = r.iloc[0]; xx = x + (i - (len(measures) - 1) / 2) * spread
        ax.errorbar(xx, r.ratio, yerr=[[r.ratio - r.lo], [r.hi - r.ratio]], fmt="s" if m == "hours" else "o", color=c, ms=ms,
                    elinewidth=1, capsize=2, markeredgecolor="#fcfcfb", markeredgewidth=.7)


# =========================================================== Fig. 1 (periods by class + wind bins)
E = pd.read_csv(os.path.join(OUT, "decoupling_events.csv")); E = E[E.zone == "total"].merge(PT[["port_type"]], left_on="port", right_index=True)
PL = pd.read_csv(os.path.join(OUT, "decoupling_placebo_classified.csv")); PL = PL[PL.zone == "total"].merge(PT[["port_type"]], left_on="port", right_index=True)
Dt = pd.read_csv(os.path.join(OUT, "rev_D_distributions.csv"))
wins1 = [("prep", "2 days\nbefore"), ("storm", "Storm\nwindow"), ("after", "10 days\nafter"), ("event30", "Whole event\n(−2 to +30 d)")]
NCL = P.intensity.value_counts()
TT1 = [f"a  Tropical storms (<50 kt, {NCL[INTS[0]]} events)", f"b  Strong tropical storms (50–63 kt, {NCL[INTS[1]]})", f"c  Hurricanes (≥64 kt, {NCL[INTS[2]]})"]
X1 = {"prep": 0, "storm": 1, "after": 2, "event30": 3.25}
M1 = [("hours", "Ship-hours (operational)", MC["hours"], "s", "--", -.07), ("co2", "CO2 (climate)", MC["co2"], "o", "-", .07)]
with plt.rc_context({"font.size": 9.5}):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, gridspec_kw=dict(wspace=.08))
    for ax, g, t in zip(axs, INTS, TT1):
        ax.axvspan(.55, 1.45, color="#efeee9", zorder=0); ax.axvline(2.63, color="#c3c2b9", lw=.8, ls=":", zorder=0); ax.axhline(1, color="#8a8984", lw=.8, zorder=0)
        for m, lab, c, mk, ls, dx in M1:
            r = DV[(DV.measure == m) & (DV.intensity == g)].set_index("window"); seq = ["prep", "storm", "after"]
            ax.plot([X1[w] + dx for w in seq], r.loc[seq, "ratio"], ls=ls, color=c, lw=1.6, zorder=2)
            for w in X1:
                rr = r.loc[w]
                ax.errorbar(X1[w] + dx, rr.ratio, yerr=[[rr.ratio - rr.lo], [rr.hi - rr.ratio]], fmt=mk, color=c, ms=7, elinewidth=1.1, capsize=2.5,
                            markeredgecolor="#fcfcfb", markeredgewidth=.8, zorder=3)
            v, hi = r.loc["storm", "ratio"], r.loc["storm", "hi"]; txt = f"{(v - 1) * 100:+.0f}%".replace("-", "\u2212")
            if m == "co2":
                ax.annotate(txt, (X1["storm"] + dx, v), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=c, fontweight="bold")
            else:
                ax.annotate(txt, (X1["storm"] + dx, hi), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#52514e", fontweight="bold")
        ax.set_xticks(list(X1.values())); ax.set_xticklabels([l for _, l in wins1], fontsize=8.5); ax.set_xlim(-.45, 3.7)
        ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
    axs[0].set_ylim(.3, 1.28); pct_axis(axs[0], [.4, .6, .8, 1, 1.2]); axs[0].set_ylabel("change from no-cyclone normal")
    axs[0].legend(handles=[Line2D([], [], color=c, ls=ls, marker=mk, ms=6, label=lab) for _, lab, c, mk, ls, _ in M1], frameon=False, fontsize=8.5, loc="lower left")
    # figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "fig1_response.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# =========================================================== Fig. 2 (zone profiles: before / during / after)
TP = pd.read_csv(os.path.join(OUT, "pollutant_tier_pairs.csv"))
TIERS = [("port", "Berth"), ("channel", "Channel"), ("anchorage", "Anchor-\nage"), ("offshore_hold", "Offshore\nhold"), ("offshore", "Open\nwater")]
MK = [("hours", "s", "--"), ("co2", "o", "-")]
PH = [("prep", "Two days before"), ("storm", "During the storm window"), ("after", "Ten days after")]
rng2 = np.random.default_rng(7); NB = 1000
cols = [p + "_" + m for m in MC for p in ("o", "e")]
Z = TP.groupby(["intensity", "phase", "SID", "tier"])[cols].sum()
zrows = []
for g in INTS:
    sids = TP[TP.intensity == g].SID.unique()
    for ph, _ in PH:
        A = np.zeros((len(sids), len(TIERS), len(cols)))
        z = Z.loc[(g, ph)]
        for i, s in enumerate(sids):
            if s in z.index.get_level_values(0):
                zz = z.loc[s]
                for j, (t, _) in enumerate(TIERS):
                    if t in zz.index:
                        A[i, j] = zz.loc[t].values
        tot = A.sum(0); bs = A[rng2.integers(0, len(sids), size=(NB, len(sids)))].sum(1)
        for j, (t, _) in enumerate(TIERS):
            for m in MC:
                io, ie = cols.index("o_" + m), cols.index("e_" + m)
                with np.errstate(all="ignore"):
                    r = tot[j, io] / tot[j, ie]; b = bs[:, j, io] / bs[:, j, ie]
                lo, hi = np.nanpercentile(b, [2.5, 97.5])
                zrows.append(dict(intensity=g, phase=ph, tier=t, measure=m, ratio=r, lo=lo, hi=hi, e_total=tot[j, ie]))
ZR = pd.DataFrame(zrows)
share = ZR[ZR.measure == "co2"].groupby(["intensity", "tier"]).e_total.sum()
share = share / share.groupby(level=0).transform("sum")
ZR["co2_share"] = [share.loc[(a, b)] for a, b in zip(ZR.intensity, ZR.tier)]
ZR.to_csv(os.path.join(OUT, "fig2_zone_ratios.csv"), index=False)

fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(3, 3, height_ratios=[.55, 1, 1], hspace=.16, wspace=.06)
x = np.arange(len(TIERS))
for r_, (ph, pl) in enumerate(PH):
    for c_, (g, gl) in enumerate(zip(INTS, [f"Tropical storms ({NCL[INTS[0]]} events)", f"Strong tropical storms ({NCL[INTS[1]]})", f"Hurricanes ({NCL[INTS[2]]})"])):
        ax = fig.add_subplot(gs[r_, c_])
        if ph == "storm":
            ax.set_facecolor("#f4f3ef")
        for i, (m, mk, ls) in enumerate(MK):
            d = ZR[(ZR.intensity == g) & (ZR.phase == ph) & (ZR.measure == m)].set_index("tier").loc[[t for t, _ in TIERS]]
            xx = x + (i - .5) * .12
            ax.plot(xx, d.ratio, ls=ls, color=MC[m], lw=.9, alpha=.45, zorder=1)
            ax.vlines(xx, d.lo.clip(lower=.12), d.hi.clip(upper=2.2), color=MC[m], lw=.9, alpha=.8, zorder=2)
            ax.scatter(xx, d.ratio, s=12 + 170 * d.co2_share.values, marker=mk, color=MC[m], edgecolor="#fcfcfb", linewidth=.5, zorder=3)
        ax.axhline(1, color="#8a8984", lw=.8, zorder=0); ax.axvline(1.5, color="#c3c2b9", lw=.8, ls=":", zorder=0)
        ax.set_yscale("log"); ax.minorticks_off()
        if ph == "prep":
            ax.set_ylim(.75, 1.33); pct_axis(ax, [.8, 1, 1.25])
        else:
            ax.set_ylim(.12, 2.2); pct_axis(ax, [.25, .5, 1, 1.5, 2])
        if c_ > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(f"{pl}\nchange from normal")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in TIERS] if r_ == 2 else [], fontsize=7.5)
        if r_ == 0:
            ax.set_title(gl, fontweight="bold", fontsize=9)
            if c_ == 0:
                ax.text(.75, 1.27, "port area", ha="center", fontsize=7, color="#52514e"); ax.text(3, 1.27, "outside the port area", ha="center", fontsize=7, color="#52514e")
        ax.text(.02, .06, "abcdefghi"[r_ * 3 + c_], transform=ax.transAxes, fontweight="bold")
fig.legend(handles=[Line2D([], [], marker=mk, ls=ls, color=MC[m], label=lab) for (m, mk, ls), (_, lab, _) in zip(MK, M2)],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .95))
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "fig2_zones.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# =========================================================== Extended Data: zone heat maps (same numbers as Fig. 2)
TP = pd.read_csv(os.path.join(OUT, "pollutant_tier_pairs.csv"))
fig = plt.figure(figsize=(15, 4.4))
gs = fig.add_gridspec(1, 3, wspace=.06)
TIERS = [("port", "Berth"), ("channel", "Channel"), ("anchorage", "Anchor-\nage"), ("offshore_hold", "Offshore\nhold"), ("offshore", "Open\nwater")]
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8431f"])
for col, (win, title) in enumerate([("prep", "a  Two days before"), ("storm", "b  During the storm window"), ("after", "c  Ten days after")]):
    ax = fig.add_subplot(gs[0, col]); rows, labels = [], []
    for g, gl in zip(INTS, ["TS", "Strong TS", "Hurricane"]):
        d = TP[(TP.intensity == g) & (TP.phase == win)].groupby("tier").sum(numeric_only=True)
        for mm, ml in [("hours", "ship-hours"), ("co2", "CO2")]:
            rows.append([d.loc[t, "o_" + mm] / d.loc[t, "e_" + mm] if t in d.index and d.loc[t, "e_" + mm] > 0 else np.nan for t, _ in TIERS]); labels.append(f"{gl} · {ml}")
    A = np.array(rows); im = ax.imshow(np.log2(A), cmap=cmap, norm=TwoSlopeNorm(0, -1.5, 1.5), aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                ax.text(j, i, "0%" if round((A[i, j] - 1) * 100) == 0 else f"{(A[i, j] - 1) * 100:+.0f}%", ha="center", va="center", fontsize=7.2, color="#fcfcfb" if abs(np.log2(A[i, j])) > .9 else "#0b0b0b")
    ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in TIERS], fontsize=7.5); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels if col == 0 else [], fontsize=7.2)
    for kk in (1.5, 3.5):
        ax.axhline(kk, color="#fcfcfb", lw=2)
    ax.spines[:].set_visible(False); ax.tick_params(length=0); ax.set_title(title, loc="left", fontweight="bold")
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "ed5_zones_heatmap.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# =========================================================== Fig. 3 (which ports and storms drive the differences)
CC = pd.read_csv(os.path.join(OUT, "het_class_counts.csv"))
R = pd.read_csv(os.path.join(OUT, "het_determinants.csv"))
TW = pd.read_csv(os.path.join(OUT, "port_window_pairs.csv")).merge(PT[["port_type", "region"]], left_on="port", right_index=True)
rng3 = np.random.default_rng(8)


def pooled_ci(d, m, B=1000):
    g = d.groupby("SID")[["obs_" + m, "exp_" + m]].sum().values
    b = g[rng3.integers(0, len(g), (B, len(g)))].sum(1); lo, hi = np.percentile(b[:, 0] / b[:, 1], [2.5, 97.5])
    return g[:, 0].sum() / g[:, 1].sum(), lo, hi


ES = pd.read_csv(os.path.join(OUT, "decoupling_events.csv")); ES = ES[(ES.zone == "total") & (ES.window == "storm") & (ES.cls_co2 != "no data")]
with plt.rc_context({"font.size": 10}):
    CLS = [("no response", "No detectable change: neither exceeded normal variation", "#dcdad4"), ("coupled", "Aligned response: traffic and CO2 changed together", "#6da7ec"),
           ("relative (amplified)", "CO2-dominant divergence: CO2 fell further than traffic", "#b8431f"), ("relative (damped)", "Traffic-dominant divergence: traffic changed more than CO2", "#184f95")]
    CC_ = {k: c for k, _, c in CLS}
    fig = plt.figure(figsize=(15, 12)); gs = fig.add_gridspec(2, 5, height_ratios=[1.25, .95], hspace=.5, wspace=.12)
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=.42)
    # ---- a scatter
    ax = fig.add_subplot(top[0])
    x, y = np.exp(ES.lh.clip(-2.6, .8)), np.exp(ES.lc.clip(-2.6, .8))
    lim = (.07, 2.3)
    xx = np.array(lim); ax.fill_between(xx, lim[0], xx, color="#b8431f", alpha=.06, lw=0, zorder=0)
    ax.plot(xx, xx, color="#52514e", lw=1, ls="--", zorder=1); ax.axhline(1, color="#8a8984", lw=.7, zorder=0); ax.axvline(1, color="#8a8984", lw=.7, zorder=0)
    SZ = {INTS[0]: 18, INTS[1]: 34, INTS[2]: 60}
    for k, _, c in CLS:
        d = ES[E.cls_co2 == k]
        ax.scatter(np.exp(d.lh.clip(-2.6, .8)), np.exp(d.lc.clip(-2.6, .8)), s=d.intensity.map(SZ), c=c, alpha=.55 if k == "no response" else .9,
                   edgecolor="#fcfcfb", linewidth=.5, zorder=2 if k == "no response" else 3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator([.1, .25, .5, 1, 2])); a.set_major_formatter(PCT); a.set_minor_locator(NullLocator())
    ax.set_xlabel("change in ship-hours during the storm window (operational)"); ax.set_ylabel("change in CO2 during the storm window (climate)")
    det = ES[E.cls_co2 != "no response"]; n_det = len(det); nA = (ES.cls_co2 == "relative (amplified)").sum(); nD = (ES.cls_co2 == "relative (damped)").sum(); nC = (ES.cls_co2 == "coupled").sum()
    ax.text(.97, .04, f"CO2 fell further than traffic: {nA} ({nA / n_det:.0%})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#b8431f", fontweight="bold")
    ax.text(.03, .97, f"Traffic changed more than CO2: {nD} ({nD / n_det:.0%})", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#184f95", fontweight="bold")
    ax.text(.03, .92, f"Aligned: {nC} ({nC / n_det:.0%})", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#3f7fc9", fontweight="bold")
    ax.text(.03, .87, f"(shares of {n_det} events with a detectable change;\n{len(ES) - n_det} of {len(ES)} events, grey, showed none)", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#52514e")
    ax.text(.64, .69, "1:1", transform=ax.transAxes, rotation=45, fontsize=8.5, color="#52514e", ha="center", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#8a8984", ms=np.sqrt(s_), label=l) for s_, l in zip(SZ.values(), ["Tropical storm", "Strong tropical storm", "Hurricane"])],
              frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, .76), title="marker size", title_fontsize=8, alignment="left")
    ax.set_title("a  Each event during the storm window", loc="left", fontweight="bold")
    # ---- b coupled vs decoupled, back to back (undetectable events given as text)
    ax = fig.add_subplot(top[1])
    rows = [("intensity", INTS[0], "Tropical storms"), ("intensity", INTS[1], "Strong tropical storms"), ("intensity", INTS[2], "Hurricanes"), None,
            ("port_type", "Cargo", "Cargo ports"), ("port_type", "Mixed", "Mixed ports"), ("port_type", "Energy", "Energy ports"), ("port_type", "Service", "Service-craft ports")]
    y = 0; yt, yl = [], []
    for r in rows:
        if r is None:
            y += .7; continue
        x_ = CC[(CC.grouping == r[0]) & (CC.level == r[1]) & (CC.window == "storm")].iloc[0]; n = x_.n - x_["no data"]
        cpl, amp, dmp, non = x_["coupled"] / n, x_["relative (amplified)"] / n, x_["relative (damped)"] / n, x_["no response"] / n
        ax.barh(y, -cpl, color="#6da7ec", height=.66); ax.barh(y, amp, color="#b8431f", height=.66)
        ax.barh(y, dmp, left=amp, color="#184f95", height=.66, edgecolor="#fcfcfb", linewidth=1.5)
        ax.text(-cpl - .01, y, f"{cpl:.0%}", ha="right", va="center", fontsize=9); ax.text(amp + dmp + .01, y, f"{amp + dmp:.0%}", ha="left", va="center", fontsize=9)
        ax.text(.62, y, f"{non:.0%}", ha="right", va="center", fontsize=9, color="#8a8984")
        ax.text(.70, y, f"n={int(n)}", ha="left", va="center", fontsize=8.5, color="#8a8984"); yt.append(y); yl.append(r[2]); y += 1
    ax.axvline(0, color="#52514e", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl); ax.invert_yaxis(); ax.set_xlim(-.5, .8)
    ax.set_xticks([-.4, -.2, 0, .2, .4]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.text(-.25, -1.05, "\u25c0 aligned", ha="center", fontsize=9.5, color="#3f7fc9", fontweight="bold")
    ax.text(.22, -1.05, "divergent \u25b6", ha="center", fontsize=9.5, color="#b8431f", fontweight="bold")
    ax.text(.62, -1.05, "no detectable\nchange", ha="right", fontsize=8.5, color="#8a8984")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#6da7ec", "#b8431f", "#184f95"]],
              labels=["Aligned response: traffic and CO2 changed together", "CO2-dominant divergence: CO2 fell further than traffic", "Traffic-dominant divergence: traffic changed more than CO2"],
              frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, handlelength=1.2, columnspacing=1.2)
    ax.set_xlabel("share of all events, during the storm window")
    ax.set_title("b  Aligned vs divergent responses by cyclone strength and port type", loc="left", fontweight="bold", y=1.28)
    # ---- c one panel: coast x port type, CO2 dots + ship-hours dashes
    W3 = [("prep", "2 days before"), ("storm", "Storm window"), ("after", "10 days after"), ("event30", "Whole event")]
    S3 = TW[TW.intensity.isin(INTS[1:])]
    panels = [("Gulf", "Energy"), ("Gulf", "Mixed"), ("Gulf", "Service"), ("Atlantic", "Cargo"), ("Atlantic", "Mixed")]
    ax = fig.add_subplot(gs[1, :]); ax.axvspan(.5, 1.5, color="#efeee9", zorder=0); ax.axhline(1, color="#8a8984", lw=.9, zorder=0)
    off = np.linspace(-.3, .3, len(panels)); hand, out3 = [], []
    for i, (rg, pt) in enumerate(panels):
        d = S3[(S3.region == rg) & (S3.port_type == pt)]; c = TCOL[pt]; fc = c if rg == "Gulf" else "#fcfcfb"
        for j, (w, _) in enumerate(W3):
            rc, lo, hi = pooled_ci(d[d.window == w], "co2"); rh, hlo, hhi = pooled_ci(d[d.window == w], "hours")
            x0 = j + off[i]
            ax.plot([x0 - .045, x0 + .045], [rh, rh], color=c, lw=2.6, solid_capstyle="butt", zorder=2)
            out3 += [dict(region=rg, port_type=pt, window=w, measure="co2", n_events=d[d.window == w].shape[0], n_ports=d.port.nunique(), ratio=rc, lo=lo, hi=hi),
                     dict(region=rg, port_type=pt, window=w, measure="hours", n_events=d[d.window == w].shape[0], n_ports=d.port.nunique(), ratio=rh, lo=hlo, hi=hhi)]
            ax.errorbar(x0, rc, yerr=[[rc - lo], [hi - rc]], fmt="o", color=c, markerfacecolor=fc, markeredgecolor=c, ms=7, mew=1.5, elinewidth=1.2, capsize=2, zorder=3)
        hand.append(Line2D([], [], marker="o", ls="", color=c, markerfacecolor=fc, markeredgecolor=c, mew=1.5, ms=7,
                           label=f"{rg} {PLAB[pt].lower()} ports ({d[d.window == 'storm'].shape[0]} events, {d.port.nunique()} ports)"))
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in W3]); ax.set_xlim(-.5, 3.5)
    ax.set_ylim(.25, 1.3); ax.yaxis.set_major_locator(FixedLocator([.4, .6, .8, 1, 1.2])); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("change from no-cyclone normal")
    hand += [Line2D([], [], marker="o", ls="", color="#52514e", ms=7, label="dot = CO2 (filled Gulf, open Atlantic)"), Line2D([], [], color="#52514e", lw=2.6, label="dash = ship-hours")]
    ax.legend(handles=hand, frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, columnspacing=1.5)
    ax.set_title("c  Ship-hours vs CO2 by port type and coast (strong tropical storms and hurricanes)", loc="left", fontweight="bold", y=1.2)
    fig.savefig(os.path.join(FIG, "fig3_ports.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
pd.DataFrame(out3).to_csv(os.path.join(OUT, "fig3_port_coast.csv"), index=False)

# ----------------------------------------------------------- Extended Data Fig. 6: regression of event-level changes on cyclone and port traits
# (the waiting-zone robustness panel b was dropped; rev_anchorage_robustness.csv is still written by the analysis)
fig, ax = plt.subplots(figsize=(8.2, 5.8))
terms = [("wind_at_closest_kt", "Wind at closest approach"), ("log_dist", "Distance (log)"), ("k_days", "Duration"), ("tanker_sh", "Tanker share"),
         ("other_sh", "Service-craft share"), ("passenger_sh", "Passenger share"), ("wait_sh", "Waiting-zone share"), ("open_sh", "Open-water share"), ("port_size", "Port size")]
outs = [("CO2, storm window", "CO2 during storm window", MC["co2"]), ("Decoupling gap, storm window", "Divergence gap during storm window", "#8a8984"),
        ("CO2, 10 days after", "CO2, 10 days after", "#6da7ec"), ("CO2, whole event", "CO2, whole event", "#184f95")]
r_ = R[R.spec == "OLS"]
for j, (o, lab, c) in enumerate(outs):
    for i, (t, _) in enumerate(terms):
        x = r_[(r_.outcome == o) & (r_.term == t)].iloc[0]; yy = i + (j - 1.5) * .19
        ax.plot([100 * x.lo, 100 * x.hi], [yy, yy], color=c, lw=1.2); ax.plot(100 * x.coef, yy, "o", color=c, ms=4, markerfacecolor=c if x.p < .05 else "#fcfcfb", label=lab if i == 0 else None)
ax.axvline(0, color="#8a8984", lw=.8); ax.set_yticks(range(len(terms))); ax.set_yticklabels([l for _, l in terms]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
ax.set_xlabel("change per 1 SD, percentage points (filled: p < 0.05)")
ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=4, lw=1.2, label=lab) for _, lab, c in outs], frameon=False, fontsize=7, loc="lower left")
# single panel: no title (the caption carries it)
fig.savefig(os.path.join(FIG, "ed6_drivers.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# =========================================================== Fig. 4 (recovery arrival management)
SC = pd.read_csv(os.path.join(OUT, "jit_scenarios.csv")); J = pd.read_csv(os.path.join(OUT, "het_jit_by_port.csv"), index_col=0).iloc[:, 0]
IC = pd.read_csv(os.path.join(OUT, "rev_intervention_events.csv")); IC = IC[IC.intensity == INTS[2]]
N_PD = PT.shape[0] * len(pd.date_range("2015-01-01", "2023-12-31"))
fig = plt.figure(figsize=(15.5, 5.4)); gs = fig.add_gridspec(1, 2, wspace=.2, width_ratios=[.8, 1.3])
ax = fig.add_subplot(gs[0, 0])
nev = 67; Wh = [12, 24, 48]
for T, c, mk in [(24, "#9cc3ee", "s"), (72, "#2a78d6", "o"), (120, "#184f95", "D")]:
    r = SC[(SC.window == "queue") & (SC.intensity == INTS[2]) & (SC.T_h == T) & (SC.W_h.astype(str).isin([str(w) for w in Wh])) & (SC.fuel_price == 650)].assign(Wn=lambda x: x.W_h.astype(int)).sort_values("Wn")
    xx = np.arange(3) + {24: -.08, 72: 0, 120: .08}[T]
    ax.errorbar(xx, r.co2_saved_t / nev / 1e3, yerr=[(r.co2_saved_t - r.co2_lo) / nev / 1e3, (r.co2_hi - r.co2_saved_t) / nev / 1e3], fmt=mk + "-", color=c, ms=6,
                capsize=2, elinewidth=1, lw=1.5, markeredgecolor="#fcfcfb", label=f"slow down over a {T}-hour approach")
r0 = SC[(SC.window == "queue") & (SC.intensity == INTS[2]) & (SC.T_h == 72) & (SC.W_h.astype(str) == "24") & (SC.fuel_price == 650)].iloc[0]
ax.annotate(f"central case:\n{round(r0.co2_saved_t / nev, -2):,.0f} t per event,\n{round(r0.co2_saved_t, -4):,.0f} t over {nev} events", (1, r0.co2_saved_t / nev / 1e3), xytext=(28, 34),
            textcoords="offset points", fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#52514e", lw=.7))
ax.set_xticks(range(3)); ax.set_xticklabels([f"{w} h" for w in Wh]); ax.set_xlim(-.4, 2.4); ax.set_ylim(0, None)
ax.set_xlabel("reduction in each queued ship's wait at anchor"); ax.set_ylabel("CO2 avoided per hurricane–port event (thousand t)")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title("a  How much CO2 slower approaches would save", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
bp = IC.groupby("port")[["x_wait_co2"]].sum().join(IC.groupby("port").size().rename("ev")).sort_values("x_wait_co2", ascending=False); bp = bp[bp.x_wait_co2 > 0]
JQ = pd.read_csv(os.path.join(OUT, "jit_pairs.csv")); JQ = JQ[(JQ.intensity == INTS[2]) & (JQ.window == "queue")].groupby("port").net_excess_wait_h.sum()
cum = bp.x_wait_co2.cumsum() / bp.x_wait_co2.sum(); days = (bp.ev * 10).cumsum() / N_PD * 100
cumh = JQ.reindex(bp.index).fillna(0).cumsum() / JQ[JQ > 0].sum()          # waiting hours covered by the same ports, same order
shp = bp.x_wait_co2 / bp.x_wait_co2.sum(); shh = JQ.reindex(bp.index).fillna(0) / JQ[JQ > 0].sum()
short = lambda p: p.replace(" Port District", "").replace(" Port Authority", "").replace(" Harbor District", "").replace("Port of Greater ", "").replace("Port of ", "").split(",")[0]
ax.plot(np.r_[0, days.values], np.r_[0, 100 * cumh.values], "--", color="#52514e", lw=1.2, zorder=1)
ax.scatter(days.values, 100 * cumh.values, s=16, facecolor="#fcfcfb", edgecolor="#52514e", linewidth=.8, zorder=2)
ax.plot(np.r_[0, days.values], np.r_[0, 100 * cum.values], "-", color="#8a8984", lw=1.4, zorder=1)
ax.scatter(days.values, 100 * cum.values, s=40, c=[TCOL[PT.loc[p, "port_type"]] for p in bp.index], edgecolor="#fcfcfb", linewidth=.6, zorder=3)
ax.text(.225, 80, "Largest ports: share of excess waiting CO2, hours", fontsize=7.6, fontweight="bold", va="center")
for i in range(8):
    ax.annotate(str(i + 1), (days.iloc[i], 100 * cum.iloc[i]), xytext=(5, -11), textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.text(.225, 30.6 + i * 6.2, f"{i + 1}. {short(bp.index[i])}: {100 * shp.iloc[i]:.0f}% CO2, {100 * shh.iloc[i]:.0f}% hours", fontsize=7.4, va="center",
            color={"Mixed": "#8a6100", "Service": "#1f6f9e"}.get(PT.loc[bp.index[i], "port_type"], TCOL[PT.loc[bp.index[i], "port_type"]]))
x5, y5, h5 = days.iloc[4], 100 * cum.iloc[4], 100 * cumh.iloc[4]
ax.plot([0, x5], [y5, y5], ls=(0, (4, 3)), color="#0b0b0b", lw=.9, zorder=0); ax.plot([x5, x5], [0, y5], ls=(0, (4, 3)), color="#0b0b0b", lw=.9, zorder=0)
ax.text(x5 + .004, 3, f"{x5:.2f}%", fontsize=8.5, fontweight="bold"); ax.text(.003, y5 + 1.5, f"{y5:.0f}%", fontsize=8.5, fontweight="bold")
ax.text(.225, 12, f"5 ports, {int(bp.ev.head(5).sum() * 10)} port-days = {x5:.2f}% of all port-days,\ncover {y5:.0f}% of excess waiting CO2 and {h5:.0f}% of excess waiting hours", fontsize=7.8, fontweight="bold", va="bottom")
h1 = [Line2D([], [], marker="o", ls="", color=TCOL[t], markeredgecolor="#fcfcfb", ms=7, label=f"{PLAB[t]} port") for t in ["Energy", "Mixed", "Service", "Cargo"] if any(PT.loc[bp.index, "port_type"] == t)]
h2 = [Line2D([], [], color="#8a8984", lw=1.4, label="excess waiting CO2 (dots by port type)"), Line2D([], [], color="#52514e", lw=1.2, ls="--", marker="o", ms=4, markerfacecolor="#fcfcfb", label="excess waiting hours, tankers and cargo ships")]
ax.legend(handles=h2 + h1, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("intervention days (10 per hurricane at each targeted port),\n% of all port-days: 47 ports × 2015–2023"); ax.set_ylabel("% of excess post-hurricane waiting covered"); ax.set_ylim(0, 105); ax.set_xlim(0, .42)
ax.set_title(f"b  Targeting: {len(bp)} ports with excess waiting after hurricanes, largest CO2 first", loc="left", fontweight="bold")
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "fig4_arrival_management.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# =========================================================== ED: winners and losers
W1 = pd.read_csv(os.path.join(OUT, "het_winners_losers.csv"))
fig, ax = plt.subplots(figsize=(4.4, 4))
d = W1[W1.window == "event30"].set_index("measure").reindex(["ship-hours", "CO2"]); x = np.arange(2)
ax.bar(x - .2, d.sig_up, width=.38, color="#ec835a", label="net increase, observed"); ax.bar(x + .2, -d.sig_down, width=.38, color="#6da7ec", label="net decrease, observed")
ax.scatter(x - .2, d.expected_up_by_chance, marker="_", s=260, color="#0b0b0b", linewidth=1.6, zorder=3, label="expected by chance")
ax.scatter(x + .2, -d.expected_down_by_chance, marker="_", s=260, color="#0b0b0b", linewidth=1.6, zorder=3)
ax.axhline(0, color="#8a8984", lw=.8); ax.set_xticks(x); ax.set_xticklabels(["Ship-hours", "CO2"]); ax.set_ylim(-16, 28)
ax.set_yticks([-15, -10, -5, 0, 5, 10, 15, 20, 25]); ax.set_yticklabels(["15", "10", "5", "0", "5", "10", "15", "20", "25"])
ax.set_ylabel(f"events with a detectable net change (of {len(P)})"); ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
fig.savefig(os.path.join(FIG, "ed4_net_change.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
# =========================================================== Supplementary Fig. 5: air pollutants on the CO2 baseline
IT = pd.read_csv(os.path.join(OUT, "pollutant_intensity_test.csv"))
fig = plt.figure(figsize=(15, 4.6)); gs = fig.add_gridspec(1, 4, wspace=.1, width_ratios=[1, 1, 1, 1.35])
MP = [m for m in M if m[0] in ("co2", "nox", "pm25")]
for k_, (g, t) in enumerate(zip(INTS, ["a  Tropical storms", "b  Strong tropical storms", "c  Hurricanes"])):
    ax = fig.add_subplot(gs[0, k_], sharey=None if k_ == 0 else fig.axes[0])
    ax.axvspan(.55, 1.45, color="#efeee9", zorder=0)
    for j_, (w, _) in enumerate(wins1):
        dots(ax, j_, w, g, ms=5, measures=MP, spread=.16)
    ax.axhline(1, color="#8a8984", lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in wins1], fontsize=7.5)
    if k_ == 0:
        ax.set_ylabel("change from no-cyclone normal"); ax.set_ylim(.3, 1.28); pct_axis(ax, [.4, .6, .8, 1, 1.2])
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for mm, l, c in MP], frameon=False, fontsize=7, loc="lower left")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(t, loc="left", fontweight="bold", fontsize=9)
ax = fig.add_subplot(gs[0, 3])
for i, (w, wl) in enumerate([("storm", "Storm window"), ("after", "10 days after"), ("event30", "Whole event")]):
    for j, (p, c) in enumerate([("nox", MC["nox"]), ("pm25", MC["pm25"])]):
        r = IT[(IT.window == w) & (IT.intensity == "All events") & (IT.pollutant == p)].iloc[0]; y = i + (j - .5) * .3
        ax.errorbar(r.storm, y, xerr=[[r.storm - r.storm_lo], [r.storm_hi - r.storm]], fmt="o", color=c, ms=5, capsize=2)
        ax.errorbar(r.placebo, y + .1, xerr=[[r.placebo - r.placebo_lo], [r.placebo_hi - r.placebo]], fmt="s", color="#8a8984", ms=3.5, capsize=1.5)
ax.axvline(1, color="#8a8984", lw=.8); ax.set_yticks(range(3)); ax.set_yticklabels(["Storm window", "10 days after", "Whole event"]); ax.invert_yaxis()
ax.set_xlim(.925, 1.015); pct_axis(ax, [.94, .96, .98, 1], axis="x"); ax.set_xlabel("change in emissions per tonne of CO2"); ax.yaxis.tick_right()
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MC["nox"], label="NOx, cyclone events"), Line2D([], [], marker="o", ls="", color=MC["pm25"], label="PM2.5, cyclone events"),
                   Line2D([], [], marker="s", ls="", color="#8a8984", label="placebo dates")], frameon=False, fontsize=7, loc="lower left")
ax.set_title("d  Pollutant per tonne of CO2, all events", loc="left", fontweight="bold", fontsize=9)
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "supp3_pollutants.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
