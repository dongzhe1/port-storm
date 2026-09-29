"""Extended Data Fig. 1 (robustness: placebo dates and the choice of normal) and Extended Data Fig. 7
(time and carbon per port call). Reads robust_*.csv (step 6) and efficiency_*.csv (step 4).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from panel import OUT, FIG
NEV = len(pd.read_csv(os.path.join(OUT, "panel_pairs.csv")))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]
ICOL = dict(zip(INTS, ["#86b6ef", "#3987e5", "#184f95"]))        # one hue, light -> dark = weaker -> stronger
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#d55e00", "#56b4e9", "#e69f00", "#8a8984"   # CVD-validated
TXT2 = "#52514e"


def ci_bar(ax, x, v, lo, hi, color, w=0.25, label=None):
    ax.bar(x, v - 1, bottom=1, width=w, color=color, label=label, zorder=2)
    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#0b0b0b", elinewidth=0.9, capsize=2, zorder=3)


# ------------------------------------------------------------------ Figure E: efficiency
O = pd.read_csv(os.path.join(OUT, "efficiency_pooled.csv"))
EQ = pd.read_csv(os.path.join(OUT, "efficiency_engine_queue.csv"))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1], wspace=.45))
WINS = [("storm", "Storm\nwindow"), ("queue", "10 days\nafter"), ("event30", "Whole event\n(−2 to +30 d)")]
for ax, metric, title in [(axs[0], "waiting hours per call", "a  Waiting time per port call"),
                          (axs[1], "CO2 per port call", "b  CO2 per port call")]:
    for i, (w, lab) in enumerate(WINS):
        for j, g in enumerate(INTS):
            r = O[(O.metric == metric) & (O.window == w) & (O.intensity == g)].iloc[0]
            ci_bar(ax, i + (j - 1) * .27, r.ratio, r.lo, r.hi, ICOL[g], label=g if i == 0 else None)
    ax.axhline(1, color=GRAY, lw=.8); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in WINS])
    ax.set_ylabel("change from normal"); ax.set_title(title, loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0%" if abs(v - 1) < 1e-9 else f"{(v - 1) * 100:+.0f}%"))
axs[0].legend(frameon=False, fontsize=8, loc="upper right")
axs[1].set_ylim(0.6, 1.6)
ax = axs[2]
rows = EQ[EQ.intensity == "Hurricane ≥64 kt"].set_index(["vclass", "tier"])
pick = [("tanker", "anchorage", "Tankers · anchorage"), ("tanker", "offshore_hold", "Tankers · offshore hold"),
        ("cargo", "anchorage", "Cargo · anchorage"), ("cargo", "offshore_hold", "Cargo · offshore hold")]
pick = [p for p in pick if (p[0], p[1]) in rows.index and rows.loc[(p[0], p[1]), "hours_ratio"] > 1]   # only tiers that gained hours
y = np.arange(len(pick))[::-1]
left = np.zeros(len(pick))
for eng, col, lab in [("main", BLUE, "Main engine"), ("aux", AQUA, "Auxiliary engines"), ("boiler", ORANGE, "Boilers")]:
    vals = np.array([max(0, rows.loc[(v, t), f"{eng}_share_of_extra_fuel"]) / sum(max(0, rows.loc[(v, t), f"{e}_share_of_extra_fuel"]) for e in ["main", "aux", "boiler"])
                     if (v, t) in rows.index else 0 for v, t, _ in pick])
    ax.barh(y, vals * 100, left=left, color=col, height=.55, label=lab, edgecolor="#fcfcfb", linewidth=1.5)
    left += vals * 100
for yy, (v, t, _) in zip(y, pick):
    if (v, t) in rows.index:
        ax.text(103, yy, f"hours {(rows.loc[(v, t), 'hours_ratio'] - 1) * 100:+.0f}%", va="center", fontsize=8, color=TXT2)
ax.set_yticks(y); ax.set_yticklabels([p[2] for p in pick]); ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of extra fuel burned in the 10 days after (%)")
ax.set_title("c  What queuing burns (hurricanes)", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(.45, -.3), ncol=3)
# figure title removed (caption carries it)
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "ed7_per_call.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure R: robustness (Extended Data Fig. 1)
PLc = pd.read_csv(os.path.join(OUT, "robust_placebo.csv")); PM = pd.read_csv(os.path.join(OUT, "robust_permutation.csv"))
NR = pd.read_csv(os.path.join(OUT, "robust_normals.csv"))
OUTLAB = {"Total CO2, storm window": "Study-region CO2, storm window", "Total CO2, after window": "Study-region CO2, 10 days after",
          "Total vessel-hours, storm window": "Study-region ship-hours, storm window", "Berth CO2, storm window": "Berth CO2, storm window",
          "Anchorage vessel-hours, after window": "Anchorage ship-hours, 10 days after",
          "Offshore-hold vessel-hours, after window": "Offshore-holding-area ship-hours, 10 days after",
          "Open-water vessel-hours, storm window": "Open-water ship-hours, storm window", "Port calls, storm window": "Port calls, storm window"}
VARLAB = {"default": "default", "reference [-60,-11]": "reference period \u221260 to \u221211 days",
          "reference [-30,-8]": "reference period \u221230 to \u22128 days", "no day pooling": "no pooling of days",
          "pool ±7 days": "pooling \u00b17 days", "mean not median": "mean instead of median",
          "exclusion 150 km": "exclusion radius 150 km", "exclusion 400 km": "exclusion radius 400 km",
          "≥4 baseline years": "\u22654 baseline years", "same-year flat (package style)": "flat same-year normal"}
LBL = lambda o: OUTLAB.get(o, o)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=.9))
ax = axs[0]
d = PLc[PLc.intensity == "Hurricane ≥64 kt"].reset_index(drop=True)
pm = PM[PM.intensity == "Hurricane ≥64 kt"].set_index("outcome")
y = np.arange(len(d))[::-1]
ax.errorbar(d.placebo_ratio, y, xerr=[d.placebo_ratio - d.placebo_lo, d.placebo_hi - d.placebo_ratio], fmt="o", color=GRAY, ms=5, label="placebo dates")
ax.scatter(d.real_ratio, y, color=ICOL[INTS[2]], s=40, zorder=3, label="hurricanes")
fmt_p = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
ylabels = [f"{LBL(o)}  ({fmt_p(pm.loc[o, 'p_perm'])})" for o in d.outcome]
ax.axvline(1, color=GRAY, lw=.8); ax.set_xscale("log"); ax.set_xlim(.35, 2.0)
ax.set_xticks([.4, .6, 1, 1.5]); ax.set_xticklabels(["\u221260%", "\u221240%", "0%", "+50%"])
ax.set_yticks(y); ax.set_yticklabels(ylabels, fontsize=8); ax.minorticks_off(); ax.set_xlabel("change from normal (log scale)")
ax.set_title("a  Real vs placebo cyclone dates", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
ax = axs[1]
outs = ["Total CO2, storm window", "Anchorage vessel-hours, after window", "Total CO2, after window"]
cols = [BLUE, ORANGE, AQUA]
vars_ = list(dict.fromkeys(NR.variant))
yv = np.arange(len(vars_))[::-1]
for o, c, off in zip(outs, cols, [-.22, 0, .22]):
    dd = NR[(NR.outcome == o) & (NR.intensity == "Hurricane ≥64 kt")].set_index("variant").reindex(vars_)
    ax.errorbar(dd.ratio, yv + off, xerr=[dd.ratio - dd.lo, dd.hi - dd.ratio], fmt="o", color=c, ms=4, elinewidth=.8, label=LBL(o))
ax.axvline(1, color=GRAY, lw=.8); ax.set_yticks(yv); ax.set_yticklabels([VARLAB.get(v, v) for v in vars_], fontsize=8)
ax.set_xlabel("change from normal (hurricanes)"); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0%" if abs(v - 1) < 1e-9 else f"{(v - 1) * 100:+.0f}%")); ax.set_title("b  Choice of normal", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.4, -.13), ncol=1)
# figure title removed (caption carries it); the dose-response panel was dropped (Supplementary Fig. 2 shows it)
fig.savefig(os.path.join(FIG, "ed1_robustness.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written")
