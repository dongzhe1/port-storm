"""Supplementary Fig. 2 - ship-hours, CO2, NOx and PM2.5 during the storm window by wind speed at closest approach,
cyclone classes shaded; study-region totals against the all-ship normal (port_window_pairs.csv from step 7).
Writes figures/supp2_wind.png and dose_bins_all_measures.csv.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from matplotlib.lines import Line2D
OUT = os.environ.get("HURR_OUT", "out")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
M = [("hours", "Ship-hours (operational)", "#8a8984"), ("co2", "CO2 (climate)", "#2a78d6"),
     ("nox", "NOx (air pollutant)", "#eb6834"), ("pm25", "PM2.5 (air pollutant)", "#e87ba4")]
T = pd.read_csv(os.path.join(OUT, "port_window_pairs.csv")); T = T[T.window == "storm"]
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))[["SID", "port", "wind_at_closest_kt"]]
T = T.merge(P, on=["SID", "port"])
LAB = ["<30", "30–39", "40–49", "50–63", "64–82 (Cat1)", "83–95 (Cat2)", "≥96 (Cat3+)"]
T["wind_bin"] = pd.cut(T.wind_at_closest_kt, [0, 30, 40, 50, 64, 83, 96, 999], right=False, labels=LAB)
rng = np.random.default_rng(3); rows = []
for b, d in T.groupby("wind_bin", observed=True):
    for m in ["hours", "co2", "nox", "pm25"]:
        g = d.groupby("SID")[["obs_" + m, "exp_" + m]].sum().values
        bs = g[rng.integers(0, len(g), (1000, len(g)))].sum(1)
        lo, hi = np.percentile(bs[:, 0] / bs[:, 1], [2.5, 97.5])
        rows.append(dict(wind_bin=b, measure=m, n=len(d), ratio=g[:, 0].sum() / g[:, 1].sum(), lo=lo, hi=hi))
B = pd.DataFrame(rows); B.to_csv(os.path.join(OUT, "dose_bins_all_measures.csv"), index=False)
M = [m for m in M if m[0] in ("hours", "co2")]
order = ["<30", "30–39", "40–49", "50–63", "64–82 (Cat1)", "83–95 (Cat2)", "≥96 (Cat3+)"]
xs = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 4.8))
NCL = pd.read_csv(os.path.join(OUT, "panel_pairs.csv")).intensity.value_counts()
for (a, b, lab, c) in [(-.5, 2.5, f"Tropical storm (<50 kt)\n{NCL['Tropical storm <50 kt']} events", "#f3f6fb"), (2.5, 3.5, f"Strong tropical\nstorm (50–63 kt)\n{NCL['Strong TS 50–63 kt']} events", "#e6eef8"),
                       (3.5, 6.5, f"Hurricane (≥64 kt)\n{NCL['Hurricane ≥64 kt']} events", "#f8ece6")]:
    ax.axvspan(a, b, color=c, zorder=0, lw=0)
    ax.text((a + b) / 2, 1.13, lab, ha="center", va="bottom", fontsize=7.5, color="#52514e")
for i, (m, lab, c) in enumerate(M):
    d = B[B.measure == m].set_index("wind_bin").reindex(order)
    ax.errorbar(xs + (i - .5) * .2, d.ratio, yerr=[d.ratio - d.lo, d.hi - d.ratio], fmt="s" if m == "hours" else "o", color=c, ms=5.5,
                capsize=1.8, elinewidth=1, markeredgecolor="#fcfcfb", markeredgewidth=.6, zorder=3)
n = B[B.measure == "co2"].set_index("wind_bin").reindex(order).n
for k, v in enumerate(n):
    ax.text(k, .245, f"n={int(v)}", ha="center", fontsize=7, color="#52514e")
ax.axhline(1, color="#8a8984", lw=.8); ax.set_ylim(.22, 1.12); ax.set_xlim(-.5, 6.5)
ax.set_xticks(xs); ax.set_xticklabels([o.replace(" (", "\n(") for o in order], fontsize=8)
ax.set_xlabel("wind speed at the cyclone's closest approach (kt)"); ax.set_ylabel("storm window, change from no-cyclone normal")
from matplotlib.ticker import FuncFormatter, FixedLocator
ax.yaxis.set_major_locator(FixedLocator([.3, .4, .6, .8, 1])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if abs(v - 1) < 1e-9 else f"{(v - 1) * 100:+.0f}%"))
ax.legend(handles=[Line2D([], [], marker="s" if m == "hours" else "o", ls="", color=c, label=l) for m, l, c in M], frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, .06))
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "supp2_wind.png"), dpi=200, bbox_inches="tight")
print("ok")
