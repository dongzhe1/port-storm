"""Extended Data Fig. 2 - engine use per ship-hour in open water during the storm window.
Fuel per ship-hour relative to normal, by engine (main, generators, boilers), for tankers and cargo ships by cyclone
class; 95% intervals from resampling cyclones. Reads efficiency_pairs.csv (step 4).
Writes ed_engine_storm.csv (also offshore holding areas and anchorages, which held too few ship-hours during storms for
stable estimates and are not drawn), engine_use_after_hurricanes.csv (open water, ten days after hurricanes: the
main-engine check quoted in Results) and figures/ed2_engine_use.png.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from matplotlib.ticker import FixedLocator, FuncFormatter

OUT = os.environ.get("HURR_OUT", "out")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
EALL = pd.read_csv(os.path.join(OUT, "efficiency_pairs.csv"))
E = EALL[EALL.window == "storm"]
INTS = ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]
ILAB = {INTS[0]: "tropical storm", INTS[1]: "strong tropical storm", INTS[2]: "hurricane"}
ENG = [("fc_main_ton", "main", "Main engine", "#2a78d6"), ("fc_AE_ton", "aux", "Generators (auxiliary engines)", "#56b4e9"),
       ("fc_boiler_ton", "boiler", "Boilers", "#d55e00")]
ZONES = [("offshore", "a  Open water"), ("offshore_hold", "b  Offshore holding areas"), ("anchorage", "c  Anchorages")]


def ratio_of_ratios(d, n1, d1, n2, d2, B=1000, seed=1):
    rng = np.random.default_rng(seed)
    G = d.groupby("SID")[[n1, d1, n2, d2]].sum().values
    est = (G[:, 0].sum() / G[:, 1].sum()) / (G[:, 2].sum() / G[:, 3].sum())
    S = G[rng.integers(0, len(G), size=(B, len(G)))].sum(axis=1)
    with np.errstate(all="ignore"):
        bs = (S[:, 0] / S[:, 1]) / (S[:, 2] / S[:, 3])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi


rows = []
for t, _ in ZONES:
    for v in ["tanker", "cargo"]:
        for g in INTS:
            dd = E[(E.intensity == g) & (E[f"exp_{v}_{t}_hours"] > 0) & (E[f"obs_{v}_{t}_hours"] > 0)]
            if len(dd) < 5:
                continue
            r = dict(tier=t, vclass=v, intensity=g, n=len(dd), hours_ratio=dd[f"obs_{v}_{t}_hours"].sum() / dd[f"exp_{v}_{t}_hours"].sum())
            for fu, k, _, _ in ENG:
                r[f"{k}_per_hour"], r[f"{k}_lo"], r[f"{k}_hi"] = ratio_of_ratios(dd, f"obs_{v}_{t}_{fu}", f"obs_{v}_{t}_hours", f"exp_{v}_{t}_{fu}", f"exp_{v}_{t}_hours")
            rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "ed_engine_storm.csv"), index=False)

# open water, ten days after hurricanes: did ships speed up on the approach? (Results, recovery section)
Q = EALL[(EALL.window == "queue") & (EALL.intensity == INTS[2])]
qrows = []
for v in ["tanker", "cargo"]:
    dd = Q[(Q[f"exp_{v}_offshore_hours"] > 0) & (Q[f"obs_{v}_offshore_hours"] > 0)]
    est, lo, hi = ratio_of_ratios(dd, f"obs_{v}_offshore_fc_main_ton", f"obs_{v}_offshore_hours", f"exp_{v}_offshore_fc_main_ton", f"exp_{v}_offshore_hours")
    qrows.append(dict(tier="offshore", vclass=v, window="10 days after", intensity=INTS[2], n=len(dd), main_per_hour=est, main_lo=lo, main_hi=hi))
pd.DataFrame(qrows).to_csv(os.path.join(OUT, "engine_use_after_hurricanes.csv"), index=False)

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(figsize=(7.6, 5.0)); axs = [ax]
order = [(v, g) for v in ["tanker", "cargo"] for g in INTS]
ylab = [f"{'Tankers' if v == 'tanker' else 'Cargo ships'}, {ILAB[g]}" for v, g in order]
LIM = (.5, 6.5)        # every open-water estimate fits inside this range
for ax, (t, title) in zip(axs, ZONES[:1]):          # open water only
    for i, (v, g) in enumerate(order):
        r = R[(R.tier == t) & (R.vclass == v) & (R.intensity == g)]
        if r.empty:
            continue
        r = r.iloc[0]
        for j, (_, k, lab, c) in enumerate(ENG):
            y = i + (j - 1) * .24; x = r[f"{k}_per_hour"]
            if not np.isfinite(x):
                continue
            ax.plot([max(r[f"{k}_lo"], LIM[0]), min(r[f"{k}_hi"], LIM[1])], [y, y], color=c, lw=1.1)
            ax.plot(np.clip(x, *LIM), y, "o", color=c, ms=4.5, label=lab if i == 0 else None)
        ax.text(11.5, i, f"hours {(r.hours_ratio - 1) * 100:+.0f}%".replace("-", "−"), ha="right", va="center", fontsize=7, color="#52514e")
    ax.axvline(1, color="#8a8984", lw=.8); ax.axhline(2.5, color="#dcdad4", lw=.8)
    ax.set_xscale("log"); ax.set_xlim(LIM[0], 12)
    ax.xaxis.set_major_locator(FixedLocator([.5, .75, 1, 1.5, 2, 4])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if abs(v - 1) < 1e-9 else f"{(v - 1) * 100:+.0f}%".replace("-", "−")))
    ax.set_xlabel("fuel per ship-hour, change from normal (log scale)")
    # single panel: no title (the caption carries it)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(ylab); axs[0].invert_yaxis(); axs[0].tick_params(axis="y", length=0)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(.55, -.01))
fig.savefig(os.path.join(FIG, "ed2_engine_use.png"), dpi=200, bbox_inches="tight")
print(R[["tier", "vclass", "intensity", "n", "hours_ratio", "main_per_hour", "main_lo", "main_hi", "aux_per_hour", "boiler_per_hour"]].round(2).to_string())
