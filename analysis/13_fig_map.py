"""Supplementary Fig. 1 - study ports, port types and cyclone tracks (a), and the Gulf ports where post-hurricane
waiting concentrates (b; rings = share of excess waiting CO2 after hurricanes, as in Fig. 4b).
Inputs: port_locations.csv, storm_fixes.csv (step 10), het_port_traits.csv (step 9), rev_intervention_events.csv
(step 12), panel_pairs.csv; the upstream storm_track_6h.csv for wind at each fix.
Needs basemap and basemap-data-hires. Also writes map_ports.csv and map_tracks.csv (source data for the R version).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
import panel

OUT = panel.OUT
FIG = panel.FIG
CORE = panel.CORE
plt.rcParams.update({"font.size": 8.5, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
L = pd.read_csv(os.path.join(OUT, "port_locations.csv")).set_index("port")
L["lat"], L["lon"] = L.lat0, L.lon0          # plot the public port locations; the fitted points are only used to place the storms
PT = pd.read_csv(os.path.join(OUT, "het_port_traits.csv"), index_col=0)
IC = pd.read_csv(os.path.join(OUT, "rev_intervention_events.csv"))
J = IC[IC.intensity == "Hurricane ≥64 kt"].groupby("port").x_wait_co2.sum()      # excess post-hurricane waiting CO2, as in Fig. 4b
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
S = pd.read_csv(os.path.join(OUT, "storm_fixes.csv"))
TW = pd.read_csv(CORE + "storm_track_6h.csv").groupby(["SID", "time"]).wind_kt.max()
S = S[S.SID.isin(P.SID) & (S.rms_km < 30)].copy()
S["wind"] = [TW.get((a, b), np.nan) for a, b in zip(S.SID, S.time)]
S["t"] = pd.to_datetime(S.time)
L = L[L.index.isin(P.port)].join(PT[["port_type", "size", "all", "days"]])
L["co2_day"] = L["all"] / L["days"]
L["n_events"] = P.groupby("port").size().reindex(L.index).fillna(0)
L["n_hur"] = P[P.intensity == "Hurricane ≥64 kt"].groupby("port").size().reindex(L.index).fillna(0)
pos = J.clip(lower=0); L["jit_share"] = (pos / pos.sum()).reindex(L.index).fillna(0)
L.reset_index()[["port", "lat", "lon", "port_type", "co2_day", "jit_share"]].to_csv(os.path.join(OUT, "map_ports.csv"), index=False)
S.rename(columns={"wind": "wind_kt"})[["SID", "name", "season", "time", "lat", "lon", "wind_kt"]].to_csv(os.path.join(OUT, "map_tracks.csv"), index=False)
TCOL = {"Energy": "#d55e00", "Cargo": "#2a78d6", "Mixed": "#e69f00", "Service": "#56b4e9"}
PLAB = {"Energy": "Energy", "Cargo": "Cargo", "Mixed": "Mixed", "Service": "Service-craft"}
WCOL = [(0, 50, "#9cc3ee"), (50, 64, "#4f8fd9"), (64, 999, "#b8431f")]


def wind_col(w):
    return next(c for a, b, c in WCOL if a <= w < b) if np.isfinite(w) else "#c3c2b9"


def draw_tracks(m, ax, sids=None, lw=.9, alpha=.8):
    for sid, g in S.groupby("SID"):
        if sids is not None and sid not in sids:
            continue
        g = g.sort_values("t")
        x, y = m(g.lon.values, g.lat.values)
        for i in range(len(g) - 1):
            if (g.t.iloc[i + 1] - g.t.iloc[i]) > pd.Timedelta(hours=6):
                continue
            ax.plot(x[i:i + 2], y[i:i + 2], color=wind_col(g.wind.iloc[i]), lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)


def base(ax, ll, ur, res):
    m = Basemap(llcrnrlon=ll[0], llcrnrlat=ll[1], urcrnrlon=ur[0], urcrnrlat=ur[1], projection="merc", resolution=res, ax=ax)
    m.drawmapboundary(fill_color="#eef3f8", linewidth=.6, color="#c3c2b9")
    m.fillcontinents(color="#f4f3ef", lake_color="#eef3f8", zorder=1)
    m.drawcoastlines(linewidth=.4, color="#8a8984", zorder=1.5)
    m.drawstates(linewidth=.25, color="#c3c2b9", zorder=1.5); m.drawcountries(linewidth=.4, color="#a9a8a1", zorder=1.5)
    return m


short = lambda p: (p.replace(" Port Authority", "").replace(" Port District", "").replace(", Port of", "").replace(" Harbor District", "")
                   .replace("Port of ", "").replace(" Regional Port", "").replace(" Nav District", "").replace(" Parish Port", "")
                   .replace("Greater ", "").replace(" District", "").replace(" Port Corp", "").replace(" International Port", ""))
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.15, 1], wspace=.06, hspace=.28)

# a  all ports and storms
ax = fig.add_subplot(gs[:, 0])
m = base(ax, (-99.5, 22.5), (-66, 47.5), "l")
draw_tracks(m, ax, lw=.8, alpha=.7)
x, y = m(L.lon.values, L.lat.values)
ax.scatter(x, y, s=8 + L.co2_day.values / 25, c=[TCOL[t] for t in L.port_type], edgecolor="#fcfcfb", linewidth=.6, zorder=4)
for p in list(dict.fromkeys(L.sort_values("co2_day", ascending=False).index[:9].tolist() + ["Houston Port Authority, TX", "Port of Savannah, GA"])):
    xx, yy = m(L.loc[p, "lon"], L.loc[p, "lat"])
    ax.annotate(short(p).split(",")[0], (xx, yy), xytext=(4, 3), textcoords="offset points", fontsize=6.8, color="#0b0b0b", zorder=5)
xg, yg = m(-99.3, 25.3); ax.add_patch(plt.Rectangle(m(-98.2, 25.4), *(np.subtract(m(-86, 31.3), m(-98.2, 25.4))), fill=False, ec="#52514e", lw=.8, ls="--", zorder=3))
ax.text(*m(-98.1, 31.5), "b", fontsize=9, fontweight="bold", color="#52514e")
for txt, lo, la in [("Gulf of Mexico", -92.5, 25.2), ("Atlantic Ocean", -70.5, 31)]:
    ax.text(*m(lo, la), txt, fontsize=8, color="#52514e", style="italic", ha="center")
ax.set_title(f"a  {len(L)} Gulf and Atlantic port areas and nearby tropical cyclones, 2015–2023", loc="left", fontweight="bold", fontsize=9)
h1 = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="#fcfcfb", ms=7, label=f"{PLAB[t]} ports ({int((L.port_type == t).sum())})") for t, c in TCOL.items()]
h2 = [Line2D([], [], color=c, lw=2, label=l) for (_, _, c), l in zip(WCOL, ["Tropical storm (<50 kt)", "Strong tropical storm (50–63 kt)", "Hurricane (≥64 kt)"])]
h3 = [Line2D([], [], marker="o", ls="", color="#a9a8a1", ms=np.sqrt(8 + v / 25), label=f"{v:,} t CO2/day") for v in [200, 1000, 4000]]
leg = ax.legend(handles=h1 + h2 + h3, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3,
                columnspacing=1.2, labelspacing=.6, handletextpad=.4)

# b  Gulf coast zoom: hurricanes, ports and arrival-management potential
ax = fig.add_subplot(gs[0, 1])
m = base(ax, (-98.2, 25.4), (-86, 31.3), "i")
hur = P[P.intensity == "Hurricane ≥64 kt"].SID.unique()
draw_tracks(m, ax, sids=None, lw=.6, alpha=.35)
draw_tracks(m, ax, sids=hur, lw=1.6, alpha=.9)
SOFF = {"Nicholas 2021": (22, 16), "Hanna 2020": (-58, 10), "Nate 2017": (4, -9)}      # keep storm labels clear of port labels and each other
for sid in hur:
    g = S[S.SID == sid]
    g = g[(g.lon > -98) & (g.lon < -86.3) & (g.lat > 25.6) & (g.lat < 31.1)]
    if len(g) < 3:
        continue
    r = g.loc[g.lat.idxmin()]
    if r.lat > 27.2:
        continue
    xx, yy = m(r.lon, r.lat)
    nm = f"{r['name'].title()} {int(r.season)}"
    ax.annotate(nm, (xx, yy), xytext=SOFF.get(nm, (3, 4)), textcoords="offset points", fontsize=6.3, color="#b8431f",
                zorder=5, rotation=0)
G = L[(L.lon > -98.2) & (L.lon < -86) & (L.lat > 25.4) & (L.lat < 31.3)]
x, y = m(G.lon.values, G.lat.values)
ax.scatter(x, y, s=18 + 1600 * G.jit_share.values, facecolor="none", edgecolor="#0b0b0b", linewidth=.8, zorder=4)
ax.scatter(x, y, s=16, c=[TCOL[t] for t in G.port_type], edgecolor="#fcfcfb", linewidth=.5, zorder=5)
off = {"Brownsville, TX": (5, 5), "Houston Port Authority, TX": (-10, 7), "Texas City, TX": (6, -13), "Galveston, TX": (8, 3), "Port Arthur, TX": (-38, 6),
       "Sabine Pass Port Authority, TX": (4, -9), "Orange County Nav District, TX": (-12, 11), "Beaumont, TX": (-30, 6),
       "Lake Charles Harbor District, LA": (10, -10), "New Orleans, LA": (4, 5), "South Louisiana, LA, Port of": (-40, 7),
       "Port of Greater Baton Rouge, LA": (4, 4), "Plaquemines Port District, LA": (6, -4), "Greater Lafourche Port, LA": (-18, -11),
       "Terrebonne Parish Port, LA": (-40, -8), "Port of Iberia District, LA": (-24, -10), "Port of Pascagoula, MS": (4, 4),
       "Mobile, AL": (4, 4), "Port Freeport, TX": (5, -6), "Calhoun Port Authority, TX": (5, 3), "Corpus Christi, TX": (5, 2),
       "Brownsville, TX": (5, 2), "Panama City Port Authority, FL": (4, 4)}
for p in G.index:
    xx, yy = m(G.loc[p, "lon"], G.loc[p, "lat"])
    dx, dy = off.get(p, (4, 3))
    ax.annotate(short(p).split(",")[0], (xx, yy), xytext=(dx, dy), textcoords="offset points", fontsize=6.6, zorder=6)
ax.set_title("b  Gulf coast: hurricane tracks and where post-hurricane waiting concentrates", loc="left", fontweight="bold")
h = [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#0b0b0b", ms=np.sqrt(18 + 1600 * v), label=f"{v:.0%}") for v in [.05, .15, .3]]
ax.legend(handles=h, frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          title="Ring = share of excess\nwaiting CO2 after\nhurricanes (as in Fig. 4b)", title_fontsize=6.8, labelspacing=1.1, borderpad=.8)
# c  events by port type and storm class
ax = fig.add_subplot(gs[1, 1])
ax.set_position([ax.get_position().x0 + .03, ax.get_position().y0 + .02, ax.get_position().width - .05, ax.get_position().height - .06])
P2 = P.merge(PT[["port_type"]], left_on="port", right_index=True)
cnt = P2.groupby(["port_type", "intensity"]).size().unstack(fill_value=0).reindex(["Cargo", "Mixed", "Energy", "Service"])[
    ["Tropical storm <50 kt", "Strong TS 50–63 kt", "Hurricane ≥64 kt"]]
left = np.zeros(len(cnt))
for (col, (_, _, c)) in zip(cnt.columns, WCOL):
    ax.barh(range(len(cnt)), cnt[col], left=left, color=c, height=.62, edgecolor="#fcfcfb", label=col)
    for i, v in enumerate(cnt[col]):
        if v >= 8:
            ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=7, color="#0b0b0b" if col.startswith("Trop") else "#fcfcfb")
    left += cnt[col].values
for i, t in enumerate(cnt.index):
    ax.text(left[i] + 2, i, f"{int(left[i])} events, {int((PT.port_type == t).sum())} ports", va="center", fontsize=7, color="#52514e")
ax.set_yticks(range(len(cnt))); ax.set_yticklabels([f"{PLAB[t]} ports" for t in cnt.index]); ax.invert_yaxis()
ax.set_xlim(0, left.max() * 1.35); ax.set_xlabel("port × cyclone events, 2015–2023")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title("c  Events by port type and cyclone strength", loc="left", fontweight="bold")
# figure notes are in the manuscript captions
fig.savefig(os.path.join(FIG, "supp1_map.png"), dpi=220, bbox_inches="tight")
print("ok")
