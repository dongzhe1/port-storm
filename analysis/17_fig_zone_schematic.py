"""Supplementary Fig. 4 - schematic of the five operational zones, the port area (berths and channels) and the
study region (all five zones, out to about 100 km). Writes figures/supp4_zone_schematic.png.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _truminus  # noqa: F401  (typographic minus in every figure)
from matplotlib.patches import Rectangle, Polygon, Circle, FancyBboxPatch
OUT = os.environ.get("HURR_OUT", "out")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(-12, 110); ax.set_ylim(-32, 32); ax.axis("off")
# sea bands (distance from port boundary, km, along x)
ax.add_patch(Rectangle((0, -32), 110, 64, color="#e8f0f9", zorder=0))
ax.add_patch(Rectangle((30, -32), 70, 64, color="#dde8f5", zorder=0))
ax.add_patch(Rectangle((100, -32), 10, 64, color="#d2e0f1", zorder=0))
# land
ax.add_patch(Rectangle((-12, -32), 12, 64, color="#f0eee8", zorder=1))
# port polygon (USACE)
ax.add_patch(FancyBboxPatch((-10, -9), 13, 18, boxstyle="round,pad=0.3", fc="none", ec="#52514e", lw=1.4, ls="--", zorder=3))
ax.text(-3.5, 11.5, "port boundary\n(USACE polygon)", ha="center", fontsize=7.5, color="#52514e")
# berths
for y in (-6, -2, 2, 6):
    ax.add_patch(Rectangle((-8, y - .8), 4, 1.6, color="#2a78d6", zorder=4))
ax.text(-6, -12.5, "berths", ha="center", fontsize=8, color="#2a78d6", fontweight="bold")
# channel
ax.add_patch(Polygon([(-4, -1.2), (25, -2.2), (25, 2.2), (-4, 1.2)], color="#8a8984", alpha=.55, zorder=3))
ax.text(5, 3.6, "channel (moving ships, ≤25 km)", ha="left", fontsize=8, color="#52514e", fontweight="bold")
# anchorage
rng = np.random.default_rng(2)
for _ in range(14):
    ax.add_patch(Circle((rng.uniform(8, 26), rng.uniform(-24, -8)), .7, color="#eb6834", zorder=4))
ax.text(17, -27, "anchorage\n(anchored ships, <30 km)", ha="center", fontsize=8, color="#eb6834", fontweight="bold")
# offshore hold
for _ in range(10):
    ax.add_patch(Circle((rng.uniform(45, 85), rng.uniform(8, 24)), .7, color="#b8431f", zorder=4))
ax.text(65, 26.5, "offshore holding area (anchored ships, 30–100 km)", ha="center", fontsize=8, color="#b8431f", fontweight="bold")
# open water tracks
for y0 in (-18, -5, 12):
    ax.annotate("", xy=(30, y0 * .2), xytext=(105, y0), arrowprops=dict(arrowstyle="->", color="#4f8fd9", lw=1.2))
ax.text(70, -14, "open water (all other cells)", ha="center", fontsize=8, color="#184f95", fontweight="bold")
# distance axis
for x, l in [(0, "0 km"), (30, "30 km"), (100, "100 km")]:
    ax.plot([x, x], [-32, -30], color="#52514e", lw=1); ax.text(x, -31.5, l, ha="center", va="bottom", fontsize=7, color="#52514e")
# port area / study region brackets: two separate spans
ax.annotate("", xy=(-10, 33.2), xytext=(25, 33.2), annotation_clip=False, arrowprops=dict(arrowstyle="<->", color="#0b0b0b", lw=1))
ax.text(-10, 34.6, "port area: berths + channels (approximates what port inventories cover)", ha="left", fontsize=7.5, va="bottom")
ax.annotate("", xy=(-10, 37.8), xytext=(100, 37.8), annotation_clip=False, arrowprops=dict(arrowstyle="<->", color="#0b0b0b", lw=1))
ax.text(45, 39.2, "study region: all five zones, out to about 100 km", ha="center", fontsize=7.5, va="bottom")
ax.set_title(" ", pad=44)  # figure title removed (caption carries it)
fig.savefig(os.path.join(FIG, "supp4_zone_schematic.png"), dpi=200, bbox_inches="tight")
print("ok")
