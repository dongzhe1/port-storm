"""Step 2 - restrict to the study sample and put every measure on the all-ship normal.
The study sample drops the Caribbean and Great Lakes ports in panel.EXCLUDE_PORTS (344 events, 47 ports, 48 cyclones).
Each zone's class-level normals are then rescaled to its all-ship normal (CO2 for fuel and CO2, ship-hours for
hours), so the ship-class split sets only the mix (Methods, 'The no-cyclone normal').
Rewrites panel_results.csv.gz and panel_daily.csv.gz (originals kept once as *_unreconciled) and writes
class_normal_factors.csv.
"""
import os, shutil
import pandas as pd
import panel

OUT = panel.OUT
for f in ["panel_results.csv.gz", "panel_daily.csv.gz"]:
    b = os.path.join(OUT, f.replace(".csv.gz", "_unreconciled.csv.gz"))
    if not os.path.exists(b):
        shutil.copy(os.path.join(OUT, f), b)
b = os.path.join(OUT, "panel_pairs_full.csv")
if not os.path.exists(b):
    shutil.copy(os.path.join(OUT, "panel_pairs.csv"), b)
# study sample: drop Caribbean and Great Lakes ports (panel.EXCLUDE_PORTS) in case 01_panel ran on the full list
keep = lambda d: d[~d.port.isin(panel.EXCLUDE_PORTS)]
P = keep(pd.read_csv(b)); P.to_csv(os.path.join(OUT, "panel_pairs.csv"), index=False)
R = keep(pd.read_csv(os.path.join(OUT, "panel_results_unreconciled.csv.gz")))
D = keep(pd.read_csv(os.path.join(OUT, "panel_daily_unreconciled.csv.gz")))
print(f"study sample: {len(P)} events, {P.port.nunique()} ports, {P.SID.nunique()} storms")
F = panel.class_factors(R)
F.to_csv(os.path.join(OUT, "class_normal_factors.csv"), index=False)
panel.apply_factors_results(R, F).to_csv(os.path.join(OUT, "panel_results.csv.gz"), index=False)
panel.apply_factors_daily(D, F).to_csv(os.path.join(OUT, "panel_daily.csv.gz"), index=False)
print(f"{len(F)} pair x zone cells (zones and port-area total) rescaled")
print(F.factor.describe().round(3).to_string()); print(F.factor_h.describe().round(3).to_string())
print(F.sort_values("factor").head(10).to_string(index=False))
