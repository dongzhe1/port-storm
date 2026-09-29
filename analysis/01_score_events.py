"""Step 1 - score every port-cyclone pair against its no-cyclone normal (default method, see panel.py).
Writes panel_results.csv.gz (pair x period x ship class x zone x measure: observed, normal, ratio),
panel_daily.csv.gz (daily observed and normal series, 2 days before to 45 days after the storm window) and
panel_pairs.csv (one row per event with cyclone covariates).
"""
import os
from panel import load, run, OUT, MEAS, TIERS

D = load()
daily_cols = {f"{v}|{t}|{m}" for v in ["tanker", "cargo", "passenger", "other", "all"] for t in TIERS + ["total"] for m in MEAS}
daily_cols |= {"all|port|port_calls", "all|port|vessels_present"}
R, Dd = run(D, daily_cols=daily_cols)
R.to_csv(os.path.join(OUT, "panel_results.csv.gz"), index=False)
Dd.to_csv(os.path.join(OUT, "panel_daily.csv.gz"), index=False)
pairs = R[["SID", "storm", "season", "port", "min_dist_km", "sshs", "wind_at_closest_kt", "intensity", "landfall",
           "surge_m", "rain_mm", "k_days", "n_base"]].drop_duplicates()
pairs.to_csv(os.path.join(OUT, "panel_pairs.csv"), index=False)
print("pairs:", len(pairs), "ports:", pairs.port.nunique(), "storms:", pairs.SID.nunique())
h = R[(R.vclass == "all") & (R.tier == "total") & (R.measure == "co2_ton") & (R.phase == "storm")]
print((h.groupby("intensity", observed=True).obs.sum() / h.groupby("intensity", observed=True).expected.sum()).round(3))
