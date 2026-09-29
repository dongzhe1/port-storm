"""Check that a fresh run reproduces the sample and headline numbers in the manuscript
(tolerance 0.01 on ratios; 0.5 percentage points on percent changes)."""
import os, sys
import pandas as pd
OUT = os.environ.get("HURR_OUT", "out")
A = pd.read_csv(os.path.join(OUT, "pooled_ratios.csv"))
P = pd.read_csv(os.path.join(OUT, "panel_pairs.csv"))
ok = len(P) == 344 and P.port.nunique() == 47 and P.SID.nunique() == 48
print(f"events {len(P)} (expect 344), ports {P.port.nunique()} (47), cyclones {P.SID.nunique()} (48)")
n = P.intensity.value_counts()
for k, v in {"Tropical storm <50 kt": 223, "Strong TS 50–63 kt": 54, "Hurricane ≥64 kt": 67}.items():
    good = n.get(k, 0) == v; ok &= good
    print(f"  {k:>22}: {n.get(k, 0)} (expect {v}) {'OK' if good else 'DIFFERENT'}")
EXPECT = {("Tropical storm <50 kt", "storm"): 0.82, ("Strong TS 50–63 kt", "storm"): 0.73, ("Hurricane ≥64 kt", "storm"): 0.47}
d = A[(A.group == "intensity") & (A.measure == "co2_ton") & (A.tier == "total")].set_index(["level", "phase"]).ratio
for k, v in EXPECT.items():
    got = d.get(k, float("nan")); good = abs(got - v) < 0.01; ok &= good
    print(f"study-region CO2, {k[0]:>22}, storm window: {got:.3f} (expect {v:.2f}, i.e. {100 * (v - 1):+.0f}%) {'OK' if good else 'DIFFERENT'}")
T = pd.read_csv(os.path.join(OUT, "rev_intervention_totals.csv"))
print("targeting totals:\n", T.round(3).to_string(index=False))
print("REPRODUCED" if ok else "DIFFERENCES FOUND - check package version and inputs")
sys.exit(0 if ok else 1)
