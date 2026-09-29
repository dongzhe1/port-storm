"""Step 20 - copy the manuscript outputs into ../results:
  results/figures/      the 15 manuscript figures (PNG)
  results/tables/       Supplementary Tables 1-4
  results/source_data/  the result tables every figure is drawn from; the R scripts in ../R read these
  results/numbers/      tables behind numbers quoted in the text but not drawn in a figure
Run after steps 1-19. RESULTS can be set to another folder."""
import os, shutil
import pandas as pd

OUT = os.environ.get("HURR_OUT", "out")
RES = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
FIGS = ["fig1_response", "fig2_zones", "fig3_ports", "fig4_arrival_management", "ed1_robustness", "ed2_engine_use",
        "ed3_divergence_placebo", "ed4_net_change", "ed5_zones_heatmap", "ed6_drivers", "ed7_per_call",
        "supp1_map", "supp2_wind", "supp3_pollutants", "supp4_zone_schematic"]
SOURCE = ["divergence_pooled", "dose_bins_all_measures", "ed_engine_storm", "efficiency_engine_queue", "efficiency_pooled",
          "fig2_zone_ratios", "fig3_port_coast", "het_class_counts", "het_determinants", "het_port_traits", "het_winners_losers",
          "jit_pairs", "jit_scenarios", "map_ports", "map_tracks", "panel_pairs", "pollutant_intensity_test",
          "pollutant_tier_pairs", "rev_D_distributions", "rev_anchorage_robustness", "rev_intervention_events",
          "robust_dose_bins", "robust_normals", "robust_permutation", "robust_placebo"]
NUMBERS = ["pooled_ratios", "recovery_pooled", "efficiency_pooled", "robust_dose_response", "robust_loo", "decoupling_counts",
           "het_grouped", "het_grouped_strong_storms", "het_jit_concentration", "rev_intervention_concentration",
           "rev_intervention_totals", "claims_loo", "claims_landfall", "engine_use_after_hurricanes", "pollutant_pooled"]
for d in ["figures", "tables", "source_data", "numbers"]:
    os.makedirs(os.path.join(RES, d), exist_ok=True)
for f in FIGS:
    shutil.copy2(os.path.join(OUT, "figures", f + ".png"), os.path.join(RES, "figures", f + ".png"))
for f in os.listdir(os.path.join(OUT, "tables")):
    shutil.copy2(os.path.join(OUT, "tables", f), os.path.join(RES, "tables", f))
for f in SOURCE:
    shutil.copy2(os.path.join(OUT, f + ".csv"), os.path.join(RES, "source_data", f + ".csv"))
# event-level tables for the study region only (the full files hold every zone and are large)
E = pd.read_csv(os.path.join(OUT, "decoupling_events.csv"))
E[E.zone == "total"][["SID", "port", "window", "intensity", "lh", "lc", "g", "cls_co2"]].to_csv(
    os.path.join(RES, "source_data", "decoupling_events_total.csv"), index=False)
PL = pd.read_csv(os.path.join(OUT, "decoupling_placebo_classified.csv"))
PL[PL.zone == "total"][["SID", "port", "placebo_year", "window", "g"]].to_csv(
    os.path.join(RES, "source_data", "decoupling_placebo_total.csv"), index=False)
for f in NUMBERS:
    shutil.copy2(os.path.join(OUT, f + ".csv"), os.path.join(RES, "numbers", f + ".csv"))
print(f"results written to {os.path.abspath(RES)}")
