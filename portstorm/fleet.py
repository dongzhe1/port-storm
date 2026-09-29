"""Build the fuel model's fleet registry from an S&P Global vessel export.

Output uses the Clarksons columns ghg4.load_fleet reads (IMO Number, Type, Dwt, Built Date,
SOx Scrubber Technology Type). S&P carries no build year: it is taken from the status date of
in-service ships and otherwise imputed from the ship's IMO-number block (built_source says
which). With a Clarksons scrubber register, the scrubber technology is added and
--install-dates writes each scrubber's install date for `portstorm zones --scrubber-dates`.

    python -m portstorm fleet --sp sp_export.csv --scrubber scrubber_fleet.xlsx \
        --out fleet_registry.csv --install-dates scrubber_install_dates.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .sulfur import load_scrubber_install

TYPE_MAP = {
    "Bulk Carrier": "Bulk Carrier",
    "Bulk Carrier, Self-discharging": "Bulk Carrier",
    "Bulk Carrier, Self-discharging, Laker": "Bulk Carrier",
    "Bulk Carrier, Laker Only": "Bulk Carrier",
    "Ore Carrier": "Ore Carrier",
    "Wood Chips Carrier": "Chip Carrier",
    "Cement Carrier": "Cement Carrier",
    "Limestone Carrier": "Aggregates Carrier",
    "Bulk/Caustic Soda Carrier (CABU)": "Bulk Carrier",
    "Bulk/Oil/Chemical Carrier (CLEANBU)": "Bulk Carrier",
    "Bulk/Oil Carrier (OBO)": "Bulk Carrier",
    "Container Ship (Fully Cellular)": "Fully Cellular Container",
    "Container Ship (Fully Cellular/Ro-Ro Facility)": "Fully Cellular Container",
    "Container/Ro-Ro Cargo Ship": "Ro-Ro/Container",
    "Barge Carrier": "Fully Cellular Container",
    "Crude Oil Tanker": "Tanker",
    "Crude/Oil Products Tanker": "Tanker",
    "Products Tanker": "Product Carrier",
    "Shuttle Tanker": "Shuttle Tanker",
    "Bunkering Tanker (Oil)": "Oil Bunkering Tanker",
    "Replenishment Tanker": "Tanker",
    "Chemical/Products Tanker": "Chemical & Oil Carrier",
    "Chemical Tanker": "Chemical Bulk Tanker",
    "LPG/Chemical Tanker": "Chemical Parcel Tanker",
    "Molten Sulphur Tanker": "Chemical Unknown Carrier",
    "Fruit Juice Carrier, Refrigerated": "Chemical Parcel Tanker",
    "LNG Tanker": "LPG Carrier",
    "LPG Tanker": "LPG Carrier",
    "Combination Gas Tanker (LNG/LPG)": "Ethylene/LPG",
    "Gas Processing Vessel": "LPG Carrier",
    "Bunkering Tanker (LNG)": "LPG Carrier",
    "FSO, Gas": "LPG Carrier",
    "Asphalt/Bitumen Tanker": "Product Carrier",
    "FSO, Oil": "Product Carrier",
    "FPSO, Oil": "Product Carrier",
    "General Cargo Ship": "General Cargo",
    "General Cargo Ship (Open Hatch)": "Open Hatch Carrier",
    "General Cargo Ship (with Ro-Ro facility)": "Multi-Purpose",
    "Heavy Load Carrier": "Heavy Lift Cargo Vessel",
    "Heavy Load Carrier, semi submersible": "Semi-Submersible Heavy Lift",
    "Yacht Carrier, semi submersible": "Semi-Submersible Heavy Lift",
    "Deck Cargo Ship": "General Cargo",
    "Palletised Cargo Ship": "General Cargo",
    "Replenishment Dry Cargo Vessel": "General Cargo",
    "Livestock Carrier": "General Cargo",
    "Nuclear Fuel Carrier": "General Cargo",
    "Vehicles Carrier": "Pure Car Carrier",
    "Rail Vehicles Carrier": "Ro-Ro",
    "Ro-Ro Cargo Ship": "Ro-Ro",
    "Landing Craft": "Ro-Ro",
    "Logistics Vessel (Naval Ro-Ro Cargo)": "Ro-Ro",
    "Refrigerated Cargo Ship": "Reefer",
    "Fish Carrier": "Reefer Fish Carrier",
    "Passenger Ship": "Passenger Vessel",
    "Passenger/Ro-Ro Ship (Vehicles)": "Ro-Ro Freight/Passenger",
    "Fishing Vessel": "__Fishing",
    "Fish Factory Ship": "__Fishing",
    "Fishery Patrol Vessel": "__Fishing",
    "Research Survey Vessel": "Seismic Survey",
    "Yacht": "__Yacht",
    "Yacht (Sailing)": "__Yacht",
    "Sail Training Ship": "__Yacht",
}

MAIN_FALLBACK = {
    "Bulker": "Bulk Carrier", "Container": "Fully Cellular Container",
    "Tanker": "Tanker", "General Cargo": "General Cargo", "RoRo": "Ro-Ro",
    "Reefer": "Reefer", "Passenger": "Passenger Vessel", "Other": "__Other",
}

SENTINELS = {"__Fishing": "Fishing Vessel", "__Yacht": "Yacht", "__Other": "Other Vessel"}


def map_type(level5: pd.Series, main: pd.Series) -> pd.Series:
    out = level5.map(TYPE_MAP)
    out = out.fillna(main.map(MAIN_FALLBACK))
    out = out.fillna("Other Vessel")
    return out.replace(SENTINELS)


def main() -> int:
    p = argparse.ArgumentParser(description="Fleet registry for the fuel model from S&P")
    p.add_argument("--sp", type=Path, required=True, help="S&P Global full export CSV")
    p.add_argument("--scrubber", type=Path, default=None, help="Clarksons scrubber register")
    p.add_argument("--sheet", default="Listing (2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--install-dates", type=Path, default=None,
                   help="also write imo,install_date from the scrubber register")
    p.add_argument("--imo-list", type=Path, default=None,
                   help="one-column file of IMO numbers to restrict to")
    args = p.parse_args()

    sp = pd.read_csv(args.sp, low_memory=False)
    sp["imo"] = pd.to_numeric(sp["IMO_Number"], errors="coerce")
    sp = sp.dropna(subset=["imo"])
    sp["imo"] = sp["imo"].astype("int64")
    if args.imo_list:
        wanted = pd.to_numeric(pd.read_csv(args.imo_list).iloc[:, 0], errors="coerce")
        sp = sp[sp["imo"].isin(set(wanted.dropna().astype("int64")))]

    eff = pd.to_datetime(sp["Ship_Status_Effective_Date"], errors="coerce", format="mixed")
    dwt = pd.to_numeric(sp["DWT"], errors="coerce")
    delivered = sp["Ship_Status"].astype(str).str.strip().eq("In Service/Commission")
    built = eff.dt.year.where(delivered)
    fit = pd.DataFrame({"imo": sp["imo"], "yr": built})
    fit = fit[fit["yr"].between(1900, 2030)]
    edges = np.arange(1_000_000, 10_100_000, 100_000)
    lut = fit.groupby(pd.cut(fit["imo"], edges), observed=True)["yr"].median()
    imputed = pd.cut(sp["imo"], edges).map(lut)
    source = np.where(built.notna(), "status_date",
                      np.where(imputed.notna(), "imo_segment", "none"))
    year = built.fillna(imputed)

    out = pd.DataFrame({
        "IMO Number": sp["imo"].to_numpy(),
        "Type": map_type(sp["Shiptype_Level_5"], sp["Main_Vessel_Type"]).to_numpy(),
        "Dwt": dwt.to_numpy(),
        "Built Date": pd.to_datetime(
            year.astype("Float64").astype("float").round(),
            format="%Y", errors="coerce").dt.strftime("01-Jul-%Y").to_numpy(),
        "built_source": source,
        "SOx Scrubber Technology Type": "No Scrubber",
        "sp_shiptype": sp["Shiptype_Level_5"].to_numpy(),
        "sp_status": sp["Ship_Status"].to_numpy(),
    })

    if args.scrubber:
        if args.scrubber.suffix.lower() in (".xlsx", ".xls"):
            sc = pd.read_excel(args.scrubber, sheet_name=args.sheet)
        else:
            sc = pd.read_csv(args.scrubber, encoding="latin-1")
        sc["imo"] = pd.to_numeric(sc["IMO Number"], errors="coerce")
        sc = sc.dropna(subset=["imo"]).drop_duplicates(subset=["imo"], keep="first")
        tech = sc.set_index(sc["imo"].astype("int64"))["SOx Scrubber Technology Type"]
        out["SOx Scrubber Technology Type"] = out["IMO Number"].map(tech).fillna("No Scrubber")
        if args.install_dates:
            dates = load_scrubber_install(args.scrubber, args.sheet)
            args.install_dates.parent.mkdir(parents=True, exist_ok=True)
            dates.to_csv(args.install_dates, index=False)
            print(f"{len(dates):,} install dates -> {args.install_dates}")

    out = out[~(out["Dwt"].isna() | (out["Dwt"] <= 0) | out["Built Date"].isna())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out):,} ships -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
