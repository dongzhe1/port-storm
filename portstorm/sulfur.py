"""SO2 from the fuel panel, split into air, discharged wash water and retained.

GHG4 Eq. (15): EF_SO2 = 2 x 0.97753 x S. Ships without a scrubber in use burn 0.1% S fuel
(North American ECA) and emit all SO2 to air; ships with one burn 3.5% S fuel and capture
98% (dry 99%), discharged with wash water for open loop, hybrid, membrane and unknown types,
kept on board for closed loop and dry. Whether a scrubber is in use comes from the panel's
scrubber_fitted flag (per-vessel install dates) or, without it, from --scrubber-from.

    python -m portstorm sulfur --panel port_daily_emissions_by_class.csv \
        --group-cols date,port --out port_daily_sulfur.csv --scenarios
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SULFUR_TO_SO2 = 2.0
SULFUR_OXIDISED = 0.97753

S_ECA_COMPLIANT = 0.001
S_RESIDUAL_FUEL = 0.035

SCRUBBER_SPEC = {
    "Open Loop":   {"removal": 0.98, "discharges": True},
    "Closed Loop": {"removal": 0.98, "discharges": False},
    "Hybrid":      {"removal": 0.98, "discharges": True},
    "Membrane":    {"removal": 0.98, "discharges": True},
    "Dry":         {"removal": 0.99, "discharges": False},
    "TBC":         {"removal": 0.98, "discharges": True},
    "Unknown":     {"removal": 0.98, "discharges": True},
    "No Scrubber": {"removal": 0.0, "discharges": False},
}

NEVER_FITTED = frozenset({"No Scrubber"})
DEFAULT_SCRUBBER_FROM = 2020

FUEL_COL = "fc_total_ton"

REG_IMO = "IMO Number"
REG_BUILT = "Built Date"
REG_RETRO_FLAG = "SOx Scrubber Retrofit Indicator"
REG_RETRO_DATE = "SOx Scrubber Retrofit Date"
REG_STATUS = "SOx Scrubber Status"


def load_scrubber_install(path: Path, sheet: str = "Listing (2)") -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        reg = pd.read_excel(path, sheet_name=sheet)
    else:
        reg = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in (REG_IMO, REG_BUILT, REG_RETRO_FLAG) if c not in reg.columns]
    if missing:
        raise SystemExit(f"registry has no {missing}; columns are {list(reg.columns)[:20]}")

    built = pd.to_datetime(reg[REG_BUILT], errors="coerce")
    retro = (pd.to_datetime(reg[REG_RETRO_DATE], errors="coerce")
             if REG_RETRO_DATE in reg.columns else pd.Series(pd.NaT, index=reg.index))
    retrofitted = reg[REG_RETRO_FLAG].astype(str).str.strip().str.upper().eq("Y")
    install = pd.to_datetime(retro.where(retrofitted, built))

    out = pd.DataFrame({"imo": pd.to_numeric(reg[REG_IMO], errors="coerce"),
                        "install_date": install})
    if REG_STATUS in reg.columns:
        out = out[reg[REG_STATUS].astype(str).str.strip().eq("Fitted").to_numpy()]
    out = out.dropna(subset=["imo", "install_date"])
    out["imo"] = out["imo"].astype("int64")
    return out.drop_duplicates(subset=["imo"], keep="first").reset_index(drop=True)


def ef_so2(sulfur_fraction: float) -> float:
    return SULFUR_TO_SO2 * SULFUR_OXIDISED * sulfur_fraction


def apply_sulfur(panel: pd.DataFrame, scrubber_from: int | None,
                 date_col: str = "date") -> pd.DataFrame:
    df = panel.copy()
    year = df[date_col].dt.year
    tech = df.get("scrubber_tech", pd.Series("Unknown", index=df.index)).fillna("Unknown")

    if scrubber_from == "never":
        fitted = pd.Series(False, index=df.index)
    elif scrubber_from == "always":
        fitted = pd.Series(True, index=df.index)
    elif "scrubber_fitted" in df.columns and scrubber_from != "year":
        fitted = df["scrubber_fitted"].astype(bool)
    elif scrubber_from is None:
        fitted = pd.Series(False, index=df.index)
    else:
        yr = scrubber_from if isinstance(scrubber_from, int) else DEFAULT_SCRUBBER_FROM
        fitted = year >= yr

    fitted = fitted & ~tech.isin(NEVER_FITTED)

    removal = tech.map(lambda t: SCRUBBER_SPEC.get(t, SCRUBBER_SPEC["Unknown"])["removal"])
    discharges = tech.map(lambda t: SCRUBBER_SPEC.get(t, SCRUBBER_SPEC["Unknown"])["discharges"])

    fuel = df[FUEL_COL].fillna(0.0)
    so2_total = fuel * fitted.map({True: ef_so2(S_RESIDUAL_FUEL),
                                   False: ef_so2(S_ECA_COMPLIANT)})

    captured = so2_total * removal.where(fitted, 0.0)
    df["so2_air_ton"] = so2_total - captured
    df["so2_washwater_ton"] = captured.where(discharges & fitted, 0.0)
    df["so2_retained_ton"] = captured.where(~discharges & fitted, 0.0)
    df["so2_total_ton"] = so2_total
    df["scrubber_fitted"] = fitted.astype(int)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, required=True,
                   help="a fuel panel carrying fc_total_ton and scrubber_tech")
    p.add_argument("--group-cols", default="date,port",
                   help="comma-separated keys to aggregate to, e.g. date,port "
                        "or date,complex,tier")
    p.add_argument("--date-col", default="date")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scrubber-from", type=int, default=DEFAULT_SCRUBBER_FROM,
                   help="year from which registry entries count as fitted "
                        f"(default {DEFAULT_SCRUBBER_FROM}; the registry has no "
                        "installation date, so this is an assumption, not a lookup)")
    p.add_argument("--scenarios", action="store_true",
                   help="also print the never-fitted and always-fitted bounds")
    args = p.parse_args()

    keys = [c.strip() for c in args.group_cols.split(",")]
    panel = pd.read_csv(args.panel, parse_dates=[args.date_col])
    for need in (FUEL_COL, "scrubber_tech"):
        if need not in panel.columns:
            raise SystemExit(f"panel has no {need!r} column; "
                             f"columns are {list(panel.columns)}")

    print(f"Panel      : {len(panel):,} rows, {panel[args.date_col].dt.year.min()}"
          f"-{panel[args.date_col].dt.year.max()}, "
          f"{panel[FUEL_COL].sum():,.0f} t fuel")
    print(f"EF check   : {ef_so2(S_ECA_COMPLIANT)*1000:.3f} kg SO2/t at 0.1% S, "
          f"{ef_so2(S_RESIDUAL_FUEL)*1000:.2f} kg/t at 3.5% S "
          f"({ef_so2(S_RESIDUAL_FUEL)/ef_so2(S_ECA_COMPLIANT):.0f}x)")

    use_flag = "scrubber_fitted" in panel.columns
    scored = apply_sulfur(panel, None if use_flag else args.scrubber_from, args.date_col)
    cols = ["so2_air_ton", "so2_washwater_ton", "so2_retained_ton", "so2_total_ton"]
    carried = {c: "sum" for c in (FUEL_COL, "co2_ton", "hours", "n_vessels",
                                  "n_vessels_in_cell", "n_messages")
               if c in scored.columns}
    out = (scored.groupby(keys, as_index=False)
           .agg({**carried, **{c: "sum" for c in cols}}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\nScenario   : " + ("per-vessel install dates from the panel"
                                  if use_flag else
                                  f"scrubbers assumed fitted from {args.scrubber_from}"))
    tot = scored[cols].sum()
    print(f"  SO2 to air         {tot['so2_air_ton']:12,.1f} t")
    print(f"  SO2 to wash water  {tot['so2_washwater_ton']:12,.1f} t  (discharged)")
    print(f"  SO2 retained       {tot['so2_retained_ton']:12,.1f} t  (closed loop / dry)")
    print(f"  sulfur mobilised   {tot['so2_total_ton']:12,.1f} t SO2-equivalent")

    if args.scenarios:
        has_flag = "scrubber_fitted" in panel.columns
        if has_flag:
            print("\nBounds, with the per-vessel install dates in between")
            runs = [("never fitted", "never"), ("per-vessel dates  <-- used", None),
                    ("always fitted", "always")]
        else:
            print("\nBounds, because no install dates were supplied")
            runs = [("never fitted", "never"), ("fitted from 2020", 2020),
                    ("fitted from 2015", 2015), ("always fitted", "always")]
        print(f"  {'assumption':>28s}  {'air':>12s}  {'wash water':>12s}  {'total':>12s}")
        for label, mode in runs:
            s = apply_sulfur(panel, mode, args.date_col)[cols].sum()
            print(f"  {label:>28s}  {s['so2_air_ton']:12,.1f}  "
                  f"{s['so2_washwater_ton']:12,.1f}  {s['so2_total_ton']:12,.1f}")
        print("\n  Air SO2 barely moves between them -- a scrubber ship emits slightly")
        print("  less SO2 to air than a compliant one, which is the point of fitting it.")
        print("  Wash water is the quantity the assumption decides, and with per-vessel")
        print("  install dates it is no longer an assumption.")

    print(f"\nWrote {args.out}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
