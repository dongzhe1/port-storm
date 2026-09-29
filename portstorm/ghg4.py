"""GHG4 bottom-up fuel and CO2 for every AIS message of a registered ship.

IMO Fourth GHG Study (2020) method: main-engine power from the propeller law with a weather
factor, auxiliary and boiler power by ship type, size and operating phase (berth, anchor,
manoeuvre, cruise), build-year SFOC with the STEAM2 load correction, and CO2 = 3.114 x fuel.
Intervals longer than DT_MAX_HOURS between two messages of a ship carry no fuel.

The fleet registry is a CSV with the Clarksons columns IMO Number, Type, Dwt, Built Date and
SOx Scrubber Technology Type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


ME_INSTA_POWER = [
    ("Container",          0,       14000,  5077),
    ("Container",      14000,       28000, 12083),
    ("Container",      28000,       42000, 20630),
    ("Container",      42000,       70000, 34559),
    ("Container",      70000,      112000, 52566),
    ("Container",     112000,      168000, 57901),
    ("Container",     168000,      203000, 61231),
    ("Container",     203000,      280000, 60202),
    ("Container",     280000,     np.inf,  60210),
    ("Bulk",               0,       10000,  1796),
    ("Bulk",           10000,       35000,  5941),
    ("Bulk",           35000,       60000,  8177),
    ("Bulk",           60000,      100000,  9748),
    ("Bulk",          100000,      200000, 16741),
    ("Bulk",          200000,     np.inf,  20094),
    ("Chemical",           0,        5000,   987),
    ("Chemical",        5000,       10000,  3109),
    ("Chemical",       10000,       20000,  5101),
    ("Chemical",       20000,       40000,  8107),
    ("Chemical",       40000,     np.inf,   8929),
    ("Oil",                0,        5000,   966),
    ("Oil",             5000,       10000,  2761),
    ("Oil",            10000,       20000,  4417),
    ("Oil",            20000,       60000,  8975),
    ("Oil",            60000,       80000, 11837),
    ("Oil",            80000,      120000, 13319),
    ("Oil",           120000,      200000, 17446),
    ("Oil",           200000,     np.inf,  27159),
    ("Tanker",             0,        1000,   687),
    ("Tanker",          1000,     np.inf,   2034),
    ("General",            0,        5000,  1454),
    ("General",         5000,       10000,  3150),
    ("General",        10000,       20000,  5280),
    ("General",        20000,     np.inf,   9189),
    ("Liquified-Gas",      0,   50000*0.45,  2236),
    ("Liquified-Gas", 50000*0.45, 100000*0.45, 12832),
    ("Liquified-Gas",100000*0.45,200000*0.45, 30996),
    ("Liquified-Gas",200000*0.45, np.inf,    36735),
    ("Refrigerated-Cargo", 0,        2000,   793),
    ("Refrigerated-Cargo", 2000,     6000,  3223),
    ("Refrigerated-Cargo", 6000,    10000,  6206),
    ("Refrigerated-Cargo",10000,   np.inf, 11505),
    ("Auto",               0,     np.inf,  15939),
    ("Yacht",              0,     np.inf,   1116),
    ("Fishing",            0,     np.inf,    983),
    ("Passenger",          0,     np.inf,  15301),
    ("Other",              0,     np.inf,  15301),
    ("Research",           0,     np.inf,  15301),
]


DESIGNED_SPEED = [
    ("Container",          0,       14000, 16.0),
    ("Container",      14000,       28000, 19.0),
    ("Container",      28000,       42000, 21.1),
    ("Container",      42000,       70000, 23.1),
    ("Container",      70000,      112000, 24.6),
    ("Container",     112000,      168000, 23.9),
    ("Container",     168000,      203000, 23.8),
    ("Container",     203000,      280000, 20.2),
    ("Container",     280000,     np.inf,  20.3),
    ("Bulk",               0,       10000, 11.8),
    ("Bulk",           10000,       35000, 13.8),
    ("Bulk",           35000,       60000, 14.3),
    ("Bulk",           60000,      100000, 14.4),
    ("Bulk",          100000,      200000, 14.5),
    ("Bulk",          200000,     np.inf,  14.6),
    ("Chemical",           0,        5000, 12.2),
    ("Chemical",        5000,       10000, 12.9),
    ("Chemical",       10000,       20000, 13.8),
    ("Chemical",       20000,       40000, 14.7),
    ("Chemical",       40000,     np.inf,  14.6),
    ("Oil",                0,        5000, 11.4),
    ("Oil",             5000,       10000, 12.1),
    ("Oil",            10000,       20000, 12.9),
    ("Oil",            20000,       60000, 14.6),
    ("Oil",            60000,       80000, 14.8),
    ("Oil",            80000,      120000, 14.8),
    ("Oil",           120000,      200000, 15.1),
    ("Oil",           200000,     np.inf,  15.5),
    ("Tanker",             0,        1000,  9.6),
    ("Tanker",          1000,     np.inf,  13.6),
    ("General",            0,        5000, 11.1),
    ("General",         5000,       10000, 12.7),
    ("General",        10000,       20000, 14.0),
    ("General",        20000,     np.inf,  15.0),
    ("Liquified-Gas",      0,   50000*0.45, 14.2),
    ("Liquified-Gas", 50000*0.45,100000*0.45, 16.4),
    ("Liquified-Gas",100000*0.45,200000*0.45, 19.0),
    ("Liquified-Gas",200000*0.45, np.inf,   19.2),
    ("Refrigerated-Cargo", 0,        2000, 12.1),
    ("Refrigerated-Cargo", 2000,     6000, 14.7),
    ("Refrigerated-Cargo", 6000,    10000, 17.4),
    ("Refrigerated-Cargo",10000,   np.inf, 20.2),
    ("Auto",               0,     np.inf, 19.6),
    ("Yacht",              0,     np.inf, 16.7),
    ("Fishing",            0,     np.inf, 11.7),
    ("Passenger",          0,     np.inf, 26.0),
    ("Other",              0,     np.inf, 18.2),
    ("Research",           0,     np.inf, 26.0),
]


AE_POWER_SEA = [
    ("Container",          0,       14000,   410),
    ("Container",      14000,       28000,   900),
    ("Container",      28000,       42000,   920),
    ("Container",      42000,       70000,  1400),
    ("Container",      70000,      112000,  1450),
    ("Container",     112000,      168000,  1800),
    ("Container",     168000,      203000,  2050),
    ("Container",     203000,      280000,  2300),
    ("Container",     280000,     np.inf,   2300),
    ("Bulk",               0,       10000,   190),
    ("Bulk",           10000,       35000,   190),
    ("Bulk",           35000,       60000,   260),
    ("Bulk",           60000,      100000,   410),
    ("Bulk",          100000,      200000,   410),
    ("Bulk",          200000,     np.inf,    410),
    ("Chemical",           0,        5000,   200),
    ("Chemical",        5000,       10000,   580),
    ("Chemical",       10000,       20000,   580),
    ("Chemical",       20000,       40000,   660),
    ("Chemical",       40000,     np.inf,    660),
    ("Oil",                0,        5000,   250),
    ("Oil",             5000,       10000,   375),
    ("Oil",            10000,       20000,   490),
    ("Oil",            20000,       60000,   510),
    ("Oil",            60000,       80000,   560),
    ("Oil",            80000,      120000,   690),
    ("Oil",           120000,      200000,   860),
    ("Oil",           200000,     np.inf,    860),
    ("Other-Liquids",      0,        1000,   500),
    ("Other-Liquids",   1000,     np.inf,    500),
    ("Tanker",             0,        1000,   500),
    ("Tanker",          1000,     np.inf,    500),
    ("General",            0,        5000,    60),
    ("General",         5000,       10000,   180),
    ("General",        10000,       20000,   520),
    ("General",        20000,     np.inf,    520),
    ("Liquified-Gas",      0,   50000*0.45,   240),
    ("Liquified-Gas", 50000*0.45,100000*0.45, 1700),
    ("Liquified-Gas",100000*0.45,200000*0.45, 2650),
    ("Liquified-Gas",200000*0.45, np.inf,    6750),
    ("Refrigerated-Cargo", 0,        2000,   570),
    ("Refrigerated-Cargo", 2000,     6000,  1200),
    ("Refrigerated-Cargo", 6000,    10000,  1650),
    ("Refrigerated-Cargo",10000,   np.inf,  3100),
    ("Auto",               0,        9999,   500),
    ("Auto",            9999,       20000,   510),
    ("Auto",           20000,     np.inf,    510),
    ("Passenger",          0,        2000,   450),
    ("Passenger",       2000,       10000,   450),
    ("Passenger",      10000,       60000,  3500),
    ("Passenger",      60000,     np.inf,  11500),
    ("Yacht",              0,     np.inf,    130),
    ("Fishing",            0,     np.inf,    200),
    ("Other",              0,     np.inf,    410),
    ("Research",           0,     np.inf,    320),
]

AE_POWER_BERTH = [
    ("Container",          0,       14000,   370),
    ("Container",      14000,       28000,   820),
    ("Container",      28000,       42000,   610),
    ("Container",      42000,       70000,  1100),
    ("Container",      70000,      112000,  1100),
    ("Container",     112000,      168000,  1150),
    ("Container",     168000,      203000,  1300),
    ("Container",     203000,      280000,  1400),
    ("Container",     280000,     np.inf,   1400),
    ("Bulk",               0,       10000,   110),
    ("Bulk",           10000,       35000,   110),
    ("Bulk",           35000,       60000,   150),
    ("Bulk",           60000,      100000,   240),
    ("Bulk",          100000,      200000,   240),
    ("Bulk",          200000,     np.inf,    240),
    ("Chemical",           0,        5000,   110),
    ("Chemical",        5000,       10000,   330),
    ("Chemical",       10000,       20000,   330),
    ("Chemical",       20000,       40000,   790),
    ("Chemical",       40000,     np.inf,    790),
    ("Oil",                0,        5000,   250),
    ("Oil",             5000,       10000,   375),
    ("Oil",            10000,       20000,   690),
    ("Oil",            20000,       60000,   720),
    ("Oil",            60000,       80000,   620),
    ("Oil",            80000,      120000,   800),
    ("Oil",           120000,      200000,  2500),
    ("Oil",           200000,     np.inf,   2500),
    ("Other-Liquids",      0,        1000,   500),
    ("Other-Liquids",   1000,     np.inf,    500),
    ("Tanker",             0,        1000,   500),
    ("Tanker",          1000,     np.inf,    500),
    ("General",            0,        5000,    90),
    ("General",         5000,       10000,   240),
    ("General",        10000,       20000,   720),
    ("General",        20000,     np.inf,    720),
    ("Liquified-Gas",      0,   50000*0.45,   240),
    ("Liquified-Gas", 50000*0.45,100000*0.45, 1700),
    ("Liquified-Gas",100000*0.45,200000*0.45, 2500),
    ("Liquified-Gas",200000*0.45, np.inf,    6750),
    ("Refrigerated-Cargo", 0,        2000,   520),
    ("Refrigerated-Cargo", 2000,     6000,  1100),
    ("Refrigerated-Cargo", 6000,    10000,  1500),
    ("Refrigerated-Cargo",10000,   np.inf,  2850),
    ("Auto",               0,        9999,   800),
    ("Auto",            9999,       20000,   850),
    ("Auto",           20000,     np.inf,    850),
    ("Passenger",          0,        2000,   450),
    ("Passenger",       2000,       10000,   450),
    ("Passenger",      10000,       60000,  3500),
    ("Passenger",      60000,     np.inf,  11500),
    ("Yacht",              0,     np.inf,    130),
    ("Fishing",            0,     np.inf,    200),
    ("Other",              0,     np.inf,    150),
    ("Research",           0,     np.inf,    320),
]

AE_POWER_PORT = AE_POWER_BERTH

AE_POWER_ANCHOR = [
    ("Container",          0,       14000,   450),
    ("Container",      14000,       28000,   910),
    ("Container",      28000,       42000,   910),
    ("Container",      42000,       70000,  1350),
    ("Container",      70000,      112000,  1400),
    ("Container",     112000,      168000,  1600),
    ("Container",     168000,      203000,  1800),
    ("Container",     203000,      280000,  1950),
    ("Container",     280000,     np.inf,   1950),
    ("Bulk",               0,       10000,   180),
    ("Bulk",           10000,       35000,   180),
    ("Bulk",           35000,       60000,   250),
    ("Bulk",           60000,      100000,   400),
    ("Bulk",          100000,      200000,   400),
    ("Bulk",          200000,     np.inf,    400),
    ("Chemical",           0,        5000,   170),
    ("Chemical",        5000,       10000,   490),
    ("Chemical",       10000,       20000,   490),
    ("Chemical",       20000,       40000,   550),
    ("Chemical",       40000,     np.inf,    550),
    ("Oil",                0,        5000,   250),
    ("Oil",             5000,       10000,   375),
    ("Oil",            10000,       20000,   500),
    ("Oil",            20000,       60000,   520),
    ("Oil",            60000,       80000,   490),
    ("Oil",            80000,      120000,   640),
    ("Oil",           120000,      200000,   770),
    ("Oil",           200000,     np.inf,    770),
    ("Other-Liquids",      0,        1000,   500),
    ("Other-Liquids",   1000,     np.inf,    500),
    ("Tanker",             0,        1000,   500),
    ("Tanker",          1000,     np.inf,    500),
    ("General",            0,        5000,    50),
    ("General",         5000,       10000,   130),
    ("General",        10000,       20000,   370),
    ("General",        20000,     np.inf,    370),
    ("Liquified-Gas",      0,   50000*0.45,   240),
    ("Liquified-Gas", 50000*0.45,100000*0.45, 1700),
    ("Liquified-Gas",100000*0.45,200000*0.45, 2000),
    ("Liquified-Gas",200000*0.45, np.inf,    7200),
    ("Refrigerated-Cargo", 0,        2000,   570),
    ("Refrigerated-Cargo", 2000,     6000,  1200),
    ("Refrigerated-Cargo", 6000,    10000,  1650),
    ("Refrigerated-Cargo",10000,   np.inf,  3100),
    ("Auto",               0,        9999,   500),
    ("Auto",            9999,       20000,   550),
    ("Auto",           20000,     np.inf,    550),
    ("Passenger",          0,        2000,   450),
    ("Passenger",       2000,       10000,   450),
    ("Passenger",      10000,       60000,  3500),
    ("Passenger",      60000,     np.inf,  11500),
    ("Yacht",              0,     np.inf,    130),
    ("Fishing",            0,     np.inf,    200),
    ("Other",              0,     np.inf,    150),
    ("Research",           0,     np.inf,    320),
]

BOILER_POWER_HIGH = [
    ("Container",          0,       14000,  250),
    ("Container",      14000,       28000,  340),
    ("Container",      28000,       42000,  460),
    ("Container",      42000,       70000,  480),
    ("Container",      70000,      112000,  590),
    ("Container",     112000,      168000,  620),
    ("Container",     168000,      203000,  630),
    ("Container",     203000,      280000,  630),
    ("Container",     280000,     np.inf,   700),
    ("Bulk",               0,       10000,   70),
    ("Bulk",           10000,       35000,   70),
    ("Bulk",           35000,       60000,  130),
    ("Bulk",           60000,      100000,  260),
    ("Bulk",          100000,      200000,  260),
    ("Bulk",          200000,     np.inf,   260),
    ("Chemical",           0,        5000,  670),
    ("Chemical",        5000,       10000,  670),
    ("Chemical",       10000,       20000, 1000),
    ("Chemical",       20000,       40000, 1350),
    ("Chemical",       40000,     np.inf,  1350),
    ("Oil",                0,        5000,  500),
    ("Oil",             5000,       10000,  750),
    ("Oil",            10000,       20000, 1250),
    ("Oil",            20000,       60000, 2700),
    ("Oil",            60000,       80000, 3250),
    ("Oil",            80000,      120000, 4000),
    ("Oil",           120000,      200000, 6500),
    ("Oil",           200000,     np.inf,  7000),
    ("Other-Liquids",      0,        1000, 1000),
    ("Other-Liquids",   1000,     np.inf,  1000),
    ("Tanker",             0,        1000, 1000),
    ("Tanker",          1000,     np.inf,  1000),
    ("General",            0,        5000,    0),
    ("General",         5000,       10000,  110),
    ("General",        10000,       20000,  150),
    ("General",        20000,     np.inf,   150),
    ("Liquified-Gas",      0,   50000*0.45, 1000),
    ("Liquified-Gas", 50000*0.45,100000*0.45, 1000),
    ("Liquified-Gas",100000*0.45,200000*0.45, 1500),
    ("Liquified-Gas",200000*0.45, np.inf,   3000),
    ("Refrigerated-Cargo", 0,        2000,  270),
    ("Refrigerated-Cargo", 2000,     6000,  270),
    ("Refrigerated-Cargo", 6000,    10000,  270),
    ("Refrigerated-Cargo",10000,   np.inf,  270),
    ("Auto",               0,     np.inf,   310),
    ("Yacht",              0,     np.inf,     0),
    ("Fishing",            0,     np.inf,     0),
    ("Passenger",          0,     np.inf,  1100),
    ("Other",              0,     np.inf,   110),
    ("Research",           0,     np.inf,   110),
]

SFOC_ME_BY_BUILT_SSD = (205, 185, 175)
SFOC_ME_BY_BUILT_MSD = (215, 195, 185)
SFOC_ME_BY_BUILT_HSD = (225, 205, 195)
SFOC_AE_BY_BUILT     = (225, 205, 195)

ME_ENGINE_CLASS = {
    "Container":          "SSD",
    "Bulk":               "SSD",
    "Chemical":           "SSD",
    "Oil":                "SSD",
    "Other-Liquids":      "MSD",
    "Tanker":             "MSD",
    "General":            "MSD",
    "Liquified-Gas":      "SSD",
    "Refrigerated-Cargo": "MSD",
    "Auto":               "SSD",
    "Passenger":          "MSD",
    "Yacht":              "HSD",
    "Fishing":            "HSD",
    "Other":              "MSD",
    "Research":           "MSD",
}

_SFOC_BANK = {
    "SSD": SFOC_ME_BY_BUILT_SSD,
    "MSD": SFOC_ME_BY_BUILT_MSD,
    "HSD": SFOC_ME_BY_BUILT_HSD,
}

def sfoc_me_from_built(built, vessel_type: str = "Container") -> float:
    if built is None or (isinstance(built, float) and np.isnan(built)):
        return float("nan")
    pre83, mid, post = _SFOC_BANK.get(ME_ENGINE_CLASS.get(vessel_type, "SSD"),
                                      SFOC_ME_BY_BUILT_SSD)
    if built < 1984:
        return float(pre83)
    if built < 2001:
        return float(mid)
    return float(post)


def sfoc_ae_from_built(built) -> float:
    if built is None or (isinstance(built, float) and np.isnan(built)):
        return float("nan")
    pre83, mid, post = SFOC_AE_BY_BUILT
    if built < 1984:
        return float(pre83)
    if built < 2001:
        return float(mid)
    return float(post)

SFOC_BOILER  = 340

CO2_FACTOR      = 3.114


SOG_BERTH        = 1.0
SOG_ANCHOR       = 3.0
CRUISE_RATIO     = 0.3
DT_MAX_HOURS     = 3.0
MCR_CAP          = 0.85
WEATHER_FACTOR   = 1.15
BOILER_LOW_FRAC  = 0.0

SFOC_LOAD_A    = 0.455
SFOC_LOAD_B    = -0.71
SFOC_LOAD_C    = 1.28
SFOC_LOAD_MIN  = 0.07

DELTA_W_LARGE_CONTAINER_DWT = 203000
DELTA_W_LARGE_CONTAINER     = 0.75
DELTA_W_CRUISE              = 0.70

SCRUBBER_TYPE_MAP = {
    "Fully Cellular Container": "Container",
    "Bulk Carrier": "Bulk",
    "Ore Carrier": "Bulk",
    "Open Hatch Carrier": "Bulk",
    "Chip Carrier": "Bulk",
    "Aggregates Carrier": "Bulk",
    "Cement Carrier": "Bulk",
    "Chemical & Oil Carrier": "Chemical",
    "Chemical Parcel Tanker": "Chemical",
    "Chemical Bulk Tanker": "Chemical",
    "Chemical Unknown Carrier": "Chemical",
    "Methanol Carrier": "Chemical",
    "Tanker": "Oil",
    "Product Carrier": "Oil",
    "Shuttle Tanker": "Oil",
    "Oil Bunkering Tanker": "Oil",
    "Asphalt & Bitumen Carrier": "Other-Liquids",
    "FSO": "Other-Liquids",
    "LPG Carrier": "Liquified-Gas",
    "Ethylene/LPG": "Liquified-Gas",
    "Ethane/LPG": "Liquified-Gas",
    "General Cargo": "General",
    "Multi-Purpose": "General",
    "Multi-Purpose/Heavy Lift Cargo": "General",
    "Heavy Lift Cargo Vessel": "General",
    "Transport (Heavy Lift)": "General",
    "Semi-Submersible Heavy Lift": "General",
    "Multi-Purpose Support": "General",
    "Reefer": "Refrigerated-Cargo",
    "Reefer Fish Carrier": "Refrigerated-Cargo",
    "Pure Car Carrier": "Auto",
    "Ro-Ro": "Auto",
    "Ro-Ro/Container": "Auto",
    "Ro-Ro Freight/Passenger": "Auto",
    "Ro-Ro/Lo-Lo": "Auto",
    "Cruise Ship": "Passenger",
    "Pass./Car Ferry": "Passenger",
    "Passenger Vessel": "Passenger",
    "Passenger/Cargo Vessel": "Passenger",
    "Seismic Survey": "Research",
    "Trailing Suction Hopper Dredger": "Other",
    "Pipe Layer": "Other",
    "Cylindrical Floating Prod. Unit": "Other",
}

def _build_lookup_arrays(table):
    types = np.array([t[0] for t in table])
    lo    = np.array([t[1] for t in table], dtype=float)
    hi    = np.array([t[2] for t in table], dtype=float)
    val   = np.array([t[3] for t in table], dtype=float)
    return types, lo, hi, val


def _vec_lookup(vessel_type: pd.Series, dwt: pd.Series, table) -> np.ndarray:
    types, lo, hi, val = _build_lookup_arrays(table)
    out = np.full(len(vessel_type), np.nan)
    vt = vessel_type.to_numpy()
    dw = dwt.to_numpy(dtype=float)
    for ut in pd.unique(vt):
        if ut is None or (isinstance(ut, float) and np.isnan(ut)):
            continue
        mask_t = (vt == ut)
        sub_lo  = lo[types == ut]
        sub_hi  = hi[types == ut]
        sub_val = val[types == ut]
        if len(sub_lo) == 0:
            continue
        d = dw[mask_t]
        idx = np.full(d.shape, -1, dtype=int)
        for k in range(len(sub_lo)):
            sel = (d >= sub_lo[k]) & (d < sub_hi[k])
            idx[sel] = k
        v = np.where(idx >= 0, sub_val[np.clip(idx, 0, None)], np.nan)
        out[mask_t] = v
    return out


def _sfoc_me_array(built: np.ndarray, vtype: np.ndarray) -> np.ndarray:
    out = np.full(built.shape, np.nan)
    for i in range(len(built)):
        b = built[i]
        if not np.isnan(b):
            out[i] = sfoc_me_from_built(int(b), str(vtype[i]))
    return out


def _sfoc_ae_array(built: np.ndarray) -> np.ndarray:
    out = np.full(built.shape, np.nan)
    for i in range(len(built)):
        b = built[i]
        if not np.isnan(b):
            out[i] = sfoc_ae_from_built(int(b))
    return out


def load_fleet(path) -> pd.DataFrame:
    sf = pd.read_csv(path, encoding="latin-1")
    sf["imo"] = pd.to_numeric(sf["IMO Number"], errors="coerce").astype("Int64")
    sf["dwt"] = (sf["Dwt"].astype(str).str.replace(",", "", regex=False)
                          .replace({"nan": np.nan}).astype(float))
    built = pd.to_datetime(sf["Built Date"], format="%d-%b-%y", errors="coerce")
    miss = built.isna()
    if miss.any():
        built = built.fillna(pd.to_datetime(sf["Built Date"], format="%d-%b-%Y",
                                            errors="coerce"))
    yr = built.dt.year
    sf["built"] = yr.where(yr <= 2027, yr - 100)
    sf["vessel_type_str"] = sf["Type"].map(SCRUBBER_TYPE_MAP).fillna("Other")
    sf["scrubber_tech"]   = sf["SOx Scrubber Technology Type"].astype(str).str.strip()
    sf.loc[sf["scrubber_tech"].isin(["", "nan", "None"]), "scrubber_tech"] = "Unknown"
    sf = sf.dropna(subset=["imo", "dwt", "built"]).copy()
    sf["built"] = sf["built"].astype(int)
    sf = sf.drop_duplicates(subset=["imo"], keep="first")
    return sf[["imo", "dwt", "built", "vessel_type_str", "scrubber_tech"]]


def fuel_messages(ais: pl.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    ais = ais.with_columns(
        pl.col("imo").str.replace(r"^IMO", "", literal=False)
                     .str.strip_chars()
                     .cast(pl.Int64, strict=False)
                     .alias("imo_int")
    )

    ais_pl = ais.join(
        pl.from_pandas(static.rename(columns={"imo": "imo_int"})),
        on="imo_int", how="inner",
    )

    if ais_pl.height == 0:
        return pd.DataFrame()

    ais_pl = (ais_pl
        .filter(pl.col("sog").is_not_null() & (pl.col("sog") >= 0)
                & (pl.col("sog") < 40))
        .sort(["imo_int", "base_date_time"])
    )

    ais_pl = ais_pl.with_columns(
        (pl.col("base_date_time").diff().over("imo_int")
                                  .dt.total_seconds() / 3600.0)
        .fill_null(0.0).alias("dt_h_raw")
    )
    ais_pl = ais_pl.with_columns(
        pl.when(pl.col("dt_h_raw") > DT_MAX_HOURS)
          .then(0.0).otherwise(pl.col("dt_h_raw")).alias("dt_h")
    )

    df = ais_pl.select([
        "mmsi", "imo_int", "base_date_time", "longitude", "latitude",
        "sog", "vessel_type_str", "dwt", "built", "scrubber_tech", "dt_h",
    ]).to_pandas().rename(columns={"imo_int": "imo"})

    df["P_inst"]      = _vec_lookup(df["vessel_type_str"], df["dwt"], ME_INSTA_POWER)
    df["V_design"]    = _vec_lookup(df["vessel_type_str"], df["dwt"], DESIGNED_SPEED)
    vt_arr  = df["vessel_type_str"].to_numpy()
    dwt_arr = df["dwt"].to_numpy(dtype=float)
    delta_w = np.where(
        vt_arr == "Passenger", DELTA_W_CRUISE,
        np.where((vt_arr == "Container") & (dwt_arr >= DELTA_W_LARGE_CONTAINER_DWT),
                 DELTA_W_LARGE_CONTAINER, 1.0))
    df["delta_w"]     = delta_w
    df["V_design_eff"] = df["V_design"] * delta_w
    df["P_AE_sea"]    = _vec_lookup(df["vessel_type_str"], df["dwt"], AE_POWER_SEA)
    df["P_AE_berth"]  = _vec_lookup(df["vessel_type_str"], df["dwt"], AE_POWER_BERTH)
    df["P_AE_anchor"] = _vec_lookup(df["vessel_type_str"], df["dwt"], AE_POWER_ANCHOR)
    df["P_boiler_hi"] = _vec_lookup(df["vessel_type_str"], df["dwt"], BOILER_POWER_HIGH)

    built_arr = df["built"].to_numpy(dtype=float)
    vtype_arr = df["vessel_type_str"].to_numpy()
    df["sfoc_ME_base"] = _sfoc_me_array(built_arr, vtype_arr)
    df["sfoc_AE_base"] = _sfoc_ae_array(built_arr)

    sog = df["sog"].to_numpy()
    vd  = df["V_design_eff"].to_numpy()
    ratio = np.where(vd > 0, sog / vd, 0.0)
    phase = np.where(
        sog <= SOG_BERTH,  "berth",
        np.where(sog <= SOG_ANCHOR, "anchor",
        np.where(ratio >= CRUISE_RATIO, "cruise", "maneuver")))
    df["phase"] = phase

    P_inst = df["P_inst"].to_numpy()
    raw    = P_inst * (np.where(vd > 0, sog / vd, 0.0) ** 3) * WEATHER_FACTOR
    P_ME   = np.clip(raw, 0.0, MCR_CAP * P_inst)
    P_ME   = np.where(np.isin(phase, ["cruise", "maneuver"]), P_ME, 0.0)
    df["P_ME"] = P_ME

    P_AE_sea_arr    = df["P_AE_sea"].to_numpy()
    P_AE_berth_arr  = df["P_AE_berth"].to_numpy()
    P_AE_anchor_arr = df["P_AE_anchor"].to_numpy()
    P_AE = np.where(phase == "cruise",   P_AE_sea_arr,
           np.where(phase == "maneuver", 0.5 * (P_AE_sea_arr + P_AE_anchor_arr),
           np.where(phase == "anchor",   P_AE_anchor_arr,
                                          P_AE_berth_arr)))
    df["P_AE"] = P_AE

    P_b_hi = df["P_boiler_hi"].to_numpy()
    P_b = np.where(phase == "cruise", BOILER_LOW_FRAC * P_b_hi, P_b_hi)
    df["P_boiler"] = P_b

    sfoc_me_base = df["sfoc_ME_base"].to_numpy()
    sfoc_ae_base = df["sfoc_AE_base"].to_numpy()
    dt           = df["dt_h"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        L_me = np.where(P_inst > 0, P_ME / P_inst, 0.0)
    L_me_eff = np.clip(L_me, SFOC_LOAD_MIN, 1.0)
    f_me = SFOC_LOAD_A * L_me_eff**2 + SFOC_LOAD_B * L_me_eff + SFOC_LOAD_C
    sfoc_me_load = np.where(P_ME > 0, sfoc_me_base * f_me, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        L_ae = np.where(P_AE_sea_arr > 0, P_AE / P_AE_sea_arr, 0.0)
    L_ae_eff = np.clip(L_ae, SFOC_LOAD_MIN, 1.0)
    f_ae = SFOC_LOAD_A * L_ae_eff**2 + SFOC_LOAD_B * L_ae_eff + SFOC_LOAD_C
    sfoc_ae_load = sfoc_ae_base * f_ae

    df["sfoc_ME_load"] = sfoc_me_load
    df["sfoc_AE_load"] = sfoc_ae_load
    df["fc_main_ton"]   = P_ME * sfoc_me_load * dt / 1e6
    df["fc_AE_ton"]     = P_AE * sfoc_ae_load * dt / 1e6
    df["fc_boiler_ton"] = P_b  * SFOC_BOILER  * dt / 1e6
    df["fc_total_ton"]  = (df["fc_main_ton"] + df["fc_AE_ton"]
                           + df["fc_boiler_ton"])
    df["co2_ton"]       = df["fc_total_ton"] * CO2_FACTOR
    return df
