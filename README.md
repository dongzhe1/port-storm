# portstorm

Hurricane impacts on U.S. port traffic, fuel, CO2 and SO2, measured from AIS vessel positions.

`portstorm` counts daily port calls the way [CyPort](https://github.com/ChenchenMobility/Maritime-Data-CyPort)
does, estimates fuel, CO2 and SO2 for every AIS message with the IMO Fourth GHG Study (GHG4)
bottom-up method, and measures what each tropical cyclone did to them: CyPort's disruption and
recovery metrics, and the elasticity of emissions to traffic during storms. `analysis/` turns
those tables into the paper's figures and tables.

```
portstorm/          the pipeline, python -m portstorm <command>
analysis/           the analysis: storm panel, figures, tables
results/pipeline/   every table the pipeline produced for 2015-2023
results/figures/ tables/ source_data/ numbers/     what analysis/ produced from them
run_analysis.sh     results/pipeline -> results/
run_mock.sh         synthetic data -> pipeline -> analysis
run_pipeline.sh     the full pipeline on an input folder (used by run_mock.sh and the real run)
```

## Install

```bash
pip install -e .                            # the pipeline; add .[prophet] for the Prophet baseline
pip install -r analysis/requirements.txt    # the analysis: matplotlib, statsmodels, scipy, basemap, openpyxl
```

Python 3.10 or later. The scripts are bash; on Windows run them from WSL or Git Bash.

## Two one-command runs

**Analysis from the tables in this repository** (about 20 minutes):

```bash
bash run_analysis.sh
```

This unpacks `results/pipeline/` into `data/processed/` (1 GB, about 10 seconds), runs the 20
steps of `analysis/`, and writes the figures, tables and source data to `results/`. Intermediate
tables and one log per step go to `data/analysis/`. The last step, `check_reproduction.py`,
checks the sample and headline numbers against the paper and prints `REPRODUCED`.

**Everything from scratch on synthetic data** (about 20 minutes):

```bash
bash run_mock.sh            # or: bash run_mock.sh <folder>, default demo/
```

This generates eleven ports at real Gulf and Atlantic locations, thirteen fake 2021 storms and four
years (2018-2021) of fake AIS, then runs every pipeline stage (`run_pipeline.sh`) and the whole
analysis on them. The numbers mean nothing; the point is that every step runs. Because
`check_reproduction.py` compares against the real run, it prints `DIFFERENCES FOUND` here, and
that is expected.

## Data in this repository

`results/pipeline/` holds every table of the 2015-2023 run. Files over 2 MB are xz-compressed,
and a compressed file over 48 MB is split into numbered parts, so every file fits in git.
`run_analysis.sh` unpacks them; to do it by hand:

```bash
python -m portstorm archive unpack --src results/pipeline --out data/processed
```

| file | one row per | contents |
|---|---|---|
| `storm_exposure.csv` | port × cyclone | exposure period (500 km buffer), closest distance, wind, SSHS in the window and over the storm's life, landfall |
| `storm_covariates.csv` | port × cyclone | surge, wind and rainfall at the nearest NOAA tide gauge and ASOS station |
| `storm_track_6h.csv` | cyclone × port × 3 h | distance from the eye to each port within 500 km, wind, SSHS |
| `port_daily_calls.csv` | port × day | port calls, distinct vessels, median stay, vessels present |
| `port_daily_emissions.csv` | port × day | fuel by engine (main, auxiliary, boiler), CO2 |
| `port_daily_emissions_by_class.csv` | port × day × class × scrubber | the same by ship class and scrubber type |
| `port_daily_sulfur.csv` | port × day | SO2 to air, to wash water, retained, with scenario bounds |
| `port_daily_sulfur_by_class.csv` | port × day × class | the same by ship class |
| `port_daily_gaps.csv` | port × day | vessel-time the fuel model's 3-hour rule credited and discarded |
| `tiers/emissions_by_zone.csv` | day × port complex × tier × class × scrubber | hours, fuel and CO2 at berth, in the channel, at anchor, in the offshore hold, offshore |
| `tiers/tier_daily_sulfur.csv` | day × port complex × tier | SO2 by tier |
| `elasticity/` | – | cargo and tanker emissions on cargo and tanker traffic: `summary.txt` (elasticity, decomposition, dose-response, placebo), `elasticity_windows.csv` (one row per port × cyclone window), `window_*.csv`, and one folder per fuel and SO2 measure |
| `elasticity_all_vessels/` | – | the same with tugs and harbour craft on the emission side |
| `resilience_<measure>/` | port × cyclone | CyPort disruption and recovery metrics, the paper's rule (95% interval) |
| `resilience_<measure>_cyport/` | port × cyclone | the same in the setting CyPort's released data follows |

Traffic counts cargo ships and tankers only; emissions cover every registered ship, so compare
them through `--classes cargo,tanker` (as `elasticity/` does). Floats have 6 significant figures.

The analysis outputs in `results/`:

| folder | contents |
|---|---|
| `figures/` | the 15 figures: `fig1`-`fig4` main text, `ed1`-`ed7` extended data, `supp1`-`supp4` supplementary |
| `tables/` | Supplementary Tables 1-4 (CSV, and 2-4 as one workbook) |
| `source_data/` | the table behind every figure |
| `numbers/` | the tables behind numbers quoted in the text |

## Reproduce from raw data

### 1. Prepare the data

Everything goes into one input folder, here `data/`:

```
data/
├── ais/2015/AIS_2015_01_01.zip ... ais/2023/AIS_2023_12_31.zip    public, about 1 TB
├── usace_principal_ports.geojson                                  public
├── ibtracs.NA.csv                                                 public
├── commercial/sp_global_export.csv                                licensed, you provide
├── commercial/scrubber_fleet.xlsx                                 licensed, you provide
├── fleet_registry.csv                                             built from the two above
└── scrubber_install_dates.csv                                     built from the two above
```

**Public data.** NOAA MarineCadastre daily AIS, 2015-2023 (3,287 daily files of about 0.3 GB);
the USACE Principal Ports polygons; IBTrACS v04r01, North Atlantic. Downloads resume, and files
already present are skipped:

```bash
python -m portstorm download reference --out data/
python -m portstorm download ais --start 2015-01-01 --end 2023-12-31 --out data/ais
python -m portstorm download cyport --out data/cyport     # only for `portstorm cyport compare`
```

**Commercial data.** The fuel model needs each ship's type, deadweight, build year and scrubber.
We took them from two licensed sources that cannot be redistributed. Put them here, with at
least these columns (other columns are ignored):

`data/commercial/sp_global_export.csv`: S&P Global (Sea-web) vessel export, CSV, one row per ship.

| column | used for |
|---|---|
| `IMO_Number` | the join key to AIS |
| `Shiptype_Level_5`, `Main_Vessel_Type` | ship type, mapped to the GHG4 classes |
| `DWT` | deadweight (t), which sets the GHG4 size bin |
| `Ship_Status` | `In Service/Commission` marks delivered ships |
| `Ship_Status_Effective_Date` | build year of delivered ships; for the others it is imputed from their IMO-number block |

`data/commercial/scrubber_fleet.xlsx`: Clarksons World Fleet Register scrubber listing,
sheet `Listing (2)` (a CSV export also works; another sheet name goes to `--sheet`), one row per
scrubber-fitted ship.

| column | used for |
|---|---|
| `IMO Number` | the join key |
| `SOx Scrubber Technology Type` | `Open Loop`, `Closed Loop`, `Hybrid`, ... ; decides where the SO2 goes |
| `Built Date` | install date of scrubbers fitted at build |
| `SOx Scrubber Retrofit Indicator` | `Y` for retrofits |
| `SOx Scrubber Retrofit Date` | install date of retrofits (optional) |
| `SOx Scrubber Status` | only `Fitted` rows are kept (optional) |

Then build the registry and the install dates:

```bash
python -m portstorm fleet --sp data/commercial/sp_global_export.csv \
    --scrubber data/commercial/scrubber_fleet.xlsx \
    --out data/fleet_registry.csv --install-dates data/scrubber_install_dates.csv
```

Without these licences, any `fleet_registry.csv` with the columns `IMO Number, Type, Dwt,
Built Date, SOx Scrubber Technology Type` (Clarksons type names such as `Bulk Carrier`,
`Fully Cellular Container`, `Tanker`; dates as `01-Jul-2008`) and any
`scrubber_install_dates.csv` with `imo, install_date` will do; `python -m portstorm fake`
writes small examples of both. Ships missing from the registry get no fuel, so coverage decides
how comparable the emission levels are: ours covers 99.5% of the IMO-carrying hulls that called
at U.S. ports.

**NOAA station data.** The storm covariates (surge, wind, rainfall) are fetched from the NOAA
CO-OPS and Iowa Environmental Mesonet ASOS services during the run, so that stage needs a
network connection. Responses are cached in `data/work/cache/`.

### 2. Run the pipeline

```bash
WORKERS=8 bash run_pipeline.sh data         # writes data/processed/, per-day files in data/work/
```

| step | command | writes |
|---|---|---|
| 1 | `calls extract`, `calls visits` | `port_daily_calls.csv`: CyPort-style port calls and vessels present |
| 2 | `storms` | `storm_exposure.csv`, `storm_track_6h.csv`: port-cyclone interactions, 500 km buffer |
| 3 | `resilience`, `cyport validate` | `resilience_port_calls*/`, `resilience_vessels_present*/` |
| 4 | `fuel` | per-message fuel and CO2, one file per day in `data/work/fuel/` |
| 5 | `zones --ports` | `port_daily_emissions*.csv`, then `resilience_co2_ton*/`, `resilience_fc_total_ton*/` |
| 6 | `gaps`, `window` | `port_daily_gaps.csv`, `elasticity_all_vessels/window_discard_share.csv` |
| 7 | `sulfur` | `port_daily_sulfur*.csv` (GHG4 Eq. 15 with per-ship scrubber dates) |
| 8 | `elasticity`, `window` | `elasticity/`, `elasticity_all_vessels/` |
| 9 | `tiers`, `zones --tier-map`, `sulfur` | `tiers/emissions_by_zone.csv`, `tiers/tier_daily_sulfur.csv` |
| 10 | `covariates` | `storm_covariates.csv` |

Every stage is `python -m portstorm <command> --help`. The per-day stages (1, 4, 5, 6, 9) take
roughly 15-20 minutes per year of AIS at 8 workers, with about 4 GB of memory per worker. They
skip days already written, so an interrupted run resumes where it stopped, and `fuel` takes
`--start`/`--end` to split the work by year across machines.

### 3. Run the analysis

```bash
bash run_analysis.sh
```

Files in `data/processed/` that are newer than the archives are kept, so this analyses your run
rather than ours. To store your run in the repository:

```bash
python -m portstorm archive pack --src data/processed --out results/pipeline
```

## Main results

Storm panel (`analysis/`; 344 port-cyclone events, 47 Gulf and Atlantic port areas, 48 cyclones,
2015-2023). CO2 in the storm window against the seasonal normal from storm-free years:

| wind at closest approach | events | CO2 ÷ normal |
|---|---|---|
| tropical storm, < 50 kt | 223 | 0.82 |
| strong tropical storm, 50-63 kt | 54 | 0.73 |
| hurricane, ≥ 64 kt | 67 | 0.47 |

Elasticity (`results/pipeline/elasticity/`; 26 ports, 85 cyclones):

| | value | 95% interval |
|---|---|---|
| elasticity of CO2 to vessels present, cargo and tanker ships | 1.28 | 1.13 – 1.43 |
| same, placebo windows (8 time shifts) | 0.95 – 1.20 | |
| elasticity of total SO2 | 1.45 | 1.30 – 1.61 |
| CO2 in storm window ÷ normal, ports within 100 km of the eye | 0.77 | |
| CO2 below normal over all 676 storm windows | 101,246 t | |

When traffic falls 10% during a storm, cargo and tanker CO2 falls about 12.8%, mostly because
the ships that stay remain for fewer hours. Sulfur falls faster still because scrubber ships,
burning 3.5% sulfur fuel, leave in greater proportion. Tugs and other harbour craft barely
respond, so total port emissions including them track traffic less closely (b = 0.73).

On CyPort's own released inputs, `resilience --threshold cyport --interval-width 0.80
--backend prophet` matches their disruption rate to within 1.2 percentage points and their
total impact at a median ratio of 0.98. Their released metrics differ from their paper in three
ways, all reproduced by the `cyport` setting: days are flagged against an 80% interval (Prophet's
default) rather than 95%, the impact is measured against the prediction rather than the lower
bound, and a day with no change counts as recovery.

## License

MIT. The GHG4 fuel model in `portstorm/ghg4.py` follows the IMO Fourth GHG Study (2020).
