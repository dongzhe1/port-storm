"""Read NOAA daily AIS archives in any of the layouts the collection ships in.

Columns are normalised to the pre-2025 names (MMSI, BaseDateTime, LAT, LON, ...) and always
addressed by name. Accepted files: AIS_YYYY_MM_DD.zip / .csv, ais-YYYY-MM-DD.csv(.zst), flat
or under per-year directories.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

CANONICAL = {
    "mmsi": "MMSI", "base_date_time": "BaseDateTime", "latitude": "LAT",
    "longitude": "LON", "sog": "SOG", "cog": "COG", "heading": "Heading",
    "vessel_name": "VesselName", "imo": "IMO", "call_sign": "CallSign",
    "vessel_type": "VesselType", "status": "Status", "length": "Length",
    "width": "Width", "draft": "Draft", "cargo": "Cargo",
    "transceiver": "TransceiverClass", "transceiverclass": "TransceiverClass",
}

DATE_RE = re.compile(r"(?:AIS_|ais-)(\d{4})[_-](\d{2})[_-](\d{2})\.(?:zip|csv|csv\.zst)$")

_NAME_FORMS = ("AIS_{u}.csv", "AIS_{u}.zip", "ais-{d}.csv", "ais-{d}.csv.zst",
               "AIS_{u}.csv.zst")


def find_day(src: Path, date: str) -> Path | None:
    u, d = date.replace("-", "_"), date
    year = date[:4]
    for base in (src, src / year):
        if not base.is_dir():
            continue
        for form in _NAME_FORMS:
            cand = base / form.format(u=u, d=d)
            if cand.is_file():
                return cand
    return None


def list_days(src: Path) -> list[str]:
    dates: set[str] = set()
    for path in (list(src.iterdir()) if src.is_dir() else []):
        if path.is_dir() and re.fullmatch(r"\d{4}", path.name):
            for f in path.iterdir():
                if (m := DATE_RE.search(f.name)):
                    dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        elif (m := DATE_RE.search(path.name)):
            dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return sorted(dates)


def raw_bytes(path: Path) -> bytes:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"no CSV inside {path.name}")
            return z.read(names[0])
    if path.name.endswith(".csv.zst"):
        import zstandard as zstd
        with path.open("rb") as fh:
            return zstd.ZstdDecompressor().stream_reader(fh).read()
    return path.read_bytes()


def read_polars(path: Path, columns: list[str] | None = None):
    import polars as pl

    raw = raw_bytes(path)
    header = raw[:4096].split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    present = [h.strip() for h in header.split(",")]
    canon = {h: CANONICAL.get(h.lower().strip(), h) for h in present}

    overrides = {}
    for src_name, want in canon.items():
        if want == "MMSI":
            overrides[src_name] = pl.Int64
        elif want in ("IMO", "CallSign", "VesselName"):
            overrides[src_name] = pl.Utf8
        elif want in ("SOG", "COG", "Heading", "LAT", "LON", "Draft"):
            overrides[src_name] = pl.Float64
        elif want in ("VesselType", "Status", "Cargo"):
            overrides[src_name] = pl.Int64

    df = pl.read_csv(io.BytesIO(raw), schema_overrides=overrides,
                     try_parse_dates=False, ignore_errors=True, quote_char=None)
    df = df.rename({k: v for k, v in canon.items() if k in df.columns and k != v})

    if "BaseDateTime" in df.columns and df["BaseDateTime"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("BaseDateTime").str.replace("T", " ")
              .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns {missing} "
                             f"(header was: {', '.join(present[:6])}...)")
        df = df.select(columns)

    if df.height:
        for col in ("LAT", "LON", "BaseDateTime"):
            if col in df.columns:
                null_rate = df[col].null_count() / df.height
                if null_rate > 0.5:
                    raise ValueError(
                        f"{path.name}: {col} is {null_rate:.0%} null after parsing. The "
                        "feed's format for it has probably changed; ignore_errors would "
                        "otherwise let this through as missing data.")
        for col in ("SOG", "VesselType", "Status"):
            if col in df.columns:
                null_rate = df[col].null_count() / df.height
                if null_rate > 0.9:
                    print(f"{path.name}: {col} is {null_rate:.0%} null after parsing",
                          file=sys.stderr)
    return df
