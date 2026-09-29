"""Pack result tables into git-sized files, and unpack them again.

Files up to PLAIN_BYTES are copied unchanged. Larger ones are xz-compressed, and a compressed
file over PART_BYTES is split into numbered parts (name.csv.xz.000, .001, ...), so no file in
the repository exceeds GitHub's 50 MB warning. Unpacking skips targets newer than their source.

    python -m portstorm archive pack --src data/processed --out results/pipeline
    python -m portstorm archive unpack --src results/pipeline --out data/processed
"""
from __future__ import annotations

import argparse
import lzma
import re
import shutil
import sys
from pathlib import Path

PLAIN_BYTES = 2 << 20
PART_BYTES = 48 << 20
CHUNK = 16 << 20
PART_RE = re.compile(r"\.xz\.\d{3}$")


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".part"))


def pack(src: Path, out: Path) -> None:
    for f in files(src):
        dest = out / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for old in [dest, *dest.parent.glob(dest.name + ".xz*")]:
            old.unlink(missing_ok=True)
        if f.stat().st_size <= PLAIN_BYTES:
            shutil.copy2(f, dest)
            print(f"  {dest.relative_to(out)}")
            continue
        xz = dest.with_name(dest.name + ".xz")
        with f.open("rb") as fi, lzma.open(xz, "wb", preset=9 | lzma.PRESET_EXTREME) as fo:
            shutil.copyfileobj(fi, fo, CHUNK)
        size = xz.stat().st_size
        if size > PART_BYTES:
            with xz.open("rb") as fi:
                i = 0
                while block := fi.read(PART_BYTES):
                    xz.with_name(f"{xz.name}.{i:03d}").write_bytes(block)
                    i += 1
            xz.unlink()
        print(f"  {xz.relative_to(out)}  {f.stat().st_size / 1e6:.0f} -> {size / 1e6:.0f} MB"
              + (f" in {-(-size // PART_BYTES)} parts" if size > PART_BYTES else ""), flush=True)


def unpack(src: Path, out: Path) -> None:
    done = skipped = 0
    for f in files(src):
        rel = f.relative_to(src)
        if PART_RE.search(f.name):
            if not f.name.endswith(".000"):
                continue
            parts = sorted(f.parent.glob(f.name[:-3] + "[0-9][0-9][0-9]"))
            dest = out / rel.parent / f.name[:-7]
        elif f.name.endswith(".xz"):
            parts, dest = [f], out / rel.parent / f.name[:-3]
        else:
            parts, dest = [], out / rel
        newest = max(p.stat().st_mtime for p in parts or [f])
        if dest.exists() and dest.stat().st_mtime >= newest:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not parts:
            shutil.copy2(f, dest)
        else:
            tmp = dest.with_name(dest.name + ".part")
            dec = lzma.LZMADecompressor()
            with tmp.open("wb") as fo:
                for part in parts:
                    with part.open("rb") as fi:
                        while chunk := fi.read(CHUNK):
                            fo.write(dec.decompress(chunk))
            if not dec.eof:
                tmp.unlink()
                raise SystemExit(f"truncated archive: {parts[-1]}")
            tmp.replace(dest)
            print(f"  {dest.relative_to(out)}", flush=True)
        done += 1
    print(f"unpacked {done} files into {out} ({skipped} already up to date)")


def main() -> int:
    p = argparse.ArgumentParser(description="Pack or unpack result tables")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "unpack"):
        s = sub.add_parser(name)
        s.add_argument("--src", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.src.is_dir():
        p.error(f"--src not found: {args.src}")
    (pack if args.cmd == "pack" else unpack)(args.src, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
