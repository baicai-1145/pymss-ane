#!/usr/bin/env python3
"""Correlate .hwx runtime containers with E5 cache entries.

hwx  = final ANE program container (magic 0xbeefface, Mach-O variant).
       LC_THREAD flavor=4  = program/queue state (address table: 0x30054000 text,
                             0x30000000 input, 0x30028000.. = per-procedure buffers)
       LC_THREAD flavor=3  = per-procedure metadata (id, op count, buffer sizes)
       __TEXT/__text        = ANE command stream ("td" records for the firmware)
       __KERN_0             = kernel/const data per procedure
E5   = aneforge-side compiled artifact (H16G.e5 style, size 0x998 etc.) whose
       TD blocks carry the same channel/operator tables.

This tool walks the aneforge cache, parses each E5 (scripts/rev/e5_parser.py),
and prints per-model: op type sequences + td geometry, so they can be matched
against hwx procedure metadata (op counts, buffer sizes).
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e5_parser import parse  # noqa: E402


def hwx_procs(path):
    d = Path(path).read_bytes()[:0x400000]
    ncmds = struct.unpack_from("<I", d, 16)[0]
    off = 0x20
    out = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", d, off)
        if cmd == 0x04:
            flavor, count = struct.unpack_from("<II", d, off + 8)
            words = struct.unpack_from(f"<{count}I", d, off + 16)
            out.append({"flavor": flavor, "words": words})
        off += cmdsize
    return out


def main():
    cache = Path.home() / "Models/.aneforge-cache"
    rows = []
    for e5 in sorted(cache.rglob("*.e5")):
        try:
            p = parse(e5.read_bytes(), str(e5))
        except Exception:
            continue
        tds = p.get("td_blocks") or []
        geos = []
        for t in tds:
            g = t.get("geo") or {}
            key = (g.get("const_size"), tuple(g.get("size_words") or []), g.get("count"))
            geos.append(key)
        top = {}
        for g in geos:
            top[g] = top.get(g, 0) + 1
        rows.append({
            "e5": str(e5.relative_to(cache)).split("/")[0],
            "n_td": len(tds),
            "td_geo_top": sorted(top.items(), key=lambda kv: -kv[1])[:4],
        })
    print(json.dumps({"models": len(rows), "sample": rows[:20]}, indent=1))


if __name__ == "__main__":
    main()


def corpus_stats():
    from collections import Counter
    cache = Path.home() / "Models/.aneforge-cache"
    geo_ct = Counter(); ntd_ct = Counter(); n = 0
    for e5 in sorted(cache.rglob("*.e5")):
        try:
            p = parse(e5.read_bytes(), str(e5))
        except Exception:
            continue
        n += 1
        tds = p.get("td_blocks") or []
        ntd_ct[len(tds)] += 1
        for t in tds:
            g = t.get("geo") or {}
            geo_ct[(g.get("const_size"), tuple(g.get("size_words") or []), g.get("count"))] += 1
    print(f"corpus: {n} e5 files")
    print("td-per-file:", dict(sorted(ntd_ct.items())))
    print("top td geometries:")
    for g, c in geo_ct.most_common(12):
        print(f"  {c:4d}x  const={g[0]:#x} words={g[1]} count={g[2]}")


def hwx_summary(paths):
    for path in paths:
        procs = hwx_procs(path)
        fl4 = [x for x in procs if x["flavor"] == 4]
        fl3 = [x for x in procs if x["flavor"] == 3]
        print(f"\n== {Path(path).name}: {len(fl4)} flavor4, {len(fl3)} procedures")
        for k, pr in enumerate(fl3[:20]):
            w = pr["words"]
            nz = {i: w[i] for i in range(min(24, len(w))) if w[i]}
            print(f"  proc{k:3d}: ops={w[7] if len(w)>7 else '?'} nz={nz}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stats":
    corpus_stats()
    hwx_summary(sys.argv[2:])
