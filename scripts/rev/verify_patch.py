#!/usr/bin/env python3
"""verify_patch.py — structural conservation checks before/after container surgery.

Ladder gate: any surgery must preserve
  1. total file size
  2. segment table (vmaddr/vmsize/fileoff/filesize) byte-identical
  3. LC_THREAD flavor structure unchanged
  4. diffs confined to the declared patched range
  5. __text cumulative-end monotonicity
"""
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kern_patch import parse_header, segments, threads, text_directory, load


def structure_fingerprint(d):
    h = parse_header(d)
    return {
        "size": len(d),
        "ncmds": h["ncmds"],
        "sizeofcmds": h["sizeofcmds"],
        "segs": [(s[0], s[1], s[2], s[3], s[4]) for s in segments(d)],
        "threads": [(t[0], t[1]) for t in threads(d)],
    }


def verify(orig_path, patched_path, patched_range=None):
    a, b = load(orig_path), load(patched_path)
    fa, fb = structure_fingerprint(a), structure_fingerprint(b)
    ok = True
    if fa["size"] != fb["size"]:
        print("FAIL size", fa["size"], fb["size"]); ok = False
    if fa["segs"] != fb["segs"]:
        print("FAIL segment table changed"); ok = False
    if fa["threads"] != fb["threads"]:
        print("FAIL thread flavors changed"); ok = False
    lo, hi = (patched_range or (0, 0))
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    outside = [i for i in diffs if not (lo <= i < hi)]
    if outside:
        print(f"FAIL {len(outside)} diffs outside declared range [{lo:#x},{hi:#x})",
              [hex(i) for i in outside[:8]]); ok = False
    for sn, saddr, ssize, soff in text_directory(b):
        if sn != "__text" or not soff:
            continue
        ends = []
        off = soff
        while off + 8 <= soff + min(ssize, 4096):
            e, tag, magic = struct.unpack_from("<IHH", b, off)
            if magic != 0x0700:
                break
            ends.append(e)
            off += 8
        mono = all(ends[i] < ends[i + 1] for i in range(len(ends) - 1))
        if ends and not mono:
            print("FAIL __text directory not monotonic"); ok = False
    if ok:
        print(f"PASS structure conserved; {len(diffs)} bytes changed in [{lo:#x},{hi:#x})")
    return ok


if __name__ == "__main__":
    pr = None
    if len(sys.argv) > 4:
        pr = (int(sys.argv[3], 0), int(sys.argv[4], 0))
    sys.exit(0 if verify(sys.argv[1], sys.argv[2], pr) else 1)
