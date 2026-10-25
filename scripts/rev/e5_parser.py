#!/usr/bin/env python3
"""H16G.e5 (Espresso E5 dispatch descriptor) full-structure parser.

Layer model (validated on M4 / macOS 26A434, espressoc 3600.x):
  * The .e5 is the DISPATCH DESCRIPTOR (FlatBuffer-ish, root vtable 0c00 1400 0400 0800)
    -- NOT the Mach-O hardware container (magic 0xbeefface) and not microcode.
  * Layout on disk (all LE):
      0x00  u32 header[8+]  -- section-blob absolute offsets (sorted desc), counts
      0x80  build-info attribute chain:
            [u32 8][u32 kcode][u32 vlen][value][u32 nlen][name\0] x N
      ...   section blobs (ops: __arg_frame / __op_attrs sub-buffers)
      tail  section directory: per record
            [u16 field-list ...][u32 nlen][name][pad4]
            then 8-byte trailer (two u32 checksum words).
  * __text__-class records (t0/t1, __const__, __text__, __sym_desc__) carry a
    u16 field list whose interpretation differs per section kind; the record
    byte sizes live in the blob-offset table at 0x18 and in the header counts.

Parser outputs a structured dump (JSON or text) + a canonical "signature" of the
descriptor for stable diffing (diff_matrix.py consumes it).
"""
from __future__ import annotations
import struct, sys, json, hashlib
from pathlib import Path

KNOWN_SECTIONS = ("__build_info__", "__extern_out_main__", "__extern_in_main__",
                  "__const__", "__text__", "__sym_desc__", "__data_main__")


def u16(d, o): return struct.unpack_from("<H", d, o)[0]
def u32(d, o): return struct.unpack_from("<I", d, o)[0]
def i32(d, o): return struct.unpack_from("<i", d, o)[0]


def parse_build_info(d: bytes, off: int) -> list[dict]:
    """Attribute chain starting at `off` (typically 0x80)."""
    attrs, cur, guard = [], off, 0
    while cur + 12 <= len(d) and guard < 64:
        guard += 1
        a, b, vl = struct.unpack_from("<III", d, cur)
        if a != 8 or vl == 0 or vl > 4096:
            break
        val = d[cur + 12: cur + 12 + vl].split(b"\0")[0].decode("utf-8", "replace")
        p = cur + 12 + vl
        nl = u32(d, p)
        if nl == 0 or nl > 128:
            attrs.append({"value": val, "name": "?"})
            break
        name = d[p + 4: p + 4 + nl].split(b"\0")[0].decode("utf-8", "replace")
        attrs.append({"name": name, "value": val})
        cur = p + 4 + nl + 1
        cur = (cur + 3) & ~3
    return attrs


def parse_section_directory(d: bytes):
    """Walk the name-terminated records backwards from the tail region.

    A record ends with [u32 nlen][name][pad4]; its body is a u16 field list.
    We anchor on every known section name and on op-style names discovered by
    scanning plausible (nlen == len) name markers.
    """
    n = len(d)
    anchors = []
    for nm in KNOWN_SECTIONS:
        i = d.find(nm.encode())
        while i >= 0:
            if i >= 4 and u32(d, i - 4) == len(nm):
                anchors.append((i, nm))
                break
            i = d.find(nm.encode(), i + 1)
    # op-style names: main__*, t0/t1, main
    import re
    for m in re.finditer(rb"[A-Za-z_][A-Za-z0-9_]{1,63}", d[n // 2:]):
        s = n // 2 + m.start()
        name = m.group()
        if u32(d, s - 4) == len(name):
            nxt = s + len(name)
            if nxt < n and d[nxt] == 0 and (len(name) <= 3 or b"__" in name or name.startswith(b"main")):
                anchors.append((s, name.decode()))
    anchors = sorted(set(anchors))
    records = []
    prev_end = None
    for s, nm in anchors:
        end4 = (s + len(nm) + 3) & ~3
        start = prev_end if prev_end is not None else None
        if start is None:
            # first anchor: scan back for the record start heuristically (<= 0x40 bytes)
            start = max(0, s - 0x40 - 4)
        rec = d[start: s + 4]
        u16s = [u16(rec, k) for k in range(0, len(rec) - 3, 2)]
        records.append({"name": nm, "name_at": s, "rec_start_approx": start, "u16_fields": u16s})
        prev_end = end4
    return records


def parse_td_blocks(d: bytes, parsed: dict) -> list[dict]:
    """Extract the __text__ payload (TD/op descriptor blocks) using the blob table.

    Validated layout (M4 26A434):
      * header 0x18 table gives absolute blob offsets; the blob whose range
        length equals `size + 0x20` where size comes from the __text__-class
        section record ('off' slot of the *preceding* record group) is __text__.
      * payload = [32B header][TD-block 1 (t0)][TD-block 2 (t1)]...
      * each TD block starts with the signature u32 pair (0x000e0008, 0x00080007)
        followed by geometry; block ends where the next signature or EOF occurs.
      * geometry slots (u16 pairs, 0x0000-padded):
          - const-size u32 @+0x30 (= __const__ size!)
          - four more size u32s @+0x34..0x40
          - count u32 @+0x5a region (4 vs 3 = conv vs matmul observed)
          - shape words near tail: C/N, M, K, W ... (conv: 0x100,0x3c00; mm: 0x40,0x3c,0x100)
    """
    n = len(d)
    offs = sorted({o for o in parsed["header"]["blob_offsets"] if 0x80 < o < n})
    sig = struct.pack("<II", 0x000E0008, 0x00080007)
    blocks = []
    for a, b in zip(offs, offs[1:] + [n]):
        blob = d[a:b]
        hits = [i for i in range(8, len(blob) - 8, 4) if blob[i:i+8] == sig]
        if not hits:
            continue
        # candidate __text__ blob: blocks start at `hits`
        for h in hits:
            blocks.append({"blob_off": a, "start": a + h,
                           "abs_off": hex(a + h)})
        parsed.setdefault("text_blobs", []).append({"off": a, "end": b, "len": b - a,
                                                    "sig_hits": [hex(h) for h in hits]})
    # split each blob into TD blocks by signature
    tds = []
    for tb in parsed.get("text_blobs", []):
        blob = d[tb["off"]:tb["end"]]
        starts = [i for i in range(8, len(blob) - 8, 4) if blob[i:i+8] == sig]
        for i, s in enumerate(starts):
            e = starts[i+1] if i + 1 < len(starts) else len(blob)
            blk = blob[s:e]
            tds.append(decode_td_block(blk, tb["off"] + s))
    return tds


def decode_td_block(blk: bytes, abs_off: int) -> dict:
    u16s = [u16(blk, k) for k in range(0, len(blk), 2)]
    u32s = [struct.unpack_from("<I", blk, k)[0] for k in range(0, len(blk), 4)]
    geo = {}
    # known slots (offsets within block, from conv/mm validation)
    def u32at(off):
        return struct.unpack_from("<I", blk, off)[0] if off + 4 <= len(blk) else None
    # u16-pair semantics: sizes live as u16 values at fixed u16 slots
    def h(idx):
        return u16s[idx] if idx < len(u16s) else None
    geo["const_size"] = h(25)                       # u16 slot 25 (0x78 conv / 0x1e mm)
    geo["size_words"] = [h(i) for i in (28, 30, 32, 34)] or None  # 0x48=72, 0x14=20, 0x44=68, 0x68=104
    geo["count"] = h(46)                             # conv=4, mm=3 (u16 slot 46)
    # tail geometry: scan last 0x30 bytes for plausible shape words
    tail = u32s[-12:]
    geo["tail_words"] = tail
    return {"abs_off": hex(abs_off), "len": len(blk), "u16": u16s, "geo": geo}


def parse(d: bytes, path: str) -> dict:
    n = len(d)
    hdr = {
        "size": n,
        "header_u32": [u32(d, o) for o in range(0, 0x18, 4)],
        "blob_offsets": [u32(d, o) for o in range(0x18, 0x38, 4)],
        "trailer": [u32(d, n - 8), u32(d, n - 4)],
    }
    # blob offsets: filter plausibles and compute inter-blob sizes
    offs = sorted({o for o in hdr["blob_offsets"] if 0 < o < n})
    hdr["blob_ranges"] = [(a, (b - a)) for a, b in zip(offs, offs[1:] + [n])]
    out = {
        "path": path,
        "sha256": hashlib.sha256(d).hexdigest(),
        "header": hdr,
        "build_info": parse_build_info(d, 0x80),
        "sections": parse_section_directory(d),
    }
    out["td_blocks"] = parse_td_blocks(d, out)
    # __text__ / kernel-ish blob extraction: the region between the last attr
    # (input-file-path end) and the section directory is the section-blob area.
    bi = out["build_info"]
    if bi:
        last = bi[-1]
        # find last name in raw bytes to get its end
        i = d.find(last["name"].encode())
        if i >= 0:
            blob_start = (i + len(last["name"]) + 1 + 3) & ~3
            out["blob_area"] = {"start": blob_start,
                                "end": out["sections"][0]["name_at"] if out["sections"] else n}
    return out


def signature(parsed: dict) -> dict:
    """Canonical, diff-stable view: strip path-dependent build-info values."""
    bi = {a["name"]: a["value"] for a in parsed["build_info"]}
    bi.pop("input-file-path", None)
    secs = []
    for r in parsed["sections"]:
        secs.append({"name": r["name"], "u16": r["u16_fields"]})
    tds = [{"len": t["len"], "geo": t["geo"], "u16": t["u16"]} for t in parsed.get("td_blocks", [])]
    return {"size": parsed["header"]["size"], "build_info": bi, "sections": secs,
            "blob_offsets": parsed["header"]["blob_offsets"], "td_blocks": tds}


def render_text(p: dict) -> str:
    L = []
    L.append(f"== {p['path']} ({p['header']['size']} bytes, sha256 {p['sha256'][:16]})")
    L.append(f"header u32[0:6]: {[hex(x) for x in p['header']['header_u32']]}")
    L.append(f"blob offsets:    {[hex(x) for x in p['header']['blob_offsets']]}")
    L.append(f"trailer:         {[hex(x) for x in p['header']['trailer']]}")
    L.append("-- build_info --")
    for a in p["build_info"]:
        L.append(f"  {a['name']} = {a['value']}")
    L.append("-- sections (name @ offset, u16 fields) --")
    for r in p["sections"]:
        fl = " ".join(f"{x:04x}" for x in r["u16_fields"])
        L.append(f"  {r['name']:34s} @0x{r['name_at']:04x}  {fl}")
    if "blob_area" in p:
        L.append(f"blob area: 0x{p['blob_area']['start']:04x}..0x{p['blob_area']['end']:04x}")
    tbs = p.get("text_blobs") or []
    for tb in tbs:
        L.append(f"__text__-class blob @0x{tb['off']:x}..0x{tb['end']:x} ({tb['len']}B), sig hits at {tb['sig_hits']}")
    for i, t in enumerate(p.get("td_blocks", [])):
        g = t["geo"]
        L.append(f"  TD[{i}] @0x{t['abs_off']} len={t['len']}B const_size={g.get('const_size')} sizes={g.get('size_words')} count={g.get('count')} tail={[hex(x) for x in g.get('tail_words', [])]}")
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("e5", nargs="+", help="H16G.e5 file(s)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--sig", action="store_true", help="emit canonical signature (diff-stable)")
    args = ap.parse_args()
    for path in args.e5:
        d = Path(path).read_bytes()
        p = parse(d, path)
        if args.json:
            print(json.dumps(signature(p) if args.sig else p, indent=1))
        else:
            print(render_text(p))
            print()


if __name__ == "__main__":
    main()
