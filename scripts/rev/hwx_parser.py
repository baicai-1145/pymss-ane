#!/usr/bin/env python3
"""Parse the ANE runtime container (.hwx, magic 0xbeefface Mach-O variant).

Header: standard 64-bit Mach-O layout with custom magic:
  [magic=0xbeefface][cputype=0x80][cpusubtype=7][filetype=2]
  [ncmds][sizeofcmds][flags=0x200000][reserved]
Load commands: LC_SEGMENT_64 (0x19) segments/sections, LC_FVMLIB (0x0d?) etc.
Segment vmaddrs use the ANE IOMMU apertures (0x30008000 input, 0x3000c000
output, 0x30010000 text, 0x30018000 weights per ane-guide ch23).
"""
import struct, sys
from pathlib import Path

LC_SEGMENT_64 = 0x19
LC_THREAD = 0x04
LC_UNIXTHREAD = 0x05


def parse(d: bytes) -> dict:
    magic, cputype, cpusub, ftype, ncmds, sizeofcmds, flags, resv = struct.unpack_from("<8I", d, 0)
    assert magic == 0xBEEFFACE, hex(magic)
    out = {
        "cputype": cputype, "cpusubtype": cpusub, "filetype": ftype,
        "ncmds": ncmds, "flags": flags, "segments": [], "load_commands": [],
    }
    off = 0x20
    for i in range(ncmds):
        if off + 8 > len(d):
            break
        cmd, cmdsize = struct.unpack_from("<II", d, off)
        if cmd == LC_SEGMENT_64:
            name = d[off + 8: off + 24].rstrip(b"\0").decode("latin1")
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<4Q", d, off + 0x18)
            maxprot, initprot, nsects, sflags = struct.unpack_from("<4I", d, off + 0x38)
            seg = {"name": name, "vmaddr": vmaddr, "vmsize": vmsize,
                   "fileoff": fileoff, "filesize": filesize, "sections": []}
            so = off + 72
            for s in range(nsects):
                sn = d[so: so + 16].rstrip(b"\0").decode("latin1")
                sseg = d[so + 16: so + 32].rstrip(b"\0").decode("latin1")
                saddr, ssize = struct.unpack_from("<2Q", d, so + 32)
                soff = struct.unpack_from("<I", d, so + 56)[0]
                seg["sections"].append({"name": sn, "seg": sseg, "addr": saddr,
                                        "size": ssize, "offset": soff})
                so += 80
            out["segments"].append(seg)
        else:
            out["load_commands"].append({"cmd": cmd, "size": cmdsize, "off": off})
        off += cmdsize
    return out


def render(p: dict, path: str):
    L = [f"== {path}",
         f"filetype={p['filetype']} ncmds={p['ncmds']} flags={p['flags']:#x}"]
    for s in p["segments"]:
        L.append(f"SEG {s['name']:16s} vm={s['vmaddr']:#014x}+{s['vmsize']:#x} file={s['fileoff']:#x}+{s['filesize']:#x}")
        for sec in s["sections"]:
            L.append(f"    {sec['name']:24s} @ {sec['addr']:#014x} size {sec['size']:#010x} fileoff {sec['offset']:#x}")
    others = {}
    for lc in p["load_commands"]:
        others.setdefault(hex(lc["cmd"]), 0)
        others[hex(lc["cmd"])] += 1
    L.append(f"other LCs: {others}")
    return "\n".join(L)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = Path(path).read_bytes()
        print(render(parse(d), path))
        print()
