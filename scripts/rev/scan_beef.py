#!/usr/bin/env python3
"""Scan own-process memory for the ANE runtime container (0xbeefface), then
FOLLOW pointer fields into the pointed-to blobs (the text/weights sections live
in separate malloc'd blocks referenced by the container header)."""
import ctypes, os, struct, sys

MAGIC = b"\xce\xfa\xef\xbe"

def scan_all(task, kern, VMRegionSubmapInfo64):
    addr = ctypes.c_uint64(0); size = ctypes.c_uint64(0)
    nesting = ctypes.c_uint32(0); info = VMRegionSubmapInfo64()
    count = ctypes.c_uint32(ctypes.sizeof(VMRegionSubmapInfo64) // 4)
    regions = []
    seen = set()
    while True:
        kr = kern.mach_vm_region_recurse(task, ctypes.byref(addr), ctypes.byref(size),
                                         ctypes.byref(nesting), ctypes.byref(info),
                                         ctypes.byref(count))
        if kr != 0: break
        a, s = addr.value, size.value
        if a in seen:
            addr.value = a + max(s, 0x1000); continue
        seen.add(a)
        if (info.protection & 1) and 0 < s < (1 << 30):
            regions.append((a, s))
        addr.value = a + max(s, 0x1000)
    return regions

_TASK = [0]
_libc = None

def read_mem(addr, n):
    """Safe read via mach_vm_read: returns b'' on failure instead of SIGBUS."""
    _libc.mach_vm_read.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64,
                                   ctypes.POINTER(ctypes.POINTER(ctypes.c_uint)), ctypes.POINTER(ctypes.c_uint)]
    data = ctypes.POINTER(ctypes.c_uint)()
    dataCnt = ctypes.c_uint(0)
    kr = _libc.mach_vm_read(_TASK[0], addr, n, ctypes.byref(data), ctypes.byref(dataCnt))
    if kr != 0 or not dataCnt.value:
        return b""
    try:
        return ctypes.string_at(data, dataCnt.value)
    finally:
        _libc.vm_deallocate(_TASK[0], ctypes.cast(data, ctypes.c_void_p), dataCnt.value)

def find_in_regions(regions, pat, limit=64):
    hits = []
    for a, s in regions:
        CH = 1 << 22
        for off in range(0, s, CH):
            n = min(CH, s - off)
            d = read_mem(a + off, n)
            if not d: continue
            pos = 0
            while True:
                i = d.find(pat, pos)
                if i < 0: break
                hits.append(a + off + i)
                pos = i + 1
                if len(hits) >= limit: return hits
    return hits

def main():
    import aneforge as af
    import numpy as np
    op = sys.argv[1] if len(sys.argv) > 1 else "conv"
    if op == "conv":
        x = af.input((1, 4, 1, 4)); w = np.zeros((8, 4, 1, 1), dtype=np.float16); w[0, 0, 0, 0] = 1.0
        y = af.conv(x, w)
    elif op == "reduce_mean":
        x = af.input((1, 8, 1, 64)); y = x.mean((2,))
    elif op == "softmax":
        x = af.input((1, 8, 1, 64)); y = x.softmax(axis=-1)
    else:
        raise SystemExit("op?")
    m = af.compile(y)
    print("compiled", op, flush=True)
    # EXECUTE once: program load to aned happens lazily at first run — that's when
    # the runtime container (cachedModel) gets built inside OUR process (Espresso
    # marshals it) or in aned. Scan our process right after.
    rng = np.random.default_rng(0)
    xin = np.zeros((1, 4, 1, 4), dtype=np.float16) if op == "conv" else np.zeros((1, 8, 1, 64), dtype=np.float16)
    xin = rng.standard_normal(xin.shape).astype(np.float16)
    out = m(xin)
    print("executed", type(out), out.shape if hasattr(out,'shape') else '', flush=True)

    libc = ctypes.CDLL(None)
    globals()['_libc'] = libc
    _TASK[0] = libc.mach_task_self()
    kern = ctypes.CDLL(None)
    kern.mach_vm_region_recurse.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]

    class I(ctypes.Structure):
        _fields_ = [("protection", ctypes.c_uint32), ("max_protection", ctypes.c_uint32),
                    ("inheritance", ctypes.c_uint32), ("shared", ctypes.c_uint32),
                    ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64),
                    ("behavior", ctypes.c_uint32), ("user_wired_count", ctypes.c_uint16),
                    ("pages_resident", ctypes.c_uint16), ("pages_shared_now_private", ctypes.c_uint16),
                    ("pages_swapped_out", ctypes.c_uint16), ("pages_dirtied", ctypes.c_uint16),
                    ("ref_count", ctypes.c_uint32), ("shadow_depth", ctypes.c_uint16),
                    ("external_pager", ctypes.c_uint8), ("share_mode", ctypes.c_uint8),
                    ("is_submap", ctypes.c_uint32), ("count", ctypes.c_uint32)]

    task = libc.mach_task_self()
    regions = scan_all(task, kern, I)
    print(f"{len(regions)} regions", flush=True)

    hits = find_in_regions(regions, MAGIC, limit=8)
    print("magic hits:", [hex(h) for h in hits], flush=True)

    outdir = os.environ.get("ANE_GRAB_DIR", "/tmp/ane_grab")
    os.makedirs(outdir, exist_ok=True)
    tag = op
    for n, h in enumerate(hits):
        d = read_mem(h, 0x100)
        if len(d) < 0x40 or d[8:13] != b"MAGIC":   # container header form
            continue
        # walk entries: [u64 count][u64 ptr][u64 len][u64 -1][16B name]
        off = 0x10
        blob = read_mem(h, 1 << 16)
        while off + 0x30 <= len(blob):
            a, b, c, e = struct.unpack_from("<QQQQ", blob, off)
            if e != 0xFFFFFFFFFFFFFFFF: break
            nm = blob[off + 0x20: off + 0x30].split(b"\0")[0].decode("latin1")
            if a == 1 and c < (1 << 24):
                data = read_mem(b, c)
                if data:
                    fn = f"{outdir}/{tag}_container{n}_{nm}.bin"
                    open(fn, "wb").write(data)
                    print(f"  entry {nm!r}: ptr={b:#x} len={c} -> {fn}", flush=True)
            off += 0x30
        # full header dump too
        open(f"{outdir}/{tag}_container{n}_header.bin", "wb").write(read_mem(h, off + 0x30))

    # Also scan whole heap for the opcode fingerprints + segment signatures
    for sig, name in ((struct.pack("<I", 0x30010000), "text@0x30010000"),
                      (struct.pack("<I", 0x5021), "reg0x5021")):
        hh = find_in_regions(regions, sig, limit=4)
        print(name, [hex(x) for x in hh], flush=True)

if __name__ == "__main__":
    main()
