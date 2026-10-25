#!/usr/bin/env python3
"""kern_patch.py — surgical byte-level patch tool for ANE runtime containers (0xbeefface).

Container model (task-10/15 reverse engineering):
  header: 8x u32 (magic, cputype=0x80, cpusub=7, filetype=2, ncmds, sizeofcmds, flags, resv)
  LC_SEGMENT_64 (0x19): __PAGEZERO / __FVMLIB* (per-procedure I/O+const buffers at
      0x30000000+, 4KB granularity) / __TEXT (__text = cumulative-ends directory,
      __const) / __KERN_* (kernel sections; *plaintext* for self-compiled models,
      AES-encrypted for Apple production assets)
  LC_THREAD (0x04): flavor=4 program address table; flavor=3 per-procedure metadata
      (w[2]=const_base=0xd38, w[4]=entry, w[7]=op_count, w[9]=batch,
       w[16]=io_bytes, w[18]/[20]=chan sizes, w[22]=chan_count)

Surgery ladder (task-17):
  L0 no-semantics byte (file re-written, checksums updated) -> identical output
  L1 DMA offset (predictable misplacement of output/input window)
  L2 real MAC/elementwise parameter

Usage:
  python3 kern_patch.py info <container.bin>
  python3 kern_patch.py sections <container.bin> [--kern]
  python3 kern_patch.py patch <in.bin> <out.bin> <segname> <file_off> <hexbytes>
  python3 kern_patch.py dma-shift <in.bin> <out.bin> <segname> <delta>
"""
import struct
import sys
from pathlib import Path

LC_SEGMENT_64 = 0x19
LC_THREAD = 0x04


def load(path):
    return bytearray(Path(path).read_bytes())


def parse_header(d):
    magic = struct.unpack_from("<I", d, 0)[0]
    if magic != 0xBEEFFACE:
        raise ValueError(f"bad magic {magic:#x} (not a 0xbeefface container)")
    vals = struct.unpack_from("<8I", d, 0)
    return {"ncmds": vals[4], "sizeofcmds": vals[5], "raw": vals}


def segments(d):
    """Yield (name, vmaddr, vmsize, fileoff, filesize, nsects, cmd_off, sections)."""
    h = parse_header(d)
    off = 0x20
    for _ in range(h["ncmds"]):
        if off + 8 > len(d):
            break
        cmd, cmdsize = struct.unpack_from("<II", d, off)
        if cmd == LC_SEGMENT_64:
            name = bytes(d[off + 8: off + 24]).rstrip(b"\0").decode("latin1")
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<4Q", d, off + 0x18)
            nsects = struct.unpack_from("<I", d, off + 0x38 + 8)[0]
            yield (name, vmaddr, vmsize, fileoff, filesize, nsects, off)
        off += cmdsize


def threads(d):
    h = parse_header(d)
    off = 0x20
    for _ in range(h["ncmds"]):
        if off + 8 > len(d):
            break
        cmd, cmdsize = struct.unpack_from("<II", d, off)
        if cmd == LC_THREAD:
            flavor, count = struct.unpack_from("<II", d, off + 8)
            yield (flavor, count, off, cmdsize, off + 16)
        off += cmdsize


def kern_sections(d):
    """Kernel sections = segments whose name starts with __KERN (task-3: IsKernelSection)."""
    return [s for s in segments(d) if s[0].startswith("__KERN")]


def text_directory(d):
    """__text directory: (u32 cumulative_end, u16 tag, u16 0x0700) entries."""
    for name, vm, vmsz, foff, fsz, nsects, coff in segments(d):
        if name == "__TEXT":
            # first section is __text; its file offset is in the section table
            so = coff + 72
            secs = []
            for _ in range(nsects):
                sn = bytes(d[so: so + 16]).rstrip(b"\0").decode("latin1")
                saddr, ssize = struct.unpack_from("<2Q", d, so + 32)
                soff = struct.unpack_from("<I", d, so + 56)[0]
                secs.append((sn, saddr, ssize, soff))
                so += 80
            return secs
    return []


def patch_bytes(d, file_off, hexstr):
    b = bytes.fromhex(hexstr.replace(" ", ""))
    d[file_off: file_off + len(b)] = b
    return len(b)


def info(d):
    h = parse_header(d)
    segs = list(segments(d))
    th = list(threads(d))
    fl3 = [t for t in th if t[0] == 3]
    fl4 = [t for t in th if t[0] == 4]
    print(f"ncmds={h['ncmds']} sizeofcmds={h['sizeofcmds']:#x}")
    print(f"segments={len(segs)} threads: flavor4={len(fl4)} flavor3={len(fl3)} (procedures)")
    kern = kern_sections(d)
    print(f"kernel sections: {len(kern)}")
    for k in kern:
        print(f"  {k[0]} vm={k[1]:#x}+{k[2]:#x} file={k[3]:#x}+{k[4]:#x}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, path = sys.argv[1], sys.argv[2]
    d = load(path)
    if cmd == "info":
        info(d)
    elif cmd == "sections":
        for name, vm, vmsz, foff, fsz, nsects, coff in segments(d):
            print(f"{name:16s} vm={vm:#014x}+{vmsz:#010x} file={foff:#010x}+{fsz:#010x} nsects={nsects}")
    elif cmd == "textdir":
        for sn, saddr, ssize, soff in text_directory(d):
            print(f"{sn:12s} addr={saddr:#x} size={ssize:#x} fileoff={soff:#x}")
    elif cmd == "patch":
        out, seg, foff_s, hexb = sys.argv[3:7]
        n = patch_bytes(d, int(foff_s, 0), hexb)
        Path(out).write_bytes(d)
        print(f"patched {n} bytes at file offset {int(foff_s, 0):#x} -> {out}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---- TD record surgery (ane-guide ch23 + task-3 ISA) ------------------------
# 44B register-write records: [count u32][reg u32][addr0 u64][addr1 u64]? —
# exact framing locked against task-16 plaintext dump. Reloc regs (v10/M1):
#   0x1344 input tile | 0x134a second operand | 0x1442 output tile write
#   0x1554 bias | 0x1558 post-scale | 0x155c palette | 0x1560 activation LUT
# NOTE: M4/H16G uses a different versioned image (field offsets move);
# validate against dump before patching.

RELOC_REGS = {
    0x1344: "input_tile_base",
    0x134A: "src2_base",
    0x1442: "output_tile_base",
    0x1554: "bias_stream",
    0x1558: "postscale_stream",
    0x155C: "palette_lut",
    0x1560: "activation_lut",
}


def find_td_records(d, kern_off, kern_len):
    """Scan a kernel section for 44B register-write records (framing TBD)."""
    # Heuristic v0: look for the known reloc register addresses as u32.
    hits = []
    for off in range(kern_off, kern_off + kern_len - 4, 4):
        v = struct.unpack_from("<I", d, off)[0]
        if v in RELOC_REGS:
            hits.append((off, v, RELOC_REGS[v]))
    return hits


# ---- M4 rt command-stream decoder (task-3 §3: ZinRtCommandIterator protocol) --
def decode_command_stream(u32_words):
    """Yield per-command dicts: word0 -> N values -> word0 -> ...

    type1 (sequential): w0>>29 !=0, ((w0>>30)&2)==0
        N = ((w0>>15)&0x3f)+1, FieldNumber = w0 & 0x7fff (base byte addr = fn*4)
    type2 (mask): ((w0>>30)&2)!=0
        mask = (w0>>15)&0xffff, N = popcount(mask)-1
    terminator: w0>>29 == 0
    """
    i, n = 0, len(u32_words)
    while i < n:
        w0 = u32_words[i]
        if (w0 >> 29) == 0:
            break
        ctype = 2 if ((w0 >> 30) & 2) else 1
        if ctype == 1:
            count = ((w0 >> 15) & 0x3F) + 1
        else:
            mask = (w0 >> 15) & 0xFFFF
            count = bin(mask).count("1") - 1
            if count < 0:
                break
        vals = list(u32_words[i + 1: i + 1 + count])
        yield {"off": i, "w0": w0, "type": ctype, "field": w0 & 0x7FFF,
               "count": count, "values": vals}
        i += 1 + count


# ---- anehash (content-address key for bundles) ------------------------------
def recompute_anehash(e5_bytes: bytes, container_bytes: bytes) -> str:
    """anehash = SHA256(e5)_SHA256(runtime container), uppercase hex, '_' joined.

    Differential evidence (task-17, c2/c4/c5/c_softmax dumps): first half
    identical across ops (same E5), second half differs per op (container
    carries op identity). Disk loads verify via verifyBundleAtPath
    (AppleNeuralEngine, dsc extract) -> patching a bundle file requires
    recomputing this key.
    """
    import hashlib
    h1 = hashlib.sha256(e5_bytes).hexdigest().upper()
    h2 = hashlib.sha256(container_bytes).hexdigest().upper()
    return f"{h1}_{h2}"
