#!/usr/bin/env python3
"""task-14: decode parseANECOptionFromString's option table from ANECompiler disasm.
Resolves adrp+add string references to actual option names."""
import re
import subprocess
from pathlib import Path

DSC = Path("/Volumes/2T/repos/mss/ane-ref/dsc/ANECompiler")
OUT = Path("/Volumes/2T/repos/mss/pymss-ane/.tmp")

asm = subprocess.run(
    ["objdump", "-d", "--start-address=0x20bda1538", "--stop-address=0x20bda1c00", str(DSC)],
    capture_output=True, text=True).stdout
(OUT / "parse_opt.asm").write_text(asm)

# collect adrp(reg)+add(reg) pairs -> absolute vaddr
refs = []
lines = asm.splitlines()
for i, l in enumerate(lines):
    m1 = re.match(r"\s*[0-9a-f]+:\s+[0-9a-f]+\s+adrp\s+(x\d+), (0x[0-9a-f]+)", l)
    if not m1:
        continue
    reg, page = m1.group(1), int(m1.group(2), 16)
    for j in range(i + 1, min(i + 4, len(lines))):
        m2 = re.search(rf"\s*[0-9a-f]+:\s+[0-9a-f]+\s+add\s+{reg}, {reg}, #(0x[0-9a-f]+)", lines[j])
        if m2:
            refs.append(page + int(m2.group(1), 16))
            break
        # ldr x, [reg, #off] also references
        m3 = re.search(rf"\s*[0-9a-f]+:\s+[0-9a-f]+\s+ldr\s+\w+, \[{reg}, #(0x[0-9a-f]+)\]", lines[j])
        if m3:
            refs.append(page + int(m3.group(1), 16))
            break

# map vaddr -> file offset via segment table (extracted dylib: vaddr notional)
# find __TEXT vmaddr and fileoff
ot = subprocess.run(["otool", "-l", str(DSC)], capture_output=True, text=True).stdout
segs = []
cur = {}
for l in ot.splitlines():
    l = l.strip()
    if l.startswith("segname"):
        cur = {"seg": l.split()[1]}
    elif l.startswith("vmaddr"):
        cur["vmaddr"] = int(l.split()[1], 16)
    elif l.startswith("fileoff"):
        cur["fileoff"] = int(l.split()[1], 16)
    elif l.startswith("filesize"):
        cur["filesize"] = int(l.split()[1], 16)
        if cur.get("seg"):
            segs.append(cur); cur = {}

data = DSC.read_bytes()

def vaddr_read(addr, n=64):
    for s in segs:
        if s["vmaddr"] <= addr < s["vmaddr"] + s["filesize"]:
            off = addr - s["vmaddr"] + s["fileoff"]
            return data[off:off + n]
    return None

print(f"{len(refs)} refs")
seen = set()
for a in refs:
    if a in seen:
        continue
    seen.add(a)
    b = vaddr_read(a)
    if b:
        s = b.split(b"\0")[0]
        if 2 < len(s) < 60:
            try:
                print(hex(a), s.decode())
            except UnicodeDecodeError:
                pass
