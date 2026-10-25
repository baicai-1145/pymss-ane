#!/usr/bin/env python3
"""task-14: dump string references from a disassembly window of ANECompiler.
Usage: strrefs.py <start-hex> <end-hex> [max]"
"""
import re
import subprocess
import sys
from collections import Counter

DSC = "/Volumes/2T/repos/mss/ane-ref/dsc/ANECompiler"

start = int(sys.argv[1], 16)
end = int(sys.argv[2], 16)
asm = subprocess.run(["objdump", "-d", f"--start-address={start}", f"--stop-address={end}", DSC],
                     capture_output=True, text=True).stdout
ot = subprocess.run(["otool", "-l", DSC], capture_output=True, text=True).stdout
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
data = open(DSC, "rb").read()


def rd(a, n=48):
    for s in segs:
        if s["vmaddr"] <= a < s["vmaddr"] + s["filesize"]:
            o = a - s["vmaddr"] + s["fileoff"]
            return data[o:o + n]
    return None


# pre-index all adrp page values: adrp xN, #page ; then any 'add xN, xN, #off' or 'ldr xM,[xN,#off]' within 5 lines
lines = asm.splitlines()
out = []
for i, l in enumerate(lines):
    m1 = re.match(r"\s*[0-9a-f]+:\s+[0-9a-f]+\s+adrp\s+(x\d+), (0x[0-9a-f]+)", l)
    if not m1:
        continue
    reg, page = m1.group(1), int(m1.group(2), 16)
    for j in range(i + 1, min(i + 6, len(lines))):
        m2 = re.search(rf"(?:add\s+{reg},\s*{reg},\s*#(0x[0-9a-f]+))|(?:ldr\s+x\d+,\s+\[{reg},\s*#(0x[0-9a-f]+)\])", lines[j])
        if m2:
            off = int(m2.group(1) or m2.group(2), 16)
            b = rd(page + off)
            if b:
                s = b.split(b"\0")[0]
                if 1 < len(s) < 60:
                    try:
                        out.append(s.decode())
                    except UnicodeDecodeError:
                        pass
            break
uniq = [k for k, _ in Counter(out).most_common()]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
print(uniq[:n])
