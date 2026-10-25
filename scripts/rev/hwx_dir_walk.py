#!/usr/bin/env python3
"""Final validation: keylm __text directory walk (run standalone)."""
import struct

d = open('/Volumes/2T/repos/mss/pymss-ane/.tmp/hwx/keylm.hwx', 'rb').read()
t = d[0xaf478:0xaf478 + 0x28f27c]
i = 0
ends = []
while True:
    o, a, b = struct.unpack_from('<IHH', t, i * 8)
    if a != 0x47 or b != 0x0700:
        break
    ends.append(o)
    i += 1
    if i > 500:
        break
print('keylm dir entries:', i, 'last end:', hex(ends[-1]) if ends else None)
print('first 8 ends:', [hex(e) for e in ends[:8]])
# offsets are cumulative ends relative to data region start (after dir)
starts = [0] + ends[:-1]
sizes = [ends[k] - starts[k] for k in range(i)]
print('first sizes:', [hex(s) for s in sizes[:10]])
