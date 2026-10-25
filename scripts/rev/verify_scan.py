#!/usr/bin/env python3
"""Verify magic scan works (run after compile+execute): fake-magic implant test."""
import ctypes, struct
import aneforge as af, numpy as np

x = af.input((1, 4, 1, 4)); w = np.zeros((8, 4, 1, 1), dtype=np.float16); w[0, 0, 0, 0] = 1.0
m = af.compile(af.conv(x, w))
out = m(np.zeros((1, 4, 1, 4), dtype=np.float16))
print('ran')

libc = ctypes.CDLL(None)
_TASK = libc.mach_task_self()

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

kern = ctypes.CDLL(None)
kern.mach_vm_region_recurse.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(I), ctypes.POINTER(ctypes.c_uint32)]

addr = ctypes.c_uint64(0); size = ctypes.c_uint64(0); nesting = ctypes.c_uint32(0)
info = I(); count = ctypes.c_uint32(ctypes.sizeof(I) // 4)
regions = []
while True:
    kr = kern.mach_vm_region_recurse(_TASK, ctypes.byref(addr), ctypes.byref(size),
                                     ctypes.byref(nesting), ctypes.byref(info), ctypes.byref(count))
    if kr != 0: break
    if (info.protection & 1) and 0 < size.value < (1 << 30):
        regions.append((addr.value, size.value))
    addr.value += max(size.value, 0x1000)

# implant fake magic
fake = ctypes.create_string_buffer(0x100)
ctypes.memmove(fake, struct.pack('<I8sQ', 0xBEEFFACE, b'MAGIC', 0), 20)
print('fake @', hex(ctypes.addressof(fake)))

MAGIC = b'\xce\xfa\xef\xbe'
hits = []
for a, s in regions:
    CH = 1 << 22
    for off in range(0, s, CH):
        n = min(CH, s - off)
        try:
            d = bytes((ctypes.c_char * n).from_address(a + off))
        except Exception:
            continue
        pos = 0
        while True:
            i = d.find(MAGIC, pos)
            if i < 0: break
            hits.append(a + off + i); pos = i + 1
            if len(hits) >= 8: break
        if len(hits) >= 8: break
    if len(hits) >= 8: break
print('regions:', len(regions), 'magic hits:', [hex(h) for h in hits])
