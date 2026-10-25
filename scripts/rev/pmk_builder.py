#!/usr/bin/env python3
"""pmk_builder.py — RtRuntimePatchMutableKernelOperationDescription constructor draft.

Recovered layouts (aned disassembly, task-17 recon):

RtOperationDescription common header (RtOperationGraph::OperationNode):
  +0x08: u32 kind      (0 = Runtime op; nonzero = expression/const)
  +0x28: u32 opcode    (3 = PatchMutableKernel; 9 = expression w/ +0x2c subkind:
                        0=immediate, 1/0x10=unary variants)
  +0x2c: u32 expr_kind (expressions only)

RtRuntimePatchMutableKernelOperationDescription (0x70 bytes, from Clone @0x10003cc94):
  +0x00: vtable (PAC-diversified)
  +0x08: 0 / +0x10: 0
  +0x18: std::string {data_ptr_or_sso, length} — target kernel-section symbol name
         (Clone copies this and inserts the name into a std::set<string> —
          the "mutable section owner" set used by
          ZinComputeProgramRTProcedureIsMutableSectionOwner)
  +0x28: u32 opcode = 3  (set unconditionally by Clone)
  +0x30..0x57: zeroed qwords ×3 — expression operand slots (patch address/value exprs)
  +0x58: (within the qword range) reserved
  +0x60: u64 = 0
  +0x68: u8 flag = 1 (enabled/active)

Consumers (aned side):
  ZinComputeProgramGetMutableKernelSectionForProcedure(program, procid, out) @0x100047e74
  ZinComputeProgramRTGetMutableKernelSectionForProcedure @0x100048158
  ZinComputeProgramRTProcedureIsMutableSectionOwner(node, bool&) @0x100047480
  ZinComputeProgramRTGetPatchMutableKernelOperationsDescription(node, span&) @0x10004664c
    → walks node->[0x18] operation list; picks op with {+0x8:0, +0x28:3}

Serialization note (task-3 §5): in the container's rt command stream the
PatchMutableKernel op appears via field-number dispatch; procedure-level
field numbers 0x200-0x22D carry the op list. The in-container opcode may
differ from the in-memory opcode 3 (task-3 lists serialized op index 12).

This module builds the *serialized* form for hwx injection experiments:
see build_pmk_record() for the TLV framing used by kern_patch.py surgery.
"""
import struct


def build_pmk_record(section_symbol: str, patch_offset: int, patch_value: int) -> bytes:
    """Serialize one PatchMutableKernel record for container surgery.

    Container form (subject to task-16 dump validation): an operation record
    inside the procedure's op-list, framing = [tag u16][len u16] TLV, fields
    little-endian u32/u64. Exact framing to be locked against a real
    self-compiled plaintext container (task-16 dependency).

    section_symbol: target kernel section name (e.g. "__kern_0_conv0")
    patch_offset:   byte offset inside the mutable kernel section
    patch_value:    value to write (u64)
    """
    name = section_symbol.encode()
    body = b"".join([
        struct.pack("<I", 3),                 # opcode = PatchMutableKernel (in-memory)
        struct.pack("<I", len(name)),
        name,
        struct.pack("<Q", patch_offset),
        struct.pack("<Q", patch_value),
        struct.pack("<B", 1),                 # enabled
    ])
    tag = 0x0403                              # provisional: op-record tag, kind=4(op) sub=3
    return struct.pack("<HH", tag, len(body)) + body


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        r = build_pmk_record("__kern_0_main", 0x1234, 0xDEADBEEF)
        print(r.hex(" "))
        print("len:", len(r))
