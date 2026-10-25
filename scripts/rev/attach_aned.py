#!/usr/bin/env python3
"""lldb script: dump aned's runtime program containers (needs sudo).

Usage:  sudo lldb --batch -o 'command script import scripts/rev/attach_aned.py'

Breaks on ZinComputeProgramLoader (program load) and dumps the 0xbeefface
container (cachedModel) plus the IOKit wired buffer at 0x100f10000 (64KB×N).

Field map (recovered offline from system .hwx assets):
  magic 0xbeefface | cputype=0x80 cpusub=7 filetype=2 | ncmds | sizeofcmds
  LC_SEGMENT_64: __FVMLIB per-procedure buffers @0x30000000+,
                 __TEXT (command stream) @~0x3xxxx000, __KERN_0 (kernel data)
  LC_THREAD flavor=4: program state, address table (text@0x30054000 ...)
  LC_THREAD flavor=3: per-procedure metadata:
      w[2]=0xd38 (const offset) w[4]=entry w[7]=op_count w[9]=batch
      w[16]=io_bytes w[18]/w[20]=chan sizes w[22]=chan_count
See hwx_parser.py / hwx_e5_correlate.py for the offline decoder.
"""
print(__doc__)
