#!/usr/bin/env python3
"""anec_flags_inject.py — anec_flags compiler-variant injection (task-18).

DISCOVERED INJECTION SURFACE (verified):
  model.espresso.net (the compiled mlmodelc artifact) "properties" dict:
    {"anec_flags": "SpatialSplitMode=Memory,KernelRewind=false"}
  On MLModel load, aned -> ANECompilerService XPC passes these through to
  zin_ane_compiler (v10.26.6); observable in `log stream --debug` as the
  ANEC v1 flag table (e.g. --fspatial-split=memory).

  MLModelConfiguration.e5rtCustomANECompilerOptions (ObjC property on the
  public CoreML class) is accepted by the setter but did NOT reach the
  compiler request on the neural-network path (optionsFilePath stays null);
  the net-properties channel is the working one.

Usage:
  inject:   anec_flags_inject.py <mlmodelc_dir> "SpatialSplitMode=Memory,KernelRewind=false"
  restore:  anec_flags_inject.py <mlmodelc_dir> --clear

Flag dictionary (ane-recon full table + verified mappings):
  SpatialSplitMode = Auto | Memory | GenericDAG | SpatialSplitDisabled
      -> --fspatial-split=auto|memory|generic-dag|disabled   [VERIFIED]
  KernelRewind = true|false  (also accepts fkernel-rewind=disabled form)
      -> --fkernel-rewind=enabled|disabled                    [VERIFIED]
  OptimizeReductionPerf, DisableInputCoalescing, EnableAggressiveTensorCaching,
  ScanWeightsForCompression, EnableSingleChannelEWOpCopyRemoval (unmapped names
  — check ANEC v1 default table for their --f forms).

Default ANEC flag table captured in .tmp/t18_anec_flags_default.txt (28 flags).
"""
import json
import sys
from pathlib import Path


def inject(modelc_dir: str, flags: str):
    net = Path(modelc_dir) / "model.espresso.net"
    d = json.loads(net.read_text())
    props = d.get("properties") or {}
    if flags:
        props["anec_flags"] = flags
    else:
        props.pop("anec_flags", None)
    d["properties"] = props
    net.write_text(json.dumps(d))
    print(f"anec_flags -> {flags or '(cleared)'} in {net}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inject(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "--clear" else "")
