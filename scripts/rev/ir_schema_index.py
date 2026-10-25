#!/usr/bin/env python3
"""ir_schema_index.py — ANEC IR plist schema index (static pre-study, task-17 IR form).

Source of truth when the dump lands:
  ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR=1 makes Espresso's
  ANECompilerEngine::dump_in_memory_ir_for_debugging save per-segment:
    - ANEC IR plist  (net.plist — netplist-family format, ANECompiler-side parser)
    - weights blob
    - compiler options (net_options.plist / online_options.plist)
  to a tmpdir laid out as /anecir.XXXXXXXXX/{net.plist,net_options.plist,...}.

Pre-study keys (dsc ANECompiler strings + aneforge _bridges/_netplist.py
cross-reference) — the IR plist speaks the netplist family:
  network level: InputNetworks, NetworkPlistName, NetworkPlistPath,
    NetworkWeights, NetworkSourcePath, OutputFileName, WeightFileName,
    WeightFilePath, net.additional.weights, Constants, Weights, Units
  unit level:    Unit/Bottoms/Top/Type/Params/BottomName/BottomTaskIDs/
    BottomProducingUnitName, unit_parsing_errors
  tensor level:  Channels/Height/Width/Depth/Index/ByteOffset/Interleave
    (Constant* family), Input/Output Dims N D C H W, Axis types, TaskIDs
  conv family:   KernelX/KernelY (kernelX, kernelY strings), Stride/Pad/Dilation/
    Group/ChannelWiseConv/ChannelWiseDilatedConv variants
  reduce family: ChannelArgMax/ChannelArgMin/ChannelMax/ChannelMin,
    TopK, CoordinateMode
  procedure:     procedure_name, CompileProcedure, Procedure, segments,
    Segmenter.mil, SegmentsIOTransformer.mil, segment_io_trace_dir

Usage:
  python3 ir_schema_index.py dump_dir   # diff a fresh dump against this index
"""
import sys
from pathlib import Path

INDEX = dict(
    network=["InputNetworks", "NetworkPlistName", "NetworkPlistPath",
             "NetworkName", "Networks", "InputList", "OperationList",
             "NetworkWeights", "NetworkSourcePath", "OutputFileName",
             "WeightFileName", "WeightFilePath", "net.additional.weights",
             "Constants", "Weights", "Units"],
    unit=["Unit", "Bottoms", "Top", "Type", "Params", "BottomName",
          "BottomTaskIDs", "BottomProducingUnitName", "Bottom",
          "OperationName", "ProcedureList", "Version"],
    tensor=["Channels", "Height", "Width", "Depth", "Index", "ByteOffset",
            "BatchSize", "Dimension", "InputChannels", "InputDepth",
            "InputHeight", "InputInterleave", "InputName", "InputType",
            "InputWidth", "Name", "OutputChannels", "OutputInterleave",
            "OutputList", "OutputName", "OutputType",
            "Interleave", "ConstantChannels", "ConstantHeight", "ConstantWidth",
            "ConstantName", "CreateOutputTensor"],
    conv=["KernelX", "KernelY", "ChannelWiseConv", "ChannelWiseDilatedConv",
          "CompressSparseKernels", "CreateConcatForCircularBuffer"],
    reduce=["ChannelArgMax", "ChannelArgMin", "ChannelMax", "ChannelMin",
            "TopK", "CoordinateMode"],
    procedure=["procedure_name", "CompileProcedure", "Procedure", "segments",
               "Segmenter.mil", "SegmentsIOTransformer.mil",
               "segment_io_trace_dir"],
    options=["online_options.plist", "net_options.plist",
             "CostModelClusterThreshold"],
)


def diff_dump(dump_dir: str):
    import plistlib
    d = Path(dump_dir)
    plists = list(d.rglob("*.plist"))
    print(f"dump dir: {d} ({len(plists)} plists)")
    known = {k for v in INDEX.values() for k in v}
    for p in plists:
        try:
            obj = plistlib.loads(p.read_bytes())
        except Exception as e:
            print(f"  {p.name}: unreadable ({e})")
            continue
        keys = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.add(str(k))
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(obj)
        new = keys - known
        print(f"  {p.name}: {len(keys)} keys, {len(new)} not in index")
        for k in sorted(new)[:40]:
            print(f"    + {k}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        diff_dump(sys.argv[1])
    else:
        for cat, ks in INDEX.items():
            print(f"[{cat}]")
            for k in ks:
                print(f"  {k}")
