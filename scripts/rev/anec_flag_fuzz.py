#!/usr/bin/env python3
"""task-13 dir-1: fuzz e5rt set_custom_ane_compiler_options flags.

compile_check(mil, cache, custom_ane_options=...) passes the string straight to
e5rt_e5_compiler_options_set_custom_ane_compiler_options (see _lib/ane_e5rt_dispatch.mm:524).
Known-good: TargetArchitecture=h13|h14|h15|h16s. Probing for ANEC dump/debug flags:
dump-il, dump-asm, emit-llvm, print-after-all, verbose-asm, debug-only, help, etc.
A flag is interesting if rc==0 AND it changes artifacts vs baseline (extra files,
bigger cache) or stdout/stderr output.

Also sweeps: KEY=VALUE forms, space-separated multi-flags, and bare '--help'-ish.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/Volumes/2T/repos/mss/pymss-ane")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aneforge as af  # noqa: F401 (ensures package importable in subprocess env)
PY = str(ROOT / ".venv/bin/python")

FLAGS = [
  # help / list class (a hit here enumerates everything else)
  "--help", "help", "--print-options", "print-options", "--list-options",
  "ListOptions=true", "PrintOptions=true",
  # dump class
  "DumpIL=true", "dump-il=true", "--dump-il", "DumpIR=true", "dump-ir",
  "DumpASM=true", "dump-asm", "--dump-asm", "VerboseASM=true", "verbose-asm",
  "EmitLLVM=true", "emit-llvm", "DumpLLVM=true",
  "DumpMidIR=true", "DumpBackend=true", "DumpSchedule=true", "DumpKernel=true",
  "DumpProgram=true", "DumpNetlist=true", "DumpChannelProgram=true",
  "DumpInstructions=true", "DumpTD=true", "DumpMachO=true", "DumpContainer=true",
  "DumpAll=true", "DumpDir=/tmp/anec_dump", "DumpPath=/tmp/anec_dump",
  # llvm-style
  "print-after-all=true", "--print-after-all", "print-machineinstrs=true",
  "debug-only=ane", "debug-only=espresso", "debug-only=anec", "DebugOnly=anec",
  # verbosity
  "Verbose=true", "verbose=1", "VerboseLevel=9", "LogLevel=debug",
  "Trace=true", "TraceCompilation=true",
  # output redirect guesses
  "OutputDir=/tmp/anec_out", "OutputFile=/tmp/anec_out.bin",
  "KeepIntermediate=true", "KeepIR=true", "EmitIntermediate=true",
  "WriteKernelSection=true", "EmitKernelSection=true",
]

# multi-flag combos (space separated — the e5rt string may take a list)
COMBOS = [
  "TargetArchitecture=h16s DumpIL=true",
  "TargetArchitecture=h16s DumpASM=true",
  "TargetArchitecture=h16s VerboseASM=true",
]


def one_flag(flag: str, mil: Path) -> dict:
  with tempfile.TemporaryDirectory(prefix="anec_fuzz_") as td:
    cache = Path(td) / "cache"
    cmd = [PY, "-B", "-c",
           "import sys; sys.path.insert(0, %r); from aneforge import _runtime as R; "
           "import pathlib; rc = R.compile_check(pathlib.Path(%r), pathlib.Path(%r), custom_ane_options=%r); print('RC', rc)"
           % (str(ROOT), str(mil), str(cache), flag)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    rc_line = [l for l in p.stdout.splitlines() if l.startswith("RC ")]
    rc = int(rc_line[0].split()[1]) if rc_line else -999
    files = sorted(str(f.relative_to(cache)) for f in cache.rglob("*") if f.is_file()) if cache.exists() else []
    return {
      "flag": flag, "rc": rc, "files": files[:6], "n_files": len(files),
      "stdout_tail": p.stdout[-200:], "stderr_tail": p.stderr[-300:],
    }


def main():
  mil = ROOT / ".tmp" / "opattrs" / "softmax.mil"
  if not mil.is_file():
    print("corpus softmax.mil missing"); return 1
  # baseline first
  base = one_flag("TargetArchitecture=h16s", mil)
  print("BASELINE TargetArchitecture=h16s: rc=%d n_files=%d" % (base["rc"], base["n_files"]))
  results = {"baseline": base}
  interesting = []
  for fl in FLAGS + COMBOS:
    try:
      r = one_flag(fl, mil)
    except subprocess.TimeoutExpired:
      r = {"flag": fl, "rc": "TIMEOUT"}
    results[fl] = r
    tag = ""
    if r.get("rc") == 0 and r.get("n_files", 0) != base["n_files"]:
      tag = " *** FILES CHANGED ***"; interesting.append(fl)
    if r.get("rc") not in (0,) and r.get("rc") != base["rc"]:
      tag = " (rc differs)"
    print("%-42s rc=%-6s n_files=%s%s" % (fl[:42], r.get("rc"), r.get("n_files"), tag))
    # any stderr content mentioning the flag => parser feedback = flag name recognized?
    err = (r.get("stderr_tail") or "") + (r.get("stdout_tail") or "")
    if err and fl.lower().strip("=-").split()[0] in err.lower():
      print("    ^ mentioned in output: %s" % err[:160])
  out = ROOT / ".tmp" / "anec_flag_fuzz.json"
  out.write_text(json.dumps(results, indent=1))
  print("\ninteresting:", interesting)
  print("json ->", out)


if __name__ == "__main__":
  raise SystemExit(main())
