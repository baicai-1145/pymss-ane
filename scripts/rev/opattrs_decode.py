#!/usr/bin/env python3
"""task-12: op_attrs differential decode — find the channel-selection switch.

Corpus strategy: compile single-op MIL graphs at fixed shape [1,480,256,256]
across op types (softmax fast-vs-slow candidates, elementwise, reduce, matmul,
conv), extract __Op0_AneInference__op_attrs + __text__ blobs from each .e5,
then diff byte-by-byte to classify bytes as: op-type-varying / shape-varying /
constant. The 50B softmax-vs-conv delta (0x1ca-0x1f9) should decompose into
op-type bits (the switch) vs shape bits.

Also probes: MIL softmax vs exp2 (chain-split hypothesis), and an enum sweep
of softmax axis positions at fixed bytes.

Usage:
  .venv/bin/python scripts/rev/opattrs_decode.py build    # compile corpus (~8 graphs)
  .venv/bin/python scripts/rev/opattrs_decode.py diff     # decode + diff matrix
  .venv/bin/python scripts/rev/opattrs_decode.py all
"""
from __future__ import annotations
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import aneforge as af

from e5_parser import parse  # task-8 parser

CACHE = Path.home() / "Models" / ".aneforge-cache"
OUT = Path("/Volumes/2T/repos/mss/pymss-ane/.tmp/opattrs")
SHAPE = (1, 480, 256, 256)
LOG2E = float(np.log2(np.e))

# op-name -> graph builder (single input SHAPE -> single output)
def _graphs():
  x = af.input(SHAPE)
  return {
    "softmax":   x.softmax(-1),
    "exp2":      x.exp2(),
    "relu":      x.relu(),
    "gelu":      x.gelu(),
    "sigmoid":   x.sigmoid(),
    "mul_scalar": x * 1.5,
    "amax":      x.amax(-1),
    "reduce_sum": x.sum(-1),
    "reduce_mean": x.reduce_mean(-1) if hasattr(x, "reduce_mean") else x.sum(-1),
    "sqrt":      x.sqrt(),
    "silu":      x.silu() if hasattr(x, "silu") else x.sigmoid() * x,
  }

def _extra_graphs():
  # matmul-family + conv at comparable sizes (different natural shapes, still useful)
  x = af.input((1, 512, 1, 15360))
  w = (np.random.default_rng(1).standard_normal((256, 512, 1, 1)) * 0.05).astype(np.float16)
  yield "conv1x1", af.conv(x, w)

  a = af.input((1, 480, 256, 64))
  b = af.input((1, 480, 64, 256))
  yield "matmul4d", a @ b


def build_corpus():
  OUT.mkdir(parents=True, exist_ok=True)
  index = {}
  graphs = _graphs()
  for name in list(graphs) + ["conv1x1", "matmul4d"]:
    try:
      if name in graphs:
        out = graphs[name]
      else:
        out = dict(_extra_graphs())[name]
      # compile() writes the content-addressed cache dir; we then locate the .e5
      net = af.compile(out)
      net.release()
      # find the newest cache entry matching our shape signature via mil text
      hit = None
      for d in sorted(CACHE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        mil = d / "model.mil"
        if not mil.is_file():
          continue
        txt = mil.read_text()[:400]
        if name == "conv1x1" and "conv(" in txt and "512" in txt:
          hit = d; break
        if name == "matmul4d" and "matmul(" in txt and "480" in txt:
          hit = d; break
        if name in graphs and f"{name}(" in txt.replace("reduce_sum(", "sum(") and "480" in txt:
          hit = d; break
      if not hit:
        print(f"{name:12s} cache miss"); continue
      e5s = list(hit.rglob("*.e5"))
      if not e5s:
        print(f"{name:12s} no .e5"); continue
      dst = OUT / f"{name}.e5"
      shutil.copy(e5s[0], dst)
      (OUT / f"{name}.mil").write_text((hit / "model.mil").read_text())
      index[name] = {"e5": str(dst), "cache": str(hit)}
      print(f"{name:12s} -> {dst.name} (from {hit.name})")
    except Exception as e:
      print(f"{name:12s} BUILD FAIL: {str(e)[:90]}")
  (OUT / "index.json").write_text(json.dumps(index, indent=1))
  print(f"\ncorpus at {OUT}")


def blob_of(e5: bytes, kind: str) -> tuple[int, int] | None:
  """Locate a section blob (op_attrs / arg_frame / text) via the section directory."""
  p = parse(e5, "")
  secs = p.get("sections", {})
  # parser gives names + offsets in tail; blob offsets come from header table
  hdr = p.get("blob_offsets") or []
  # map: __text__ -> 0x384 entry known; op_attrs via directory record's offset field
  for name, rec in secs.items():
    if kind in name:
      off = rec.get("blob_off")
      size = rec.get("blob_size")
      if off is not None:
        return off, size or (len(e5) - off)
  return None


def diff_matrix():
  idx = json.loads((OUT / "index.json").read_text())
  blobs = {}
  raws = {}
  for name, meta in idx.items():
    d = (Path(meta["e5"])).read_bytes()
    raws[name] = d
    p = parse(d, str(meta["e5"]))
    blobs[name] = p
  names = sorted(raws)
  ref = raws.get("softmax")
  if ref is None:
    print("no softmax corpus entry"); return
  print(f"corpus: {names}")
  # align-by-size diff: all corpus .e5s should be same-or-close size
  for name in names:
    d = raws[name]
    if len(d) == len(ref):
      diffs = [i for i in range(len(d)) if d[i] != ref[i]]
      # group into runs
      runs, cur = [], None
      for i in diffs:
        if cur and i == cur[1] + 1:
          cur[1] = i
        else:
          cur = [i, i]; runs.append(cur)
      rng = ", ".join(f"0x{a:x}-0x{b:x}({b-a+1}B)" for a, b in runs)
      print(f"{name:12s} vs softmax: {len(diffs):3d} bytes in {len(runs)} runs: {rng}")
    else:
      print(f"{name:12s} vs softmax: SIZE differs ({len(d)} vs {len(ref)})")
  # dump the 0x1ca-0x200 window for all
  print("\n== byte window 0x1c0-0x200 ==")
  for name in names:
    d = raws[name]
    w = d[0x1c0:0x200]
    print(f"{name:12s} {w.hex(' ')}")
  # structured: op_attrs record offsets from parser
  print("\n== parser section offsets ==")
  for name in names:
    p = blobs[name]
    for sname, rec in p.get("sections", {}).items():
      if "op_attrs" in sname or "arg_frame" in sname:
        print(f"{name:12s} {sname}: off={rec.get('blob_off')} size={rec.get('blob_size')}")


def main():
  mode = sys.argv[1] if len(sys.argv) > 1 else "all"
  if mode in ("build", "all"): build_corpus()
  if mode in ("diff", "all"): diff_matrix()


if __name__ == "__main__":
  raise SystemExit(main())
