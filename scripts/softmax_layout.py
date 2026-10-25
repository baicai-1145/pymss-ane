#!/usr/bin/env python3
"""task-11: MIL-internal softmax layout matrix — replicate the netlist fast path
(C=480 fold, reduction on W) with zero dispatch cost.

Hypothesis (task-9 verdict + lead's variant-f): the 3.3x gap (12.43ms MIL vs
3.78ms netlist at the same tensor) is purely Espresso layout/channel routing.
Variants (all on [60,8,256,256] fp16 random data):

  a. baseline        [60,8,256,256] softmax(-1)           (12.43ms reference)
  b. reshape-fold    [1,480,256,256] softmax(-1)          (netlist-isomorphic)
  c. transpose-axes  [1,256,256,480] / [1,256,480,256] softmax variants
  d. 2D flatten      [122880,256] softmax(-1)
  e. axis sweep      softmax(-1/-2/-3) on b
  f. transpose-wrap  [60,8,256,256] -> transpose -> softmax(W) -> transpose back

Each variant: compile -> numeric check (cosine vs numpy > 0.9999) -> p50 ms.

Usage:
  .venv/bin/python scripts/softmax_layout.py            # full matrix
  .venv/bin/python scripts/softmax_layout.py --variant b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import aneforge as af

B, H, S = 60, 8, 256          # sim tensor [60,8,256,256]
N = B * H * S * S             # 31,457,280 elems (63 MB fp16)


def cosine(a, b):
  a = np.asarray(a, np.float32).ravel()
  b = np.asarray(b, np.float32).ravel()
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def bench(net, feeds, n=20, warmup=3):
  for _ in range(warmup):
    net(*feeds)
  t0 = time.perf_counter()
  for _ in range(n):
    net(*feeds)
  return (time.perf_counter() - t0) / n * 1e3


# ---------------------------------------------------------------- variants --

def v_baseline(x):
  """a. [60,8,256,256] softmax(-1) — the 12.43ms reference."""
  return af.compile(x.softmax(-1))


def v_reshape(x):
  """b. reshape to [1,480,256,256] (netlist C=480 fold), softmax(-1)."""
  t = x.reshape(1, B * H, S, S)
  return af.compile(t.softmax(-1))


def v_transpose_c_last(x):
  """c1. transpose to [1,256,256,480]: softmax over the -2 axis (S rows)."""
  t = x.reshape(1, B * H, S, S).transpose([0, 2, 3, 1])   # [1,256,256,480]
  t = t.softmax(-2)                                        # reduce over W' axis (rows)
  return af.compile(t)


def v_transpose_hmid(x):
  """c2. transpose to [1,256,480,256]: softmax(-1) on the trailing 256."""
  t = x.reshape(1, B * H, S, S).transpose([0, 2, 1, 3])   # [1,256,480,256]
  return af.compile(t.softmax(-1))


def v_flatten2d(x):
  """d. 2D [122880,256] softmax(-1)."""
  t = x.reshape(B * H * S, S)
  return af.compile(t.softmax(-1))


def v_axis_neg2(x):
  """e1. [1,480,256,256] softmax(-2) — reduction on H instead of W."""
  t = x.reshape(1, B * H, S, S)
  return af.compile(t.softmax(-2))


def v_axis_neg3(x):
  """e2. [1,480,256,256] softmax(-3) — reduction over channel blocks."""
  t = x.reshape(1, B * H, S, S)
  return af.compile(t.softmax(-3))


def v_transpose_wrap(x):
  """f. 4D -> transpose so reduction axis lands on W -> softmax -> transpose back."""
  t = x.reshape(1, B * H, S, S)                # [1,480,256,256]
  t = t.transpose([0, 2, 3, 1])                # [1,256,256,480] rows in last axis
  t = t.softmax(-2)                            # reduce along the 256-row axis (now H')
  t = t.transpose([0, 3, 1, 2])                # back to [1,480,256,256]
  return af.compile(t.reshape(B, H, S, S))


VARIANTS = {
  "a_baseline": v_baseline,
  "b_reshape_c480": v_reshape,
  "c1_transpose_256256480": v_transpose_c_last,
  "c2_transpose_256480256": v_transpose_hmid,
  "d_flatten2d": v_flatten2d,
  "e1_axis_neg2": v_axis_neg2,
  "e2_axis_neg3": v_axis_neg3,
  "f_transpose_wrap": v_transpose_wrap,
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=sorted(VARIANTS))
  ap.add_argument("--bench-n", type=int, default=20)
  a = ap.parse_args()

  rng = np.random.default_rng(0)
  x = (rng.standard_normal((B, H, S, S)) * 2.0).astype(np.float16)

  # numpy fp32 reference
  x32 = x.astype(np.float32)
  p_ref = np.exp(x32 - x32.max(-1, keepdims=True))
  p_ref /= p_ref.sum(-1, keepdims=True)

  names = [a.variant] if a.variant else list(VARIANTS)
  results = {}
  print(f"softmax layout matrix: {B}x{H}x{S}x{S} = {N/1e6:.1f}M elems ({N*2/1e6:.0f} MB fp16)")
  for name in names:
    fn = VARIANTS[name]
    t0 = time.time()
    try:
      net = fn(af.input((B, H, S, S)))
    except Exception as e:
      print(f"{name:24s} COMPILE FAIL: {str(e)[:110]}")
      results[name] = {"ok": False, "err": str(e)[:200]}
      continue
    ct = time.time() - t0
    try:
      y = net(x)
    except Exception as e:
      print(f"{name:24s} EVAL FAIL: {str(e)[:110]}")
      results[name] = {"ok": False, "err": str(e)[:200], "compile_s": ct}
      continue
    cos = cosine(y, p_ref)
    ms = bench(net, (x,), n=a.bench_n)
    ok = cos > 0.9999
    print(f"{name:24s} {'OK ' if ok else 'BAD'}  cos={cos:.6f}  p50={ms:7.2f} ms  (compile {ct:.0f}s)")
    results[name] = {"ok": ok, "cos": cos, "ms": ms, "compile_s": ct}
    net.release()

  good = {k: v for k, v in results.items() if v.get("ok") and v.get("ms")}
  if good:
    best = min(good, key=lambda k: good[k]["ms"])
    print(f"\nBEST: {best}  {good[best]['ms']:.2f} ms  (baseline ref 12.43 ms)")
  out = Path("/Volumes/2T/repos/mss/pymss-ane/.tmp/softmax_layout.json")
  out.write_text(json.dumps(results, indent=1))
  print(f"json -> {out}")


if __name__ == "__main__":
  raise SystemExit(main())
