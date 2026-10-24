#!/usr/bin/env python3
"""task-2: systematic netplist op-template scan on M4 (h16g).

Enumerates every build_* op in aneforge/_bridges/_netplist.py, generates a
minimal net.plist + weights, compiles/executes via layer_invoker (Path A:
AppleNeuralEngine.framework private classes), and records OK/FAIL + error.

Also:
  --mode limits   parameter-limit binary scan (conv width/channels, softmax width)
  --mode sdpa     SDPA Type-field spelling variants
  --mode chain    multi-unit conv->elementwise->conv chain compile test
  --mode softmax-bench  netplist softmax latency vs MIL-route figure

Usage:
  .venv/bin/python scripts/netlist_scan.py --mode ops [--op NAME] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import plistlib
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pymss-ane" / ".venv" / "lib" / "python3.12" / "site-packages"))
sys.path.insert(0, "/Volumes/2T/repos/mss/pymss-ane/.venv/lib/python3.12/site-packages")

import aneforge._bridges._netplist as NL  # noqa: E402
from aneforge._bridges._netplist import (  # noqa: E402
    build, build_plist, ensure_invoker, input_entry, output_entry,
    weights_for_op, write_model,
)

def _nl_invoke(invoker, net_plist, **kwargs):
  """Robust invoke: fetch the pruned/oddly-keyed invoke_netplist from _netplist."""
  import aneforge._bridges._netplist as m
  for k, v in vars(m).items():
    if callable(v) and getattr(v, "__name__", "") == "invoke_netplist":
      return v(invoker, net_plist, **kwargs)
  raise RuntimeError("invoke_netlist missing from aneforge._bridges._netplist")

VENV_SITE = Path("/Volumes/2T/repos/mss/pymss-ane/.venv/lib/python3.12/site-packages")

# The full op list from _netplist.main() choices, derived from the module tables.
ALL_OPS = sorted(set(list(NL.UNARY_NEURONS) + list(NL.UNARY_ELEMENTWISE)
    + list(NL.ELEMENTWISE) + list(NL.NATIVE_NETPLIST_OPS) + list(NL.RUNTIME_CONST_TEMPLATE_OPS) + [
    "elu", "leaky_relu", "relu6", "clamped_relu", "prelu", "silu", "softsign",
    "thresholded_relu", "linear_activation", "hard_sigmoid", "hard_swish",
    "div", "floor_div", "less_equal", "greater_equal", "softmax",
    "maxpool2x2", "avgpool2x2", "reduce_sum", "reduce_mean", "reduce_max",
    "reduce_min", "slice", "upsample_nearest", "concat", "reshape",
    "reshape_alias_3d", "slice_update_concat_equiv", "transpose", "matmul",
    "conv1x1", "add_relu", "conv_relu", "conv_gelu", "conv1x1_gelu",
    "conv_add_relu", "relu_chain", "conv_chain", "sdpa",
]))

REPORT = Path("/Volumes/2T/repos/mss/.pi/herdr-team/6kq413/reports")


def run_one(op: str, *, width=4, height=1, channels=1, depth=1, extra_kwargs=None) -> dict:
  """Generate + compile + run one op netplist. Returns {op, ok, err, ms}."""
  t0 = time.time()
  try:
    invoker = ensure_invoker("layer_invoker")
    with tempfile.TemporaryDirectory(prefix=f"ane_scan_{op}_") as d:
      cd = Path(d)
      kwargs = dict(extra_kwargs or {})
      write_model(op, cd, depth=depth, width=width, height=height, channels=channels, **kwargs)
      # small deterministic fp16 input covering plausible sizes
      n_in = max(width * height * channels, 16)
      x = np.linspace(-1, 1, n_in, dtype=np.float32).astype(np.float16)
      (cd / "in_x.bin").write_bytes(x.tobytes())
      info = _nl_invoke(
        invoker, cd / "net.plist",
        weights=sorted(cd.glob("weights.*")),
        inputs=[("x", cd / "in_x.bin")],
        outputs=[("y", cd / "out_y.bin")],
        warmup=0,
      )
      return {"op": op, "ok": True, "err": "", "ms": info.get("elapsedMs", info.get("ms"))}
  except Exception as e:
    msg = str(e)
    # keep the interesting part of compile errors
    for marker in ("error:", "failed", "unsupported", "Unsupported", "invalid", "Invalid"):
      i = msg.find(marker)
      if i >= 0:
        msg = msg[max(0, i - 40): i + 200]
        break
    return {"op": op, "ok": False, "err": msg[:400], "ms": None}
  finally:
    pass


def mode_ops(only: str | None, out_json: Path | None):
  results = []
  ops = [only] if only else ALL_OPS
  print(f"scanning {len(ops)} ops (layer_invoker Path-A compile+run)...")
  for i, op in enumerate(ops):
    r = run_one(op)
    results.append(r)
    tag = "OK  " if r["ok"] else "FAIL"
    print(f"[{i+1:3d}/{len(ops)}] {tag} {op:36s} {('%.3fms' % r['ms']) if r['ms'] else ''} {r['err'][:80]}")
  ok = [r["op"] for r in results if r["ok"]]
  fail = [r["op"] for r in results if not r["ok"]]
  print(f"\n== OK ({len(ok)}): {ok}")
  print(f"== FAIL ({len(fail)}): {fail}")
  if out_json:
    out_json.write_text(json.dumps(results, indent=1))
    print(f"json -> {out_json}")


def mode_limits(out_json: Path | None):
  """Binary-scan parameter limits for conv1x1 (channels), softmax/reduction (width)."""
  results = {}

  def scan(op, param, values, fixed=None, label=None):
    ok_vals, fails = [], {}
    for v in values:
      kw = {param: v}
      if fixed: kw.update(fixed)
      r = run_one(op, **kw)
      (ok_vals if r["ok"] else fails).__setitem__(v, "" if r["ok"] else r["err"][:120]) if False else None
      if r["ok"]: ok_vals.append(v)
      else: fails[v] = r["err"][:120]
      print(f"  {label or op} {param}={v}: {'OK' if r['ok'] else 'FAIL ' + r['err'][:60]}")
    results[f"{op}.{param}"] = {"ok": ok_vals, "fail": fails}

  print("== conv1x1 channels scan ==")
  scan("conv1x1", "channels", [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
  print("== conv1x1 width scan ==")
  scan("conv1x1", "width", [4, 16, 64, 256, 1024, 4096, 16384, 32768])
  print("== softmax width scan ==")
  scan("softmax", "width", [4, 16, 64, 256, 1024, 4096, 16384, 32768])
  print("== reduce_sum width scan ==")
  scan("reduce_sum", "width", [4, 16, 64, 256, 1024, 4096, 16384, 32768])
  print("== softmax channels scan ==")
  scan("softmax", "channels", [1, 4, 16, 64, 256, 1024])
  if out_json:
    out_json.write_text(json.dumps(results, indent=1))
    print(f"json -> {out_json}")


def mode_sdpa():
  """SDPA Type-field spelling variants: does any alias template exist on M4?"""
  import copy
  variants = ["SDPA", "Sdpa", "sdpa", "Attention", "SelfAttention", "ScaledDotProductAttention",
              "scaled_dot_product_attention", "FlashAttention", "SDPAv2", "SDPA_V2", "SoftmaxAttention",
              "DotProductAttention", "AttentionFused", "FusedAttention", "CausalSDPA", "SDPA_Causal"]
  base = build("sdpa")  # raises not: sdpa builder exists; compile fails on M4 per lead
  invoker = ensure_invoker("sdpa_invoker")
  print(f"{'variant':30s} result")
  for v in variants:
    pl = copy.deepcopy(base)
    net = next(iter(pl["Networks"]))
    for u in pl[net]["Units"]:
      if pl[net][u].get("Type") == "SDPA":
        pl[net][u]["Type"] = v
    try:
      with tempfile.TemporaryDirectory(prefix="ane_sdpa_") as d:
        cd = Path(d)
        with (cd / "net.plist").open("wb") as f:
          plistlib.dump(pl, f, sort_keys=True)
        (cd / "weights.0").write_bytes(b"")
        (cd / "weights.1").write_bytes(NL._float16_payload(1.0 / 4.0, 1))
        (cd / "in_q.bin").write_bytes(np.zeros(64 * 16 * 16, np.float16).tobytes())
        (cd / "in_k.bin").write_bytes(np.zeros(64 * 16 * 16, np.float16).tobytes())
        (cd / "in_v.bin").write_bytes(np.zeros(64 * 16 * 16, np.float16).tobytes())
        info = _nl_invoke(
          invoker, cd / "net.plist",
          weights=[cd / "weights.0", cd / "weights.1"],
          inputs=[("query", cd / "in_q.bin"), ("key", cd / "in_k.bin"), ("value", cd / "in_v.bin")],
          outputs=[("y", cd / "out_y.bin")], warmup=0)
        print(f"{v:30s} OK!!!  {info}")
    except Exception as e:
      msg = str(e)
      i = max(msg.find("error"), msg.find("Error"), 0)
      print(f"{v:30s} fail: {msg[i:i+120]}")
      print(f"  -> no alias template for {v!r}")


def _conv_unit(name: str, bottom: str, channels: int, out_channels: int) -> dict:
  return {"Bottom": [bottom], "InputType": ["Float16"], "Name": name,
          "NumGroups": 1, "OutputChannels": out_channels, "OutputType": "Float16",
          "Params": {"KernelDepth": 1, "KernelGroupReuse": False, "KernelHeight": 1,
                     "KernelIndex": 1, "KernelMode": "Dense", "KernelMutable": False,
                     "KernelOffset": 0, "KernelType": "Float16", "KernelWidth": 1,
                     "PadBack": 0, "PadBot": 0, "PadFront": 0, "PadLeft": 0, "PadRight": 0,
                     "PadTop": 0, "PaddingMode": "Zero", "Step": [1, 1, 1], "Type": "Conv"},
          "Type": "Conv"}


def mode_chain():
  """conv -> elementwise(mul) -> conv two/three-unit chain: compile-through test (task-4 route A precondition)."""
  invoker = ensure_invoker("layer_invoker")
  for n_units, layout in [(2, "conv+mul"), (3, "conv+mul+conv"), (3, "conv+relu+conv")]:
    net = "network_chain"
    units = {}
    units["c0"] = _conv_unit("c0", "x", 4, 4)
    if layout == "conv+mul":
      units["m1"] = {"Bottom": ["c0", "x2"], "InputType": ["Float16", "Float16"], "Name": "m1",
                     "OutputChannels": 4, "OutputType": "Float16",
                     "Params": {"Type": "Mult"}, "Type": "ElementWise"}
      inputs = [input_entry("x", width=4, entry_name="c0"),
                input_entry("x2", width=4, channels=4, entry_name="m1")]
      out_unit = "m1"
    else:
      mid = "relu" if layout.endswith("relu+conv") else "mul"
      if mid == "relu":
        units["m1"] = {"Bottom": ["c0"], "InputType": ["Float16"], "Name": "m1",
                       "OutputChannels": 4, "OutputType": "Float16",
                       "Params": {"Type": "ReLU"}, "Type": "Neuron"}
      else:
        units["m1"] = {"Bottom": ["c0", "x2"], "InputType": ["Float16", "Float16"], "Name": "m1",
                       "OutputChannels": 4, "OutputType": "Float16",
                       "Params": {"Type": "Mult"}, "Type": "ElementWise"}
      units["c2"] = _conv_unit("c2", "m1", 4, 4)
      ins = [input_entry("x", width=4, entry_name="c0")]
      if mid == "mul":
        ins.append(input_entry("x2", width=4, channels=4, entry_name="m1"))
      inputs = ins
      out_unit = "c2"
    pl = build_plist(net, out_unit, inputs, [output_entry("y", out_unit)], units)
    pl[net]["Weights"] = ["weights.0", "weights.1"]
    try:
      with tempfile.TemporaryDirectory(prefix="ane_chain_") as d:
        cd = Path(d)
        with (cd / "net.plist").open("wb") as f:
          plistlib.dump(pl, f, sort_keys=True)
        (cd / "weights.0").write_bytes(b"")
        (cd / "weights.1").write_bytes(b"\x00" * 64)
        (cd / "in_x.bin").write_bytes(np.zeros(4 * 4, np.float16).tobytes())
        ins = [("x", cd / "in_x.bin")]
        if "x2" in [e["InputName"] for e in inputs]:
          (cd / "in_x2.bin").write_bytes(np.zeros(4 * 4, np.float16).tobytes())
          ins.append(("x2", cd / "in_x2.bin"))
        info = _nl_invoke(invoker, cd / "net.plist",
                               weights=[cd / "weights.0", cd / "weights.1"],
                               inputs=ins, outputs=[("y", cd / "out_y.bin")], warmup=0)
        print(f"chain {layout}: OK ({info.get('elapsedMs')}ms)")
    except Exception as e:
      msg = str(e)
      i = msg.find("error")
      print(f"chain {layout}: FAIL {msg[max(0,i-20):i+180] if i>=0 else msg[:200]}")


def mode_softmax_bench():
  """Netplist softmax latency at attention-core sizes vs MIL-route 12.5ms/layer figure."""
    # [60,8,256,256] => as (C=60*8, H=256, W=256)? try C=480,H=256,W=256 fp16
  invoker = ensure_invoker("layer_invoker")
  for (c, h, w) in [(480, 256, 256), (60, 256, 256), (8, 256, 256), (1, 256, 256)]:
    n = c * h * w
    if n * 2 > 64 * 1024 * 1024:
      print(f"skip {c}x{h}x{w} (too big)")
      continue
    try:
      with tempfile.TemporaryDirectory(prefix="ane_sm_") as d:
        cd = Path(d)
        write_model("softmax", cd, width=w, height=h, channels=c)
        x = np.random.default_rng(0).standard_normal(n).astype(np.float16)
        (cd / "in_x.bin").write_bytes(x.tobytes())
        info = _nl_invoke(invoker, cd / "net.plist",
                               weights=sorted(cd.glob("weights.*")),
                               inputs=[("x", cd / "in_x.bin")],
                               outputs=[("y", cd / "out_y.bin")],
                               repeats=10, warmup=3)
        ms = info.get("elapsedMs")
        gbps = (n * 2 * 2) / (ms / 1e3) / 1e9 if ms else None
        print(f"softmax C={c:4d} H={h} W={w}: {ms} ms  ({gbps:.0f} GB/s eff)" if ms
              else f"softmax C={c} H={h} W={w}: {info}")
    except Exception as e:
      msg = str(e)[:200]
      print(f"softmax C={c} H={h} W={w}: FAIL {msg}")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", default="ops", choices=["ops", "limits", "sdpa", "chain", "softmax-bench", "all"])
  ap.add_argument("--op", help="scan a single op")
  ap.add_argument("--json", type=Path, default=Path("/Volumes/2T/repos/mss/pymss-ane/.tmp/netlist_scan.json"))
  a = ap.parse_args()
  if a.mode in ("ops", "all"): mode_ops(a.op, a.json)
  if a.mode in ("limits", "all"): mode_limits(a.json.with_suffix(".limits.json"))
  if a.mode in ("sdpa", "all"): mode_sdpa()
  if a.mode in ("chain", "all"): mode_chain()
  if a.mode in ("softmax-bench", "all"): mode_softmax_bench()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
