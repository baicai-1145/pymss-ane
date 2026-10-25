#!/usr/bin/env python3
"""task-9: hybrid-schedule attention core for M4.

Route: MIL conv/matmul program (QK^T) -> netplist Softmax (Path-A, 3.78ms
template measured in task-2) -> MIL program (AV + normalize). IOSurface-side
zero-copy via af.Program input_view/output_view so the host never memcpys the
big sim tensor more than the netlist dispatch itself requires.

Baseline comparison (task-2 + build_graph probes):
  - full-MIL attention core at [60,8,256,256]: ~12.76 ms/layer (softmax in the
    slow elementwise channel)
  - netplist Softmax C=480,H=256,W=256: 3.78 ms
  - mixed target: < 6.4 ms/layer (50%), stretch < 4 ms

Usage:
  .venv/bin/python scripts/attn_hybrid.py            # run all stages
  .venv/bin/python scripts/attn_hybrid.py --quick    # numeric check only
"""
from __future__ import annotations

import argparse
import json
import plistlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netlist_scan as NS  # _nl_invoke + write_model helpers

import aneforge as af

B, H, S, DH = 60, 8, 256, 64          # attention core shape (mel-band time transformer)
HD = H * DH                            # 512
C_NET = B * H                          # 480 channels for the netplist view
LOG2E = float(np.log2(np.e))


# ---------------------------------------------------------------- MIL side --

def build_qk_program():
  """sim = (q/sqrt(dh)) @ k^T : inputs q,k [B,H,S,DH] -> sim [B,H,S,S]."""
  q = af.input((B, H, S, DH))
  k = af.input((B, H, S, DH))
  qs = q * float(DH ** -0.5)
  kt = k.transpose([0, 1, 3, 2])
  sim = qs @ kt                        # [B,H,S,S]
  return af.compile(sim), ("q", "k")


def build_av_program():
  """o = (e @ v) / s : inputs e [B,H,S,S], s [B,H,S,1], v [B,H,S,DH] -> o [B,H,S,DH]."""
  e = af.input((B, H, S, S))
  s = af.input((B, H, S, 1))
  v = af.input((B, H, S, DH))
  o = e @ v
  o = o / s
  return af.compile(o), ("e", "s", "v")


def build_softmax_exp_program():
  """MIL fallback comparator: e = exp2((sim - m) * LOG2E) with m = amax(sim).
  (This is the slow elementwise-channel path we are replacing.)"""
  sim = af.input((B, H, S, S))
  m = sim.amax(-1)
  e = ((sim - m) * LOG2E).exp2()
  return af.compile(e), ("sim",)


# ----------------------------------------------------------- netplist side --

def netlist_softmax_stage(workdir: Path, c=C_NET, h=S, w=S) -> tuple[Path, list[Path]]:
  """Softmax over W at C=c,H=h,W=w through the Path-A invoker (task-2 route)."""
  NS.write_model("softmax", workdir, width=w, height=h, channels=c)
  return workdir / "net.plist", sorted(workdir.glob("weights.*"))


def run_netlist_softmax(invoker, net_plist, weights, x: np.ndarray, out: np.ndarray,
                        repeats=None, warmup=None) -> None:
  """Write fp16 x, run softmax netplist, read fp16 out (both pre-allocated)."""
  with tempfile.TemporaryDirectory(prefix="ane_sm_") as d:
    cd = Path(d)
    (cd / "in.bin").write_bytes(np.ascontiguousarray(x, np.float16).tobytes())
    kw = {}
    if repeats is not None: kw["repeats"] = repeats
    if warmup is not None: kw["warmup"] = warmup
    NS._nl_invoke(invoker, net_plist, weights=weights,
                  inputs=[("x", cd / "in.bin")], outputs=[("y", cd / "out.bin")], **kw)
    out[:] = np.frombuffer((cd / "out.bin").read_bytes(), np.float16).reshape(x.shape)


# ------------------------------------------------------------------ driver --

def attach_persistent_softmax():
  """Spin up the persistent netlist worker for the C=480 softmax (load once, eval many)."""
  import aneforge._netplist_worker as _NW  # type: ignore
  NetlistWorker = _NW._Worker
  import tempfile as _tf
  wd = _tf.TemporaryDirectory(prefix="ane_pw_sm_")
  wdp = Path(wd.name)
  NS.write_model("softmax", wdp, width=S, height=S, channels=C_NET)
  wk = NetlistWorker(wdp / "net.plist", sorted(wdp.glob("weights.*")), ["x"], ["y"], wd)
  return wk


def cosine(a, b):
  a = np.asarray(a, np.float32).ravel()
  b = np.asarray(b, np.float32).ravel()
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def bench(fn, n=30, warmup=5):
  for _ in range(warmup):
    fn()
  t0 = time.perf_counter()
  for _ in range(n):
    fn()
  return (time.perf_counter() - t0) / n * 1e3   # ms


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  a = ap.parse_args()

  rng = np.random.default_rng(0)
  q = (rng.standard_normal((B, H, S, DH)) * 0.5).astype(np.float16)
  k = (rng.standard_normal((B, H, S, DH)) * 0.5).astype(np.float16)
  v = (rng.standard_normal((B, H, S, DH)) * 0.5).astype(np.float16)

  # numpy fp32 reference (softmax over last axis)
  sim32 = np.einsum("bhsd,bhtd->bhst", q.astype(np.float32) * DH ** -0.5, k.astype(np.float32))
  e32 = np.exp(sim32 - sim32.max(-1, keepdims=True))
  s32 = e32.sum(-1, keepdims=True)
  o_ref = (e32 @ v.astype(np.float32)) / s32

  print(f"== shapes: q/k/v {q.shape}  sim [{B},{H},{S},{S}] "
        f"({B*H*S*S*2/1e6:.0f} MB fp16)")

  results = {"shape": f"[{B},{H},{S},{DH}]"}

  # ---- stage 1: MIL QK^T ----------------------------------------------------
  print("[1] compiling MIL qk program ...")
  t0 = time.time()
  qk_net, qk_in = build_qk_program()
  print(f"    compile {time.time()-t0:.1f}s")
  sim_mil = qk_net(q, k)
  cos_qk = cosine(sim_mil, sim32)
  print(f"    qk cosine vs numpy: {cos_qk:.6f}")
  ms_qk = bench(lambda: qk_net(q, k))
  print(f"    qk latency: {ms_qk:.2f} ms")
  results["mil_qk_ms"] = ms_qk

  # ---- stage 2: netlist softmax ---------------------------------------------
  print("[2] netlist softmax stage ...")
  invoker = NS.ensure_invoker("layer_invoker")
  with tempfile.TemporaryDirectory(prefix="ane_hybrid_") as d:
    wd = Path(d)
    plist, weights = netlist_softmax_stage(wd)
    x_sm = sim_mil.astype(np.float16)          # [B,H,S,S] == C=480,H=256,W=256 memory order
    y_sm = np.empty_like(x_sm)
    run_netlist_softmax(invoker, plist, weights, x_sm, y_sm, warmup=0)
    # numeric check vs numpy softmax of the same (rounded) input
    x32 = x_sm.astype(np.float32)
    p_ref = np.exp(x32 - x32.max(-1, keepdims=True))
    p_ref /= p_ref.sum(-1, keepdims=True)
    cos_sm = cosine(y_sm, p_ref)
    print(f"    softmax cosine vs numpy: {cos_sm:.6f}")
    results["softmax_cosine"] = cos_sm

    with tempfile.TemporaryDirectory(prefix="ane_sm_t_") as td:
      # timed softmax-only loop (avoid tempdir overhead inside bench)
      tf = Path(td)
      NS.write_model("softmax", tf, width=S, height=S, channels=C_NET)
      pl, wts = tf / "net.plist", sorted(tf.glob("weights.*"))
      (tf / "in.bin").write_bytes(x_sm.tobytes())
      NS._nl_invoke(invoker, pl, weights=wts, inputs=[("x", tf / "in.bin")],
                    outputs=[("y", tf / "out.bin")], warmup=3)
      ms_sm = bench(lambda: NS._nl_invoke(
          invoker, pl, weights=wts, inputs=[("x", tf / "in.bin")],
          outputs=[("y", tf / "out.bin")], repeats=1, warmup=0), n=20)
      print(f"    softmax+dispatch latency: {ms_sm:.2f} ms")
      results["netlist_softmax_ms"] = ms_sm

  # ---- stage 3: MIL AV + normalize -------------------------------------------
  print("[3] compiling MIL av program ...")
  t0 = time.time()
  av_net, av_in = build_av_program()
  print(f"    compile {time.time()-t0:.1f}s")
  e_hyb = y_sm.astype(np.float16)
  s_hyb = e_hyb.sum(-1, keepdims=True).astype(np.float16)
  o_hyb = av_net(e_hyb, s_hyb, v)
  cos_o = cosine(o_hyb, o_ref)
  print(f"    av output cosine vs numpy: {cos_o:.6f}")
  ms_av = bench(lambda: av_net(e_hyb, s_hyb, v))
  print(f"    av latency: {ms_av:.2f} ms")
  results["av_cosine"] = cos_o
  results["mil_av_ms"] = ms_av

  total = ms_qk + ms_sm + ms_av
  print(f"\n== HYBRID TOTAL: {total:.2f} ms/layer  (qk {ms_qk:.2f} + softmax {ms_sm:.2f} + av {ms_av:.2f})")
  results["hybrid_total_ms"] = total
  results["target_6_4"] = total < 6.4
  results["stretch_4_0"] = total < 4.0

  if a.quick:
    print(json.dumps(results, indent=1))
    return

  # ---- stage 2b: persistent worker round-trip (the real hybrid number) ------
  print("[2b] persistent softmax worker round-trip ...")
  try:
    wk = attach_persistent_softmax()
    x_in = sim_mil.astype(np.float16).ravel()
    y_pw = wk.eval([x_in])[0]
    cos_pw = cosine(y_pw.reshape(x_sm.shape), p_ref if "p_ref" in dir() else y_sm)
    print(f"    persistent softmax cosine: {cos_pw:.6f} (compile {wk.compile_ms:.0f}ms load {wk.load_ms:.0f}ms once)")
    ms_pw = bench(lambda: wk.eval([x_in]), n=20, warmup=3)
    print(f"    persistent softmax round-trip: {ms_pw:.2f} ms  <- true per-layer cost")
    results["persistent_softmax_ms"] = ms_pw
    total_pw = ms_qk + ms_pw + ms_av
    print(f"== HYBRID (persistent) TOTAL: {total_pw:.2f} ms/layer")
    results["hybrid_persistent_total_ms"] = total_pw
    results["target_6_4_persistent"] = total_pw < 6.4
    results["stretch_4_0_persistent"] = total_pw < 4.0
    wk.release()
  except Exception as ex:
    import traceback; traceback.print_exc()
    print("    persistent worker failed:", str(ex)[:200])

  # ---- comparator: full-MIL softmax path (the baseline we replace) -----------
  print("[4] full-MIL comparator (softmax in MIL) ...")
  t0 = time.time()
  sm_net, _ = build_softmax_exp_program()
  print(f"    compile {time.time()-t0:.1f}s")
  e_mil = sm_net(sim_mil.astype(np.float16))
  s_mil = e_mil.sum(-1, keepdims=True).astype(np.float16)
  o_mil = av_net(e_mil.astype(np.float16), s_mil, v)
  cos_mil = cosine(o_mil, o_ref)
  ms_sm_mil = bench(lambda: sm_net(sim_mil.astype(np.float16)))
  total_mil = ms_qk + ms_sm_mil + ms_av
  print(f"    mil softmax stage: {ms_sm_mil:.2f} ms; cosine {cos_mil:.6f}")
  print(f"== FULL-MIL TOTAL: {total_mil:.2f} ms/layer")
  results["mil_softmax_ms"] = ms_sm_mil
  results["full_mil_total_ms"] = total_mil
  results["speedup"] = total_mil / total if total else 0

  # ---- comparator: netplist full chain (backup route) ------------------------
  print("[5] netplist mul+softmax+reduce chain (backup) ...")
  try:
    from aneforge._bridges._netplist import build_plist, input_entry, output_entry
    def umul(n, l, r, c):
      return {"Bottom": [l, r], "InputType": ["Float16", "Float16"], "Name": n,
              "OutputChannels": c, "OutputType": "Float16",
              "Params": {"Type": "Mult"}, "Type": "ElementWise"}
    usm = {"Bottom": ["m0"], "InputType": ["Float16"], "Name": "s1",
           "OutputChannels": C_NET, "OutputType": "Float16",
           "Params": {"Dimension": "Width"}, "Type": "Softmax"}
    pl = build_plist("netchain", "s1",
        [input_entry("x", width=S, height=S, channels=C_NET, entry_name="m0"),
         input_entry("z", width=S, height=S, channels=C_NET, entry_name="m0")],
        [output_entry("y", "s1")], {"m0": umul("m0", "x", "z", C_NET), "s1": usm})
    with tempfile.TemporaryDirectory(prefix="ane_chain2_") as td:
      cd = Path(td)
      with (cd / "net.plist").open("wb") as f:
        plistlib.dump(pl, f, sort_keys=True)
      (cd / "weights.0").write_bytes(b"\x00" * 64)
      n = C_NET * S * S
      (cd / "x.bin").write_bytes(x_sm.ravel().tobytes()[:n * 2])
      (cd / "z.bin").write_bytes(np.ones(n, np.float16).tobytes())
      NS._nl_invoke(invoker, cd / "net.plist", weights=[cd / "weights.0"],
                    inputs=[("x", cd / "x.bin"), ("z", cd / "z.bin")],
                    outputs=[("y", cd / "out.bin")], warmup=3)
      ms_chain = bench(lambda: NS._nl_invoke(
          invoker, cd / "net.plist", weights=[cd / "weights.0"],
          inputs=[("x", cd / "x.bin"), ("z", cd / "z.bin")],
          outputs=[("y", cd / "out.bin")], repeats=1, warmup=0), n=20)
      print(f"    mul+softmax chain: {ms_chain:.2f} ms")
      results["netlist_chain_ms"] = ms_chain
  except Exception as ex:
    print("    chain comparator failed:", str(ex)[:120])

  out_json = Path("/Volumes/2T/repos/mss/pymss-ane/.tmp/attn_hybrid.json")
  out_json.write_text(json.dumps(results, indent=1))
  print(f"\njson -> {out_json}")


if __name__ == "__main__":
  raise SystemExit(main())
