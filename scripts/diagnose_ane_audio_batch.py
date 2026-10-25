"""Audio-batch counterfactual: fold N audio clips into the band-batch dim (60 -> 60N).

The model's attention batch is 60 frequency bands (not audio batch). Processing N
clips per execute = batch 60N: flops xN, bytes xN, per-op setup unchanged -> the
cost-model setup tax amortizes. This probe measures attention core + QKV projection
at B=60/120/240/480 to quantify, and checks the score-matrix memory wall
(T512 x B480 score = 1.9GB -- may exceed limits; T256 fits).
Gates: cos>0.9999 vs fp32 ref; then per-clip latency (total/N) trend.
"""
from __future__ import annotations
import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import aneforge as af

ROOT = Path(__file__).resolve().parents[1]


def attention_core(B, S, H=8, D=64):
    q = af.input((B, H, S, D))
    kt = af.input((B, H, D, S))
    v = af.input((B, H, S, D))
    p = (q @ kt * (D ** -0.5)).softmax(-1)
    return q, kt, v, p @ v


def qkv_proj(B, S, K=256, N=1536):
    return af.input((B, S, K)) @ np.zeros  # placeholder replaced below


def bench_core(B, S, reps):
    rng = np.random.default_rng(71)
    q, kt, v, o = attention_core(B, S)
    d = ROOT / f'.tmp/ane_batch/core_B{B}_S{S}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(o, opt=0, build_dir=str(d))
    q_np = rng.normal(0, 0.5, (B, 8, S, 64)).astype(np.float16)
    k_np = rng.normal(0, 0.5, (B, 8, 64, S)).astype(np.float16)
    v_np = rng.normal(0, 0.5, (B, 8, S, 64)).astype(np.float16)
    for (p, sh), x in zip(net._inputs, [q_np, k_np, v_np], strict=True):
        np.copyto(net.input_view(p), x)
    net.execute()
    out = net.output_view().copy()
    assert np.isfinite(out.astype(np.float32)).all()
    sc = np.einsum('bhud,bhdk->bhuk', q_np.astype(np.float32), k_np.astype(np.float32)) * 64 ** -0.5
    sc -= sc.max(-1, keepdims=True)
    pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
    ref = np.einsum('bhuk,bhkd->bhud', pr, v_np.astype(np.float32))
    cos = float(out.astype(np.float64).ravel() @ ref.ravel() / (np.linalg.norm(out) * np.linalg.norm(ref)))
    assert cos > 0.9999, (B, S, cos)
    for _ in range(3):
        net.execute()
    lat = []
    for r in range(reps):
        t = time.perf_counter_ns()
        net.execute()
        lat.append((time.perf_counter_ns() - t) / 1e6)
    lat.sort()
    net.release()
    med = lat[len(lat) // 2]
    return dict(op='attn_core', B=B, S=S, median_ms=round(med, 3),
                per_clip_ms=round(med / (B / 60), 3), cos=round(cos, 6))


def bench_proj(B, S, reps):
    rng = np.random.default_rng(73)
    w = rng.normal(0, 0.05, (256, 1536)).astype(np.float16)
    x = rng.normal(0, 1.0, (B, S, 256)).astype(np.float16)
    d = ROOT / f'.tmp/ane_batch/proj_B{B}_S{S}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(af.input((B, S, 256)) @ w, opt=0, build_dir=str(d))
    np.copyto(net.input_view(net._inputs[0][0]), x)
    net.execute()
    assert np.isfinite(net.output_view().copy().astype(np.float32)).all()
    for _ in range(3):
        net.execute()
    lat = []
    for r in range(reps):
        t = time.perf_counter_ns()
        net.execute()
        lat.append((time.perf_counter_ns() - t) / 1e6)
    lat.sort()
    net.release()
    med = lat[len(lat) // 2]
    return dict(op='qkv_proj', B=B, S=S, median_ms=round(med, 3),
                per_clip_ms=round(med / (B / 60), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_batch.json')
    args = ap.parse_args()
    rows = []
    for B in (60, 120, 240, 480):
        rows.append(bench_core(B, 256, args.reps))
        print(rows[-1], flush=True)
    for B in (60, 240):
        rows.append(bench_core(B, 512, args.reps))
        print(rows[-1], flush=True)
    for B in (60, 240):
        rows.append(bench_proj(B, 512, args.reps))
        print(rows[-1], flush=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    base = next(r for r in rows if r['op'] == 'attn_core' and r['B'] == 60 and r['S'] == 256)
    for r in rows:
        if r['op'] == 'attn_core' and r['S'] == 256:
            print(f"B{r['B']:>3}: {r['median_ms']:7.2f}ms total, {r['per_clip_ms']:6.3f}ms/clip "
                  f"({base['per_clip_ms']/r['per_clip_ms']:.2f}x throughput vs B60)", flush=True)


if __name__ == '__main__':
    main()
