"""D=128 head-remap probe: same math, same weights, attention re-shaped for ANE bmm.

Model: dim=256, heads=8, dim_head=64 (hd=512). Earlier bmm sweep: inner dim D=64 runs
1.32T, D=128 runs 1.87T (1.6x). ANE ops don't care about "true" head structure — the
attention math softmax(q·k^T·s)·v is invariant if we REGROUP the 512 channels as
H'=4 pseudo-heads of D'=128: any [512] split into 4x[128] blocks gives an equivalent
computation (each block attends independently — same as 8 heads of 64, since attention
is per-head independent and channel blocks never mix).

This differs from a MODEL change (which would need retraining): it is a pure graph
re-shaping of the SAME trained weights. Weights are loaded as [hd, x] and consumed
after a reshape [8,64,...]→[4,128,...] — but heads never interact in attention, so
concatenating adjacent head pairs into 128-wide pseudo-heads computes the identical
function. The rope cos/sin tables stay per-64-pair (interleaved pair layout is inside
each 64-block, untouched).

Actually CAUTION: rope rotation pairs adjacent dims within each head's 64 channels.
Merging heads h,h+1 into one 128 block keeps those pairs intact (channel layout
unchanged, just regrouping which channels belong to the same attention row-group).
softmax then runs over the SAME per-head scores — no, wait: with pseudo-heads the
scores are computed per 128-block = per ORIGINAL head pair? NO — attention output
for row s is sum over its OWN head's keys. A pseudo-head of 128 channels would mix
two original heads' q with both heads' k — WRONG unless we keep q·k within blocks.

So this probe is only valid if we reshape [H=8,S,64] -> [H=4, S*2? ...] — the only
sound remap: fold HEADS INTO S (each head becomes extra rows):
  q'[B, H=4, S*2, 128]? That computes DIFFERENT score patterns (queries of head h
  attending to keys of head h and h+1) — mathematically different.

CONCLUSION (documented in-code): D=128 remap of an 8x64 model is NOT function-preserving.
The 1.6x bmm gain requires the MODEL to be trained with dim_head=128. This probe
measures the counterfactual: same-model-shape graph with dim=512, heads=4 — to
quantify what a retrained D=128 model would gain on ANE. Numbers gate the
retrain-or-not decision.
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
LOG2E = float(np.log2(np.e))


def attention_core(B, S, H, D, native=True):
    q = af.input((B, H, S, D))
    kt = af.input((B, H, D, S))
    v = af.input((B, H, S, D))
    sim = q @ kt * (D ** -0.5)
    if native:
        p = sim.softmax(-1)
    else:
        m = sim.amax(-1)
        e = ((sim - m) * LOG2E).exp2()
        s = e @ np.ones((S, 1), np.float16)
        p = e / s if False else None
    o = p @ v
    return q, kt, v, o


def bench(label, B, S, H, D, reps):
    rng = np.random.default_rng(61)
    q, kt, v, o = attention_core(B, S, H, D)
    d = ROOT / f'.tmp/ane_d128/{label}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(o, opt=0, build_dir=str(d))
    q_np = rng.normal(0, 0.5, (B, H, S, D)).astype(np.float16)
    k_np = rng.normal(0, 0.5, (B, H, D, S)).astype(np.float16)
    v_np = rng.normal(0, 0.5, (B, H, S, D)).astype(np.float16)
    for (p, sh), x in zip(net._inputs, [q_np, k_np, v_np], strict=True):
        assert tuple(sh) == x.shape
        np.copyto(net.input_view(p), x)
    net.execute()
    out = net.output_view().copy()
    assert np.isfinite(out.astype(np.float32)).all()
    # fp32 reference
    sc = np.einsum('bhud,bhdk->bhuk', q_np.astype(np.float32), k_np.astype(np.float32)) * D ** -0.5
    sc -= sc.max(-1, keepdims=True)
    pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
    ref = np.einsum('bhuk,bhkd->bhud', pr, v_np.astype(np.float32))
    cos = float(out.astype(np.float64).ravel() @ ref.ravel() / (np.linalg.norm(out) * np.linalg.norm(ref)))
    assert cos > 0.9999, (label, cos)
    for _ in range(3):
        net.execute()
    lat = []
    order = random.Random(67)
    for r in range(reps):
        t = time.perf_counter_ns()
        net.execute()
        lat.append((time.perf_counter_ns() - t) / 1e6)
    lat.sort()
    net.release()
    return dict(label=label, H=H, D=D, median_ms=round(lat[len(lat)//2], 3), min_ms=round(lat[0], 3), cos=round(cos, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_d128.json')
    args = ap.parse_args()
    B, S = 60, 256
    rows = []
    # production shape (counterfactual base): H=8 D=64
    rows.append(bench('prod_H8_D64', B, S, 8, 64, args.reps))
    # retrained-model counterfactuals: hd=512 kept, wider heads
    rows.append(bench('cf_H4_D128', B, S, 4, 128, args.reps))
    # dim=512 model counterfactual (hd doubled? keep 512: H=4 D=128 is the apples-to-apples)
    # larger dim: hd=1024
    rows.append(bench('cf_H8_D128_hd1024', B, S, 8, 128, args.reps))
    # also T=512 variants for the long-context direction
    for H, D, tag in [(8, 64, 'prod'), (4, 128, 'cf')]:
        rows.append(bench(f'{tag}_T512_H{H}_D{D}', 60, 512, H, D, args.reps))
    for r in rows:
        print(r, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    base = next(r for r in rows if r['label'] == 'prod_H8_D64')
    cf = next(r for r in rows if r['label'] == 'cf_H4_D128')
    print(f"\nD=64→128 (H8→4, same hd=512): {base['median_ms']} → {cf['median_ms']} ms = {base['median_ms']/cf['median_ms']:.2f}x", flush=True)


if __name__ == '__main__':
    main()
