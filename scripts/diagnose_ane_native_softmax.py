"""Native MIL softmax op vs our manual exp2 decomposition on production attention cores.

Our build_graph deliberately uses the manual chain (amax → sub·log2e → exp2 → @ones →
div) citing task-21's finding that Espresso's transform_softmax decomposes MIL softmax.
But maderix's TRAINING pipeline uses the native softmax(axis=-1) op successfully, and
we never A/B'd the two lowerings for SPEED on our shapes. If the compiler's own
decomposition schedules better than our hand chain (fewer graph nodes → fewer setups),
this is a direct production win. B=60 heads=8 production shapes, opt=0, interleaved.
Gates: cosine>0.9999, rel_l2<3e-3 vs numpy fp32; both variants same math.
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
from diagnose_ane_utilization import memory_counters, stat

ROOT = Path(__file__).resolve().parents[1]
LOG2E = float(np.log2(np.e))


def core_manual(q, kt, v, S):
    score = (q @ kt) * (q.shape[-1] ** -0.5)
    m = score.amax(-1)
    e = ((score - m) * LOG2E).exp2()
    s = e @ np.ones((S, 1), np.float16)
    o = e @ v
    return o / s


def core_native(q, kt, v, S):
    score = (q @ kt) * (q.shape[-1] ** -0.5)
    p = score.softmax(-1)
    return p @ v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_native_softmax.json')
    args = ap.parse_args()
    rng = np.random.default_rng(411)
    rows = []
    for direction, B, S in [('time', 60, 256), ('freq', 256, 60)]:
        H, D = 8, 64
        q_np = rng.normal(0, 0.5, (B, H, S, D)).astype(np.float16)
        kt_np = rng.normal(0, 0.5, (B, H, D, S)).astype(np.float16)
        v_np = rng.normal(0, 0.5, (B, H, S, D)).astype(np.float16)
        nets, outs = {}, {}
        try:
            for label, fn in [('manual', core_manual), ('native', core_native)]:
                q = af.input((B, H, S, D)); kt = af.input((B, H, D, S)); v = af.input((B, H, S, D))
                g = fn(q, kt, v, S)
                d = ROOT / f'.tmp/ane_native_softmax/{direction}_{label}'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    net = af.compile(g, opt=0, build_dir=str(d))
                for (p, sh), x in zip(net._inputs, [q_np, kt_np, v_np], strict=True):
                    np.copyto(net.input_view(p), x)
                net.execute()
                outs[label] = net.output_view().copy()
                assert np.isfinite(outs[label].astype(np.float32)).all()
                nets[label] = net
            # numpy fp32 reference
            sc = np.einsum('bhud,bhdk->bhuk', q_np.astype(np.float32), kt_np.astype(np.float32)) * D ** -0.5
            sc -= sc.max(-1, keepdims=True)
            p = np.exp(sc); p /= p.sum(-1, keepdims=True)
            ref = np.einsum('bhuk,bhkd->bhud', p, v_np.astype(np.float32))
            parity = {}
            for label in outs:
                o = outs[label].astype(np.float64)
                parity[label] = dict(
                    cosine=float(o.ravel() @ ref.ravel() / (np.linalg.norm(o) * np.linalg.norm(ref))),
                    relative_l2=float(np.linalg.norm(o - ref) / np.linalg.norm(ref)))
                assert parity[label]['cosine'] > 0.9999 and parity[label]['relative_l2'] < 3e-3, (label, parity[label])
            for _ in range(3):
                for n in nets.values():
                    n.execute()
            samples = []
            order = random.Random(419)
            for r in range(args.reps):
                keys = list(nets)
                order.shuffle(keys)
                for k in keys:
                    t = time.perf_counter_ns()
                    nets[k].execute()
                    samples.append(dict(round=r, kind=k, ms=(time.perf_counter_ns() - t) / 1e6))
            summary = {k: stat([s['ms'] for s in samples if s['kind'] == k]) for k in nets}
            speedup = summary['manual']['median_ms'] / summary['native']['median_ms']
            rows.append(dict(direction=direction, B=B, S=S, summary=summary, parity=parity,
                             native_speedup=round(speedup, 3)))
            print(f'{direction}: manual {summary["manual"]["median_ms"]:.3f}ms | native {summary["native"]["median_ms"]:.3f}ms | native speedup {speedup:.3f}x', flush=True)
            print(f'{direction}: parity={parity}', flush=True)
        finally:
            for n in nets.values():
                n.release()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
