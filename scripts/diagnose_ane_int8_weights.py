"""Benchmark: fp16 vs int8 weight quantization (W8A16) for production projections.

aneforge's matmul emitter natively supports per-channel int8 weights; maderix
measured 1.88x from W8A8 on conv shapes. This probe measures our projection
shapes with compile(int8=True): throughput change AND numeric error vs fp16
output (int8 weights change numerics — gate on relative error, not bitwise).
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


def graph_for(x_shape, w):
    return af.input(x_shape) @ w


def bench(label, x_shape, w, x, int8):
    directory = ROOT / f'.tmp/ane_int8/{label}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(graph_for(x_shape, w), int8=int8, opt=0, build_dir=str(directory))
    for (port, shape), data in zip(net._inputs, [x], strict=True):
        assert tuple(shape) == data.shape
        np.copyto(net.input_view(port), data)
    net.execute()
    out = net.output_view().copy()
    for _ in range(3):
        net.execute()
    return net, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_int8_proj.json')
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    cases = [
        ('qkv_T512', (60, 512, 256), 1536),
        ('out_T512', (60, 512, 512), 256),
        ('ffn_up_T512', (60, 512, 256), 1024),
        ('ffn_dn_T512', (60, 512, 1024), 256),
    ]
    rows = []
    for label, xs, N in cases:
        B, S, K = xs
        w = (rng.normal(0, 0.05, (K, N))).astype(np.float16)
        x = (rng.normal(0, 1.0, (B, S, K))).astype(np.float16)
        nets, outs = {}, {}
        try:
            nets['fp16'], outs['fp16'] = bench(f'{label}_f16', xs, w, x, False)
            nets['int8'], outs['int8'] = bench(f'{label}_i8', xs, w, x, True)
            a, b = outs['fp16'].astype(np.float64), outs['int8'].astype(np.float64)
            denom = np.linalg.norm(a)
            parity = dict(
                cosine=float(a.ravel() @ b.ravel() / (denom * np.linalg.norm(b))),
                relative_l2=float(np.linalg.norm(a - b) / denom),
                max_abs=float(np.abs(a - b).max()),
            )
            samples = []
            for _ in range(2):
                for n in nets.values():
                    n.execute()
            order = random.Random(7)
            for r in range(args.reps):
                keys = list(nets)
                order.shuffle(keys)
                for k in keys:
                    t = time.perf_counter_ns()
                    nets[k].execute()
                    samples.append(dict(round=r, kind=k, ms=(time.perf_counter_ns() - t) / 1e6))
            def stat(v):
                v = sorted(v)
                return dict(min=round(v[0], 4), p50=round(v[len(v)//2], 4), max=round(v[-1], 4))
            summary = {k: stat([s['ms'] for s in samples if s['kind'] == k]) for k in nets}
            flops = 2 * B * S * K * N
            tflops = {k: round(flops / (summary[k]['p50'] * 1e-3) / 1e12, 2) for k in summary}
            speedup = round(summary['fp16']['p50'] / summary['int8']['p50'], 3)
            rows.append(dict(case=label, x=xs, N=N, parity=parity, summary=summary,
                             tflops_p50=tflops, int8_speedup=speedup))
            print(f'{label}: fp16 {summary["fp16"]["p50"]}ms ({tflops["fp16"]}T) | '
                  f'int8 {summary["int8"]["p50"]}ms ({tflops["int8"]}T) | speedup {speedup}x | '
                  f'rel_l2={parity["relative_l2"]:.4f}', flush=True)
        finally:
            for n in nets.values():
                n.release()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
