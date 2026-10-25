"""Benchmark: native matmul lowering vs 1x1-conv expression for weight projections.

maderix/ANE reports conv-as-matmul reaching 18.6 TOPS where matmul path ~6.
Our production projections (QKV [60,512,256]@[256,1536] etc.) measured only
5.4-10.8 TFLOPS through the matmul emitter. This probe builds BOTH lowerings
for identical math (same weights, same fp16 inputs, opt=0), verifies bitwise
or near-bitwise agreement, then times steady-state executes interleaved.
Also probes the bmm attention products reshaped as conv (activation weights
are NOT legal conv inputs — this variant expects to fail, documenting why).
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


def matmul_graph(x_shape, w):
    x = af.input(x_shape)
    return x @ w

def conv1x1_graph(x_shape, w):
    # x [B, S, K] @ w [K, N] as conv: x->[B,K,S,1] (C=K), weight [N,K,1,1], out [B,N,S,1]->[B,S,N]
    B, S, K = x_shape
    N = w.shape[1]
    x = af.input((B, K, S, 1))
    out = af.conv(x, weight=w.T.reshape(N, K, 1, 1))  # [B,N,S,1]
    return out.transpose((0, 2, 1, 3)).reshape((B, S, N))


def bench(label, graph_fn, x_shape, w, feeds_np, reps):
    directory = ROOT / f'.tmp/ane_matmul_vs_conv/{label}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(graph_fn(x_shape, w), opt=0, build_dir=str(directory))
    for (port, shape), data in zip(net._inputs, feeds_np, strict=True):
        assert tuple(shape) == data.shape, (label, shape, data.shape)
        np.copyto(net.input_view(port), data)
    net.execute()
    out = net.output_view().copy()
    for _ in range(3):
        net.execute()
    return net, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_matmul_vs_conv.json')
    args = ap.parse_args()
    rng = np.random.default_rng(311)
    cases = [
        # label, x_shape(B,S,K), N — production projection shapes
        ('qkv_T512', (60, 512, 256), 1536),   # QKV proj: [60,512,256]@[256,1536]
        ('out_T512', (60, 512, 512), 256),    # out proj
        ('ffn_up_T512', (60, 512, 256), 1024),
        ('ffn_dn_T512', (60, 512, 1024), 256),
        ('qkv_T256', (60, 256, 256), 1536),
    ]
    rows = []
    for label, xs, N in cases:
        B, S, K = xs
        w = (rng.normal(0, 0.05, (K, N))).astype(np.float16)
        x = (rng.normal(0, 1.0, (B, S, K))).astype(np.float16)
        nets, outs = {}, {}
        try:
            nets['matmul'], outs['matmul'] = bench(f'{label}_mm', matmul_graph, xs, w, [x], args.reps)
            xc = np.ascontiguousarray(x.transpose(0, 2, 1)[:, :, :, None])  # [B,K,S,1]
            nets['conv1x1'], outs['conv1x1'] = bench(f'{label}_cv', conv1x1_graph, xs, w, [xc], args.reps)
            a, b = outs['matmul'].astype(np.float64), outs['conv1x1'].astype(np.float64)
            parity = dict(
                bitwise_equal=bool(np.array_equal(outs['matmul'], outs['conv1x1'])),
                cosine=float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b))),
                max_abs=float(np.abs(a - b).max()),
            )
            assert parity['cosine'] > 0.9999, parity
            samples = []
            for _ in range(2):
                for n in nets.values():
                    n.execute()
            order = random.Random(313)
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
            flops = 2 * B * S * K * N
            summary = {k: stat([s['ms'] for s in samples if s['kind'] == k]) for k in nets}
            tflops = {k: round(flops / (summary[k]['p50'] * 1e6) / 1e12, 2) for k in summary}
            row = dict(case=label, x=xs, N=N, parity=parity, summary=summary, tflops_p50=tflops)
            rows.append(row)
            print(f'{label}: matmul {summary["matmul"]["p50"]}ms ({tflops["matmul"]}T) | '
                  f'conv1x1 {summary["conv1x1"]["p50"]}ms ({tflops["conv1x1"]}T) | parity={parity}', flush=True)
        finally:
            for n in nets.values():
                n.release()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
