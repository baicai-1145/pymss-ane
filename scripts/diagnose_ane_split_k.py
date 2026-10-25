"""Probe: split-K matmul (K chunks of 256, partial sums) vs monolithic K>=512.

The throughput cliff is on the reduction dim: K=256 runs 11-12.7T, K=512/1024
runs ~6T. Split-K trades one wide matmul for ceil(K/256) narrow ones plus an
fp16 add chain. Gates: cosine>0.9999 and rel_l2<5e-3 vs the monolithic output
(split changes accumulation order; bitwise is NOT expected), then latency.
Also sweeps K at fixed N to map the cliff, and K=256/N=256 to test interaction.
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


def compile_case(label, B, S, K, N, split):
    rng = np.random.default_rng(97)
    w = (rng.normal(0, 0.05, (K, N))).astype(np.float16)
    x = (rng.normal(0, 1.0, (B, S, K))).astype(np.float16)
    xin = af.input((B, S, K))
    if split is None:
        g = xin @ w
        tag = 'mono'
    else:
        parts = []
        for i in range(0, K, split):
            k2 = min(split, K - i)
            xi = xin.slice_by_size([0, 0, i], [B, S, k2])
            parts.append(xi @ w[i:i + k2])
        g = parts[0]
        for p in parts[1:]:
            g = g + p
        tag = f'split{split}'
    d = ROOT / f'.tmp/ane_split_k/{label}_{tag}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(g, opt=0, build_dir=str(d))
    for (port, shape), data in zip(net._inputs, [x], strict=True):
        assert tuple(shape) == data.shape
        np.copyto(net.input_view(port), data)
    net.execute()
    out = net.output_view().copy()
    for _ in range(3):
        net.execute()
    return net, out, x, w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_split_k.json')
    args = ap.parse_args()
    rows = []
    # K sweep at fixed N=1536 (map the cliff)
    for K in (256, 384, 512, 768, 1024):
        rows.append(case(f'ksweep_K{K}', 60, 512, K, 1536, [None, 256], args.reps))
    # production splits
    rows.append(case('outproj_T512', 60, 512, 512, 256, [None, 256], args.reps))
    rows.append(case('ffndn_T512', 60, 512, 1024, 256, [None, 256], args.reps))
    # interaction: K=256 N=256
    rows.append(case('k256_n256', 60, 512, 256, 256, [None], args.reps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


def case(label, B, S, K, N, variants, reps):
    nets, outs = {}, {}
    try:
        for split in variants:
            nets['mono' if split is None else f'split{split}'], outs[('mono' if split is None else f'split{split}')], _, _ = \
                None, None, None, None  # placeholder replaced below
        nets, outs = {}, {}
        for split in variants:
            tag = 'mono' if split is None else f'split{split}'
            net, out, _, _ = compile_case(label, B, S, K, N, split)
            nets[tag], outs[tag] = net, out
        ref = outs['mono'].astype(np.float64)
        parity = {}
        for tag in outs:
            if tag == 'mono':
                continue
            o = outs[tag].astype(np.float64)
            parity[tag] = dict(
                cosine=float(o.ravel() @ ref.ravel() / (np.linalg.norm(o) * np.linalg.norm(ref))),
                relative_l2=float(np.linalg.norm(o - ref) / np.linalg.norm(ref)))
            assert parity[tag]['cosine'] > 0.9999 and parity[tag]['relative_l2'] < 5e-3, (label, tag, parity[tag])
        for _ in range(2):
            for n in nets.values():
                n.execute()
        samples = []
        order = random.Random(13)
        for r in range(reps):
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
        tf = {k: round(flops / (summary[k]['p50'] * 1e-3) / 1e12, 2) for k in summary}
        line = ' | '.join(f'{k} {summary[k]["p50"]}ms ({tf[k]}T)' for k in summary)
        print(f'{label}: {line} | parity={parity}', flush=True)
        return dict(case=label, B=B, S=S, K=K, N=N, summary=summary, tflops=tf, parity=parity)
    finally:
        for n in nets.values():
            if n is not None:
                n.release()


if __name__ == '__main__':
    main()
