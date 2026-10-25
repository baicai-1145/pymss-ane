"""Controlled reproduction of ANE/MLX alternation slowdown, not a hardware diagnosis.

Run alone: .venv/bin/python scripts/diagnose_ane_gpu.py --T 512 --rounds 8
Writes raw per-boundary perf_counter_ns samples and an executable symptom check.
Exit 2 means median GPU-after-ANE / GPU-after-GPU >= --threshold.
All GPU inputs are materialized before timing; no baseline-subtraction attribution.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time
import warnings

import numpy as np
import mlx.core as mx
import aneforge as af
import hybrid_probe as hp

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--threshold', type=float, default=1.5)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_gpu_repro.json')
    args = ap.parse_args()
    cfg, w, dims = hp.load_all()
    cfg = dict(cfg)
    cfg['depth'] = 1
    print('Compiling real depth-0 PM2 (same graph as hybrid A-prime)', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(af.reduce_sum_to_matmul(hp.build_pm2(args.T, cfg, w, dims, 0)))
    h, dh, dim = cfg['heads'], cfg['dim_head'], cfg['dim']
    rng = np.random.default_rng(127)
    o = rng.normal(0, 0.1, (60, args.T, h * dh)).astype(np.float16)
    x = rng.normal(0, 0.1, (60, args.T, dim)).astype(np.float16)
    qs = [mx.array(rng.normal(0, 0.1, (60, h, args.T, dh)).astype(np.float16)) for _ in range(3)]
    mx.eval(*qs)
    def ane():
        return net(o, x)
    def gpu():
        y = mx.fast.scaled_dot_product_attention(*qs, scale=dh ** -0.5)
        mx.eval(y)
        return y
    def timed(fn):
        t = time.perf_counter_ns()
        result = fn()
        return (time.perf_counter_ns() - t) / 1e6, result
    # Warm both paths without timing lazy random generation or first compilation.
    for _ in range(5):
        ane()
        gpu()
    cases = {'gpu_after_gpu': (gpu, gpu), 'gpu_after_ane': (ane, gpu),
             'ane_after_ane': (ane, ane), 'ane_after_gpu': (gpu, ane)}
    rows = []
    order_rng = random.Random(19)
    for r in range(args.rounds):
        order = list(cases)
        order_rng.shuffle(order)
        for name in order:
            before, target = cases[name]
            before_ms, first = timed(before)
            target_ms, second = timed(target)
            rows.append(dict(round=r, case=name, before_ms=before_ms, target_ms=target_ms))
            del first, second
    summary = {}
    for name in cases:
        ts = [row['target_ms'] for row in rows if row['case'] == name]
        summary[name] = dict(min_ms=min(ts), median_ms=statistics.median(ts), max_ms=max(ts))
    ratio = summary['gpu_after_ane']['median_ms'] / summary['gpu_after_gpu']['median_ms']
    result = dict(T=args.T, heads=h, dh=dh, dim=dim, rounds=args.rounds,
                  pid=os.getpid(), dtype='fp16', reduce_fix=True,
                  summary=summary, gpu_slowdown_ratio=ratio, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for name, val in summary.items():
        print(f'{name}: min={val["min_ms"]:.3f} median={val["median_ms"]:.3f} max={val["max_ms"]:.3f}ms')
    print(f'GPU alternation ratio={ratio:.3f}; threshold={args.threshold}; symptom={ratio >= args.threshold}')
    print(f'Raw samples: {args.output}')
    net.release()
    raise SystemExit(2 if ratio >= args.threshold else 0)


if __name__ == '__main__':
    main()
