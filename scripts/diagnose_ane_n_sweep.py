"""Pure N-sweep at K=512 (no slicing/padding tricks): map the narrow-N cliff.

Data so far: N=256 ~6-7T, N=1536 ~14.6T at K=512. Pad-to-512+slice was slower
(wasted flops). Missing: where exactly between 256 and 1024 does throughput
saturate? If the curve is a step at 384 (MAC tile width), a REAL architectural
fix is fusing the two N=256 producers in the model graph; if it's gradual,
narrow-N is bandwidth-bound and unfixable by shape tricks.
Also: N=768 = out_proj-of-two-heads stacked (batching two N=256 ops as one N=512).
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_n_sweep.json')
    args = ap.parse_args()
    rng = np.random.default_rng(23)
    B, S, K = 60, 512, 512
    rows = []
    nets = {}
    try:
        for N in (256, 320, 384, 448, 512, 768, 1024, 1536):
            w = (rng.normal(0, 0.05, (K, N))).astype(np.float16)
            x = (rng.normal(0, 1.0, (B, S, K))).astype(np.float16)
            d = ROOT / f'.tmp/ane_n_sweep/N{N}'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                net = af.compile(af.input((B, S, K)) @ w, opt=0, build_dir=str(d))
            np.copyto(net.input_view(net._inputs[0][0]), x)
            net.execute()
            out = net.output_view().copy()
            assert np.isfinite(out.astype(np.float32)).all()
            nets[N] = (net, x)
        for _ in range(3):
            for net, _ in nets.values():
                net.execute()
        samples = []
        order = random.Random(29)
        for r in range(args.reps):
            keys = list(nets)
            order.shuffle(keys)
            for k in keys:
                t = time.perf_counter_ns()
                nets[k][0].execute()
                samples.append(dict(round=r, N=k, ms=(time.perf_counter_ns() - t) / 1e6))
        def stat(v):
            v = sorted(v)
            return dict(min=round(v[0], 4), p50=round(v[len(v)//2], 4), max=round(v[-1], 4))
        for N in sorted(nets):
            s = stat([x['ms'] for x in samples if x['N'] == N])
            flops = 2 * B * S * K * N
            tf = round(flops / (s['p50'] * 1e-3) / 1e12, 2)
            rows.append(dict(N=N, **s, tflops=tf))
            print(f'N={N:5}: p50={s["p50"]}ms → {tf}T', flush=True)
    finally:
        for net, _ in nets.values():
            net.release()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(B=B, S=S, K=K, rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
