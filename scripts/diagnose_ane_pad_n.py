"""Probe: does padding the output-channel dim N recover throughput for narrow-N matmuls?

out_proj/ffn_dn (N=256) run at ~6T while N>=1024 shapes reach 11-12.7T. Test:
zero-pad W to N=512/1024, matmul, slice the real 256 channels out. If the narrow-N
penalty is tile fill (not bandwidth), padded-N should run at near-wide-N throughput
and the slice may still win despite 2-4x flops. Gates: exact-match on real channels
(weights are zero-padded, sliced output must be bitwise-equal to unpadded run),
then latency.
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


def run(label, B, S, K, N, reps):
    rng = np.random.default_rng(hash(label) % 2**31)
    w = (rng.normal(0, 0.05, (K, N))).astype(np.float16)
    x = (rng.normal(0, 1.0, (B, S, K))).astype(np.float16)
    nets, outs, slices = {}, {}, {}
    try:
        # baseline N
        d = ROOT / f'.tmp/ane_pad_n/{label}_n{N}'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nets['n256'] = af.compile(af.input((B, S, K)) @ w, opt=0, build_dir=str(d))
        np.copyto(nets['n256'].input_view(nets['n256']._inputs[0][0]), x)
        nets['n256'].execute()
        outs['n256'] = nets['n256'].output_view().copy()
        # padded variants
        for NP in (512, 1024):
            if NP <= N:
                continue
            wp = np.zeros((K, NP), np.float16)
            wp[:, :N] = w
            xin = af.input((B, S, K))
            y = xin @ wp
            ys = y.slice_by_size([0, 0, 0], [B, S, N])  # take real channels
            d = ROOT / f'.tmp/ane_pad_n/{label}_n{NP}'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nets[f'n{NP}'] = af.compile(ys, opt=0, build_dir=str(d))
            np.copyto(nets[f'n{NP}'].input_view(nets[f'n{NP}']._inputs[0][0]), x)
            nets[f'n{NP}'].execute()
            outs[f'n{NP}'] = nets[f'n{NP}'].output_view().copy()
            assert np.array_equal(outs[f'n{NP}'], outs['n256']), f'{label} n{NP} slice mismatch'
        for _ in range(3):
            for n in nets.values():
                n.execute()
        samples = []
        order = random.Random(11)
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
        flops_real = 2 * B * S * K * N
        eff = {k: round(flops_real / (summary[k]['p50'] * 1e-3) / 1e12, 2) for k in summary}
        print(f'{label}: ' + ' | '.join(f'{k} {summary[k]["p50"]}ms ({eff[k]}T-eff)' for k in summary), flush=True)
        return dict(case=label, B=B, S=S, K=K, N=N, summary=summary, effective_tflops=eff,
                    bitwise_slice_match=True)
    finally:
        for n in nets.values():
            n.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_pad_n.json')
    args = ap.parse_args()
    rows = [
        run('outproj_T512', 60, 512, 512, 256, args.reps),
        run('ffndn_T512', 60, 512, 1024, 256, args.reps),
        run('outproj_T256', 60, 256, 512, 256, args.reps),
    ]
    args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
