"""Matched full-depth comparison: legacy hybrid / fp16-view hybrid / full ANE.
No cross-run baselines, no repeated-depth weights, no claimed hardware mechanism.
Use --T 64 --depth 2 --reference for the quick correctness gate, then T512.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import time
import warnings

import numpy as np
import aneforge as af
import hybrid_probe as hp

ROOT = Path(__file__).resolve().parents[1]


class ViewHybrid:
    """Keep ANE ports fp16; MLX import still copies and is included in timing.

    Returned output is borrowed storage, overwritten by the next forward().
    A separate program is retained for every depth's actual weights.
    """
    def __init__(self, nets, core, cfg, dims, T):
        self.nets, self.core, self.cfg, self.dims, self.T = nets, core, cfg, dims, T
        self.inputs = {key: [net.input_view(name) for name, _ in net._inputs]
                       for key, net in nets.items()}
        self.outputs = {key: net.output_view() for key, net in nets.items()}

    def forward(self, xin):
        if self.T >= 385:
            feeds, off = [], 0
            for start, end, dim in hp.bg.dim_groups(self.dims):
                width = (end - start) * dim
                feeds.append(xin[:, :, off:off + width].reshape(1, self.T, end - start, dim))
                off += width
        else:
            feeds = [xin]
        for dst, src in zip(self.inputs['p1'], feeds, strict=True):
            np.copyto(dst, src)
        self.nets['p1'].execute()
        state = self.outputs['p1']
        for d in range(self.cfg['depth']):
            pa, pm = f'pa{d}', f'pm2_{d}'
            np.copyto(self.inputs[pa][0], state)
            self.nets[pa].execute()
            attended = self.core.run_bridge(self.outputs[pa])
            np.copyto(self.inputs[pm][0], attended)
            np.copyto(self.inputs[pm][1], state[0].transpose(1, 0, 2))
            self.nets[pm].execute()
            state = self.outputs[pm]
        np.copyto(self.inputs['p2'][0], state)
        self.nets['p2'].execute()
        return self.outputs['p2']


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pack_reference(ref, dims, T, stems):
    ref = ref.reshape(stems, T, -1)
    offsets = np.cumsum([0, *dims])
    pieces = []
    for start, end, width in hp.bg.dim_groups(dims * stems):
        for row in range(start, end):
            stem, band = divmod(row, len(dims))
            pieces.append(ref[stem, :, offsets[band]:offsets[band] + width])
    return np.concatenate(pieces, axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--reduce-fix', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/hybrid_e2e_audit.json')
    args = ap.parse_args()
    cfg, w, dims = hp.load_all()
    cfg = dict(cfg)
    cfg['depth'] = args.depth
    print(f'T={args.T} depth={args.depth} reduce_fix={args.reduce_fix}', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nets = hp.compile_a2(args.T, cfg, w, dims, reduce_fix=args.reduce_fix)
        full_graph = hp.bg.build(args.T, cfg, w, dims)
        full = af.compile(af.reduce_sum_to_matmul(full_graph) if args.reduce_fix else full_graph)
    core = hp.MLXSDPACore(w, cfg, args.T)
    view_hybrid = ViewHybrid(nets, core, cfg, dims, args.T)
    rng = np.random.default_rng(42)
    xin = rng.normal(0, 0.5, (1, args.T, sum(dims))).astype(np.float16)
    fns = dict(legacy=lambda: hp.hybrid_forward_a2(nets, core, xin, cfg, dims, args.T),
               views=lambda: view_hybrid.forward(xin),
               full_ane=lambda: hp.feed_p1(full, xin, args.T, dims))
    outputs = {key: fn().astype(np.float16).copy() for key, fn in fns.items()}
    bitwise = bool(np.array_equal(outputs['legacy'], outputs['views']))
    # This is the regression gate for removing public-call conversion/copy chains.
    np.testing.assert_array_equal(outputs['legacy'], outputs['views'])
    parity = dict(legacy_views_bitwise=bitwise,
                  legacy_views_cos=cosine(outputs['legacy'], outputs['views']),
                  hybrid_full_ane_cos=cosine(outputs['views'], outputs['full_ane']))
    if args.reference:
        import parity as par
        w64 = {k: v.astype(np.float64) for k, v in w.items() if v.dtype == np.float16}
        ref = par.ref_forward(xin.astype(np.float64), w64, cfg, dims)
        packed = pack_reference(ref, dims, args.T, cfg['num_stems'])
        parity['hybrid_numpy_cos'] = cosine(outputs['views'], packed)
        parity['full_ane_numpy_cos'] = cosine(outputs['full_ane'], packed)
        assert parity['hybrid_numpy_cos'] > 0.999, parity
    print(f'Correctness gates: {parity}', flush=True)
    for _ in range(2):
        for fn in fns.values():
            fn()
    rows = []
    order_rng = random.Random(83)
    for r in range(args.rounds):
        order = list(fns)
        order_rng.shuffle(order)
        for mode in order:
            start = time.perf_counter_ns()
            out = fns[mode]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            rows.append(dict(round=r, mode=mode, elapsed_ms=elapsed))
        print(f'Round {r + 1}/{args.rounds}: ' + ', '.join(
            f'{row["mode"]}={row["elapsed_ms"]:.2f}ms' for row in rows[-3:]), flush=True)
    summary = {}
    for mode in fns:
        ts = [r['elapsed_ms'] for r in rows if r['mode'] == mode]
        summary[mode] = dict(min_ms=min(ts), median_ms=statistics.median(ts), max_ms=max(ts))
    result = dict(T=args.T, depth=args.depth, reduce_fix=args.reduce_fix,
                  dim=cfg['dim'], dim_head=cfg['dim_head'], heads=cfg['heads'],
                  parity=parity, summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Raw samples: {args.output}')
    for net in [*nets.values(), full]:
        net.release()


if __name__ == '__main__':
    main()
