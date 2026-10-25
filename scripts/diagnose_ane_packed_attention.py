"""Gate batch/head packing at full-attention-block or full-model scope.

The production default remains unpacked. Compare identical actual weights,
inputs, precision rewrites and execution APIs; randomize timed case order.
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
import ane_attention_candidates as candidates
from diagnose_ane_utilization import memory_counters, stat
from diagnose_hybrid_e2e import cosine, pack_reference

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--stage', choices=['block', 'full'], default='block')
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--variants', help='Optional comma-separated case labels; include baseline')
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_packed_attention_T256.json')
    args = ap.parse_args()
    cfg, w, dims = hp.load_all()
    cfg = dict(cfg)
    cfg['depth'] = args.depth
    rng = np.random.default_rng(53)
    rows = []
    for direction in (['time', 'freq'] if args.stage == 'block' else ['model']):
        nets, outputs = {}, {}
        try:
            if direction == 'model':
                feeds = []
                xin = rng.normal(0, 0.5, (1, args.T, sum(dims))).astype(np.float16)
                if args.T >= 385:
                    off = 0
                    for s, e, d in hp.bg.dim_groups(dims):
                        width = (e-s)*d
                        feeds.append(xin[:, :, off:off+width].reshape(1,args.T,e-s,d))
                        off += width
                else:
                    feeds = [xin]
                options = [('baseline', False), ('freq_packed', 'freq'), ('all_packed', True)]
            else:
                B, S = (60, args.T) if direction == 'time' else (args.T, 60)
                feeds = [rng.normal(0, 0.1, (B,S,cfg['dim'])).astype(np.float16)]
                options = [('baseline', False), ('packed', True)]
            options.append(('rope_halves', False))
            if args.variants:
                selected = set(args.variants.split(','))
                assert 'baseline' in selected
                options = [(label, value) for label, value in options if label in selected]
            for label, option in options:
                local_cfg = dict(cfg, attn_pack_heads=option, rope_half_layout=label == 'rope_halves')
                if direction == 'model':
                    # Single-process diagnostic override, scoped to graph construction.
                    # Production build_graph is unchanged.
                    original_attn = hp.bg.attn
                    try:
                        hp.bg.attn = candidates.attn
                        graph = hp.bg.build(args.T, local_cfg, w, dims)
                    finally:
                        hp.bg.attn = original_attn
                else:
                    branch = 0 if direction == 'time' else 1
                    prefix = f'layers.0.{branch}.layers.0.0'
                    rope = hp.bg.rope_bakes(w[prefix + '.rotary_embed.freqs'], S)
                    graph = candidates.attn(af.input(feeds[0].shape), w, local_cfg, rope, prefix)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    net = af.compile(af.reduce_sum_to_matmul(graph))
                nets[label] = net
                for (port, shape), data in zip(net._inputs, feeds, strict=True):
                    assert tuple(shape) == data.shape
                    np.copyto(net.input_view(port), data)
                for _ in range(3):
                    net.execute()
                outputs[label] = net.output_view().copy()
                print(f'Compiled {direction}/{label}, {net.n_ops} ops', flush=True)
            parity = {}
            for label, actual in outputs.items():
                if label == 'baseline':
                    continue
                delta = actual.astype(np.float64) - outputs['baseline'].astype(np.float64)
                rel_l2 = float(np.linalg.norm(delta.ravel()) / np.linalg.norm(outputs['baseline'].astype(np.float64).ravel()))
                parity[label] = dict(cosine=cosine(actual, outputs['baseline']), relative_l2=rel_l2,
                                     bitwise_equal=bool(np.array_equal(actual, outputs['baseline'])))
                assert parity[label]['cosine'] > 0.999 and rel_l2 < 0.02, parity
            if args.reference:
                assert direction == 'model', '--reference requires --stage full'
                import parity as par
                w64 = {k:v.astype(np.float64) for k,v in w.items() if v.dtype == np.float16}
                ref = pack_reference(par.ref_forward(xin.astype(np.float64), w64, cfg, dims), dims, args.T, cfg['num_stems'])
                parity['numpy_reference'] = {key: cosine(val,ref) for key,val in outputs.items()}
                assert all(c > 0.999 for c in parity['numpy_reference'].values()), parity
            print(f'{direction} correctness: {parity}', flush=True)
            before = memory_counters()
            samples = []
            order_rng = random.Random(97)
            for rep in range(args.reps):
                order = list(nets)
                order_rng.shuffle(order)
                for label in order:
                    t = time.perf_counter_ns()
                    nets[label].execute()
                    elapsed = (time.perf_counter_ns()-t)/1e6
                    samples.append(dict(rep=rep, label=label, execute_ms=elapsed))
            after = memory_counters()
            summary = {key:stat([r['execute_ms'] for r in samples if r['label']==key]) for key in nets}
            row = dict(direction=direction, parity=parity, summary=summary, samples=samples,
                       paging_delta={k:after[k]-before[k] for k in before})
            rows.append(row)
            print(f'{direction} timing: {summary}', flush=True)
        finally:
            for net in nets.values():
                net.release()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(T=args.T, depth=args.depth, stage=args.stage,
                                                  reduce_fix=True, rows=rows), indent=2))
    print(f'Raw samples: {args.output}', flush=True)


if __name__ == '__main__':
    main()
