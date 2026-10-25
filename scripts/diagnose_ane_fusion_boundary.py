"""Matched one-program versus QK-cut attention, with explicit copy accounting.

This is an optimization-boundary experiment, NOT a DRAM/SRAM traffic counter.
Use identical fp16 inputs, head dim64, and e@V / rowsum(e) in both versions.
execute-wall timing includes synchronous runtime overhead. Split includes a
score np.copyto (no public-call fp32 conversions). Never sum isolated medians
and label the difference hardware fusion. Raw paired rounds are persisted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import aneforge as af
from diagnose_ane_utilization import memory_counters, stat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rev'))
from e5_parser import parse as parse_e5


def tail_graph(score, value, seq):
    e = ((score - score.amax(-1)) * float(np.log2(np.e))).exp2()
    return (e @ value) / (e @ np.ones((seq, 1), np.float16))


def compile_record(graph, directory):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(graph, build_dir=str(directory))
    metadata = dict(build_dir=str(directory), frontend_nodes=net.n_ops,
                    input_shapes=[list(shape) for _, shape in net._inputs],
                    mil_sha256=hashlib.sha256((directory / 'model.mil').read_bytes()).hexdigest())
    metadata['e5'] = []
    for path in directory.rglob('H16G.e5'):
        p = parse_e5(path.read_bytes(), str(path))
        names = [s['name'] for s in p['sections']]
        metadata['e5'].append(dict(path=str(path), sha256=p['sha256'],
                                   section_names=names, parser_td_blocks=len(p['td_blocks'])))
    return net, metadata


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))


def numpy_lane(q, k, v):
    a = q[:1, :1].astype(np.float64) @ k[:1, :1].astype(np.float64)
    a *= q.shape[-1] ** -0.5
    e = np.exp(a - a.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ v[:1, :1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--rounds', type=int, default=15)
    ap.add_argument('--head-dim', type=int, default=64)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_fusion_boundary_T256.json')
    args = ap.parse_args()
    rng = np.random.default_rng(107)
    rows = []
    for direction, B, S in [('time', 60, args.T), ('freq', args.T, 60)]:
        H, D = 8, args.head_dim
        nets = []
        try:
            q = af.input((B,H,S,D))
            kt = af.input((B,H,D,S))
            v = af.input((B,H,S,D))
            score = (q @ kt) * (D ** -0.5)
            full_graph = tail_graph(score, v, S)
            full, full_meta = compile_record(full_graph, ROOT / f'.tmp/ane_fusion_boundary/T{args.T}_{direction}_full')
            nets.append(full)
            qk, qk_meta = compile_record(score, ROOT / f'.tmp/ane_fusion_boundary/T{args.T}_{direction}_qk')
            nets.append(qk)
            score_in = af.input((B,H,S,S))
            value_in = af.input((B,H,S,D))
            tail, tail_meta = compile_record(tail_graph(score_in,value_in,S), ROOT / f'.tmp/ane_fusion_boundary/T{args.T}_{direction}_tail')
            nets.append(tail)
            q_np = rng.normal(0,0.1,(B,H,S,D)).astype(np.float16)
            k_np = rng.normal(0,0.1,(B,H,D,S)).astype(np.float16)
            v_np = rng.normal(0,0.1,(B,H,S,D)).astype(np.float16)
            for (port, shape), array in zip(full._inputs, [q_np,k_np,v_np], strict=True):
                assert tuple(shape)==array.shape
                np.copyto(full.input_view(port),array)
            for (port, shape), array in zip(qk._inputs,[q_np,k_np],strict=True):
                assert tuple(shape)==array.shape
                np.copyto(qk.input_view(port),array)
            tail_score = tail.input_view(tail._inputs[0][0])
            tail_value = tail.input_view(tail._inputs[1][0])
            np.copyto(tail_value,v_np)
            qk_score = qk.output_view()
            def split():
                t = time.perf_counter_ns()
                qk.execute()
                a = time.perf_counter_ns()
                np.copyto(tail_score,qk_score)
                b = time.perf_counter_ns()
                tail.execute()
                c = time.perf_counter_ns()
                return dict(qk_ms=(a-t)/1e6, score_copy_ms=(b-a)/1e6,
                            tail_ms=(c-b)/1e6, total_ms=(c-t)/1e6)
            full.execute()
            split()
            expected = full.output_view().copy()
            actual = tail.output_view().copy()
            a = actual.astype(np.float64).ravel()
            b = expected.astype(np.float64).ravel()
            parity = dict(cosine=cosine(actual,expected),
                          relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                          bitwise_equal=bool(np.array_equal(actual,expected)))
            ref = numpy_lane(q_np,k_np,v_np)
            parity['full_numpy_lane_cosine'] = cosine(expected[:1,:1],ref)
            parity['split_numpy_lane_cosine'] = cosine(actual[:1,:1],ref)
            assert parity['cosine']>0.999 and parity['relative_l2']<0.003, parity
            assert min(parity['full_numpy_lane_cosine'],parity['split_numpy_lane_cosine'])>0.999, parity
            for _ in range(4):
                full.execute()
                split()
            before = memory_counters()
            samples = []
            order_rng = random.Random(113)
            for r in range(args.rounds):
                order = ['full','split']
                order_rng.shuffle(order)
                for mode in order:
                    if mode=='full':
                        t = time.perf_counter_ns()
                        full.execute()
                        result = dict(total_ms=(time.perf_counter_ns()-t)/1e6)
                    else:
                        result = split()
                    samples.append(dict(round=r,mode=mode,**result))
            after = memory_counters()
            summary = {mode:stat([r['total_ms'] for r in samples if r['mode']==mode]) for mode in ['full','split']}
            parts = {key:stat([r[key] for r in samples if r['mode']=='split'])
                     for key in ['qk_ms','score_copy_ms','tail_ms']}
            # Derive per-sample residual first, rather than subtracting medians.
            residual = []
            for r in range(args.rounds):
                single = next(x for x in samples if x['round']==r and x['mode']=='full')
                cut = next(x for x in samples if x['round']==r and x['mode']=='split')
                residual.append(cut['qk_ms']+cut['tail_ms']-single['total_ms'])
            row = dict(direction=direction,B=B,H=H,S=S,D=D,score_bytes=int(qk_score.nbytes),
                       parity=parity,summary=summary,split_parts=parts,
                       paired_split_compute_minus_full_ms=stat(residual),samples=samples,
                       paging_delta={k:after[k]-before[k] for k in before},
                       artifacts=dict(full=full_meta,qk=qk_meta,tail=tail_meta))
            rows.append(row)
            print(f'{direction}: score={qk_score.nbytes/1e6:.2f}MB; correctness={parity}',flush=True)
            print(f'{direction}: total={summary}; split parts={parts}; compute gap={stat(residual)}',flush=True)
            print(f'{direction}: paging={row["paging_delta"]}',flush=True)
        finally:
            for net in nets:
                net.release()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(T=args.T,head_dim=args.head_dim,rows=rows),indent=2))
    print(f'Raw samples: {args.output}',flush=True)


if __name__=='__main__':
    main()
