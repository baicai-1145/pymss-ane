"""Falsifiable layout/alignment and RoPE tests, using execute-wall timing only.

Same mathematical inputs per comparison; padded variants include in-graph zero
padding and crop. Packing head/batch dimensions preserves independent products.
No GPU, no projection rewriting, no claim that isolated wins carry to full graph.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import warnings

import numpy as np
import aneforge as af
from aneforge.graph import _const
import hybrid_probe as hp
from diagnose_ane_utilization import memory_counters, stat

ROOT = Path(__file__).resolve().parents[1]


def crop(x, shape):
    return x.slice_by_size([0] * len(shape), list(shape))


def bmm_case(op, layout, B, H, S, D):
    left_shape = (B, H, S, D) if op == 'qk' else (B, H, S, S)
    right_shape = (B, H, D, S) if op == 'qk' else (B, H, S, D)
    left, right = af.input(left_shape), af.input(right_shape)
    if layout == 'packed4d':
        a = left.reshape((1, B*H, *left_shape[-2:]))
        b = right.reshape((1, B*H, *right_shape[-2:]))
        out = (a @ b).reshape((B, H, S, S if op == 'qk' else D))
    elif layout == 'packed3d':
        a = left.reshape((B*H, *left_shape[-2:]))
        b = right.reshape((B*H, *right_shape[-2:]))
        out = (a @ b).reshape((B, H, S, S if op == 'qk' else D))
    elif layout == 'pad64':
        P = (-S) % 64
        assert P > 0
        if op == 'qk':
            a = af.concat([left, _const(np.zeros((B, H, P, D), np.float16))], axis=2)
            b = af.concat([right, _const(np.zeros((B, H, D, P), np.float16))], axis=3)
            out = crop(a @ b, (B, H, S, S))
        else:
            a = af.concat([left, _const(np.zeros((B, H, S, P), np.float16))], axis=3)
            a = af.concat([a, _const(np.zeros((B, H, P, S+P), np.float16))], axis=2)
            b = af.concat([right, _const(np.zeros((B, H, P, D), np.float16))], axis=2)
            out = crop(a @ b, (B, H, S, D))
    else:
        out = left @ right
    return out


def rope_case(implementation, shape, rope):
    B, S, H, D = shape
    q = af.input(shape)
    cd, sd, permutation = rope
    if implementation == 'matmul':
        rotated = q @ permutation
    else:
        # [even,odd] -> [-odd,even] without a dense D x D matrix multiply.
        pairs = q.reshape((B, S*H, D//2, 2))
        even = pairs.slice_by_size([0, 0, 0, 0], [B, S*H, D//2, 1])
        odd = pairs.slice_by_size([0, 0, 0, 1], [B, S*H, D//2, 1])
        rotated = af.concat([odd * -1.0, even], axis=-1).reshape(shape)
    return q * cd[None, :, None, :] + rotated * sd[None, :, None, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_attention_layout_T256.json')
    args = ap.parse_args()
    cfg, w, _ = hp.load_all()
    H, D = cfg['heads'], cfg['dim_head']
    rng = np.random.default_rng(93)
    rows = []
    def run(name, graph_fn, feeds, reference=None):
        net = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                net = af.compile(graph_fn())
            for (port, _), a in zip(net._inputs, feeds, strict=True):
                np.copyto(net.input_view(port), a)
            for _ in range(4):
                net.execute()
            before = memory_counters()
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter_ns()
                net.execute()
                ts.append((time.perf_counter_ns()-t)/1e6)
            after = memory_counters()
            actual = net.output_view().copy()
            row = dict(name=name, execute=stat(ts), samples_ms=ts,
                       paging_delta={k:after[k]-before[k] for k in before})
            if reference is not None:
                a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
                row['relative_l2'] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                row['bitwise_equal'] = bool(np.array_equal(actual, reference))
                assert row['relative_l2'] < 0.002, row
            rows.append(row)
            print(f'{name}: median={row["execute"]["median_ms"]:.3f} min={row["execute"]["min_ms"]:.3f}ms '
                  f'rel_l2={row.get("relative_l2",0):.6f} paging={row["paging_delta"]}', flush=True)
            return actual
        except Exception as e:
            rows.append(dict(name=name, error=f'{type(e).__name__}: {str(e)[:300]}'))
            print(f'{name}: FAIL {type(e).__name__}: {str(e)[:160]}', flush=True)
            return None
        finally:
            if net is not None:
                net.release()
            gc.collect()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(T=args.T, heads=H, dim_head=D, rows=rows), indent=2))

    B, S = args.T, 60
    for op in ['qk', 'av']:
        lshape = (B, H, S, D) if op == 'qk' else (B, H, S, S)
        rshape = (B, H, D, S) if op == 'qk' else (B, H, S, D)
        feeds = [rng.normal(0, 0.1, shape).astype(np.float16) for shape in (lshape, rshape)]
        ref = run('freq_' + op + '_baseline', lambda op=op:bmm_case(op, 'baseline', B,H,S,D), feeds)
        if ref is not None:
            for layout in ['packed4d', 'packed3d', 'pad64']:
                run(f'freq_{op}_{layout}', lambda op=op, layout=layout:bmm_case(op, layout, B,H,S,D), feeds, ref)
    for direction, B, S, freqkey in [('time',60,args.T,'layers.0.0.layers.0.0.rotary_embed.freqs'),
                                    ('freq',args.T,60,'layers.0.1.layers.0.0.rotary_embed.freqs')]:
        shape = (B,S,H,D)
        rope = hp.bg.rope_bakes(w[freqkey], S)
        feeds = [rng.normal(0,0.1,shape).astype(np.float16)]
        ref = run(direction + '_rope_matmul', lambda:rope_case('matmul',shape,rope), feeds)
        if ref is not None:
            run(direction + '_rope_slice', lambda:rope_case('slice',shape,rope), feeds, ref)
    print(f'Raw samples: {args.output}', flush=True)


if __name__ == '__main__':
    main()
