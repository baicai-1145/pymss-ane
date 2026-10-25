"""Audit real A-prime data handoff with direct, additive boundary timings.
Uses one real depth, fixed seeded input, and paired randomized legacy/view paths.
This measures implementation overhead; it does not establish a hardware cause.
"""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import random
import statistics
import time
import warnings

import numpy as np
import mlx.core as mx
import aneforge as af
import aneforge._runtime as rt
import hybrid_probe as hp

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--reduce-fix', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_gpu_bridge_audit.json')
    args = ap.parse_args()
    cfg, w, dims = hp.load_all()
    core = hp.MLXSDPACore(w, cfg, args.T)
    def compile_net(out):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return af.compile(af.reduce_sum_to_matmul(out) if args.reduce_fix else out)
    print('Compiling real depth-0 PA and PM2', flush=True)
    pa = compile_net(hp.build_pa(args.T, cfg, w, dims, 0))
    pm = compile_net(hp.build_pm2(args.T, cfg, w, dims, 0))
    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.1, (1, args.T, 60, cfg['dim'])).astype(np.float16)
    residual = x[0].transpose(1, 0, 2)
    pa_input = pa.input_view()
    pa_output = pa.output_view()
    pm_inputs = [pm.input_view(name) for name, _ in pm._inputs]
    pm_output = pm.output_view()

    def mark(timings, name, fn):
        t = time.perf_counter_ns()
        val = fn()
        timings[name] = (time.perf_counter_ns() - t) / 1e6
        return val

    def profiled_call(net, arrays, timings, prefix):
        # Same operations as Model.__call__ + Program.eval, separated without
        # baseline subtraction. Preserve the original copies and dtypes.
        p = net._prog
        feed = mark(timings, prefix + '.astype_f16', lambda: [a.astype(np.float16) for a in arrays])
        def bind():
            for (name, _), a in zip(net._inputs, feed):
                p._feed(name, a)
        mark(timings, prefix + '.set_input', bind)
        mark(timings, prefix + '.execute', p.execute)
        def read():
            for name, buf in p._out_buffers.items():
                v = buf.view(np.uint16)
                rc = rt._lib.ane_e5rt_program_get_output_fp16(p._handle, name.encode(), rt._ptr(v), v.size)
                if rc != 0:
                    raise RuntimeError(f'get_output rc={rc}')
            return p._out_buffers[net._out_name]
        out = mark(timings, prefix + '.get_output', read)
        return mark(timings, prefix + '.astype_f32', lambda: out.astype(np.float32))

    def forward(mode):
        timings = {}
        start = time.perf_counter_ns()
        if mode == 'legacy':
            qkv = profiled_call(pa, [x], timings, 'pa')
            qkv = mark(timings, 'bridge.qkv_to_f16', lambda: np.asarray(qkv, dtype=np.float16))
        else:
            mark(timings, 'pa.set_input_view', lambda: np.copyto(pa_input, x))
            mark(timings, 'pa.execute', pa.execute)
            qkv = pa_output
        g = mark(timings, 'bridge.mx_array', lambda: mx.array(qkv))
        y = mark(timings, 'gpu.graph_build', lambda: core.core(g))
        mark(timings, 'gpu.eval', lambda: mx.eval(y))
        o = mark(timings, 'bridge.np_export', lambda: np.asarray(y))
        o = mark(timings, 'bridge.contiguous', lambda: np.ascontiguousarray(o))
        if mode == 'legacy':
            result = profiled_call(pm, [o, residual], timings, 'pm')
            result = mark(timings, 'pm.back_to_f16', lambda: np.asarray(result).astype(np.float16))
        else:
            mark(timings, 'pm.copy_o', lambda: np.copyto(pm_inputs[0], o))
            mark(timings, 'pm.copy_residual', lambda: np.copyto(pm_inputs[1], residual))
            mark(timings, 'pm.execute', pm.execute)
            result = pm_output
        total = (time.perf_counter_ns() - start) / 1e6
        return dict(mode=mode, total_ms=total, stages_ms=timings,
                    unaccounted_ms=total - sum(timings.values())), result

    # Verify the manually instrumented legacy call against the public API.
    expected = pm(core.run_bridge(pa(x).astype(np.float16)), residual).astype(np.float16)
    _, legacy = forward('legacy')
    legacy = legacy.copy()
    _, views = forward('views')
    views = views.copy()
    np.testing.assert_array_equal(expected, legacy)
    np.testing.assert_array_equal(expected, views)
    source = np.arange(16, dtype=np.float16)
    wrapped = mx.array(source)
    mx.eval(wrapped)
    exported = np.asarray(wrapped)
    shares = bool(np.shares_memory(source, exported))
    # Warm both implementations before the randomized measured block.
    for _ in range(3):
        forward('legacy')
        forward('views')
    rows = []
    order_rng = random.Random(23)
    for r in range(args.rounds):
        order = ['legacy', 'views']
        order_rng.shuffle(order)
        for mode in order:
            row, _ = forward(mode)
            row['round'] = r
            rows.append(row)
    summary = {}
    for mode in ['legacy', 'views']:
        samples = [r for r in rows if r['mode'] == mode]
        summary[mode] = dict(total_median_ms=statistics.median(r['total_ms'] for r in samples),
                             total_min_ms=min(r['total_ms'] for r in samples),
                             max_unaccounted_ms=max(r['unaccounted_ms'] for r in samples),
                             stage_median_ms={k: statistics.median(r['stages_ms'][k] for r in samples)
                                              for k in samples[0]['stages_ms']})
    result = dict(T=args.T, rounds=args.rounds, dim=cfg['dim'], heads=cfg['heads'],
                  dim_head=cfg['dim_head'], reduce_fix=args.reduce_fix,
                  mx_array_shares_numpy_storage=shares, output_bitwise_equal=True,
                  summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f'Bitwise parity public-call/legacy/views PASS; mx.array shares source storage={shares}')
    for mode, val in summary.items():
        print(f'{mode}: total median={val["total_median_ms"]:.3f} min={val["total_min_ms"]:.3f}ms; '
              f'max unaccounted={val["max_unaccounted_ms"]:.3f}ms')
        for k, v in val['stage_median_ms'].items():
            print(f'  {k}: {v:.3f}ms')
    print(f'Raw samples: {args.output}')
    pa.release()
    pm.release()


if __name__ == '__main__':
    main()
