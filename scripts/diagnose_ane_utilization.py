"""ANE shape/utilization audit: MAC-heavy controls versus actual model shapes.

Reports device API execute-wall timing separately from public Python-call timing.
execute() is synchronous host wall time, NOT a hardware MAC utilization counter.
FMA counts as 2 FLOPs. Isolated operator times must not be summed as fused-graph
attribution. Ideal bytes/intensity describe unfused fp16 I/O only, not measured
DRAM traffic. System paging deltas are sampled around each timed block.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import warnings

import numpy as np
import aneforge as af
import hybrid_probe as hp

ROOT = Path(__file__).resolve().parents[1]


def memory_counters():
    p = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True)
    keys = {'Swapins', 'Swapouts', 'Pageins', 'Pageouts', 'Decompressions', 'Compressions'}
    return {line.split(':', 1)[0]: int(line.split(':', 1)[1].strip().rstrip('.'))
            for line in p.stdout.splitlines() if ':' in line and line.split(':', 1)[0] in keys}


def stat(ts):
    return dict(min_ms=min(ts), median_ms=statistics.median(ts), max_ms=max(ts))


def factories(T, cfg, weights, suite):
    B, D, H, dh = 60, cfg['dim'], cfg['heads'], cfg['dim_head']
    M, hd = B * T, H * dh
    rng = np.random.default_rng(75)
    if 'projections' in suite:
        p = 'layers.0.0.layers.0'
        for label, key in [('qkv', p + '.0.to_qkv.weight'), ('out_projection', p + '.0.to_out.0.weight'),
                           ('ffn_up', p + '.1.net.1.weight'), ('ffn_down', p + '.1.net.4.weight')]:
            w = weights[key]
            N, K = w.shape
            def build(w=w, K=K):
                return af.input((M, K)).linear(w)
            yield label, build, 2 * M * K * N, dict(M=M, K=K, N=N, ideal_io_bytes=2 * (M*K + K*N + M*N))
    if 'attention' in suite:
        for direction, batch, seq in [('time', B, T), ('freq', T, B)]:
            def qk(batch=batch, seq=seq):
                return af.input((batch, H, seq, dh)) @ af.input((batch, H, dh, seq))
            def av(batch=batch, seq=seq):
                return af.input((batch, H, seq, seq)) @ af.input((batch, H, seq, dh))
            io = 2 * batch * H * (2 * seq * dh + seq * seq)
            flops = 2 * batch * H * seq * seq * dh
            geometry = dict(batch=batch, heads=H, sequence=seq, dim_head=dh, ideal_io_bytes=io)
            yield direction + '_qk', qk, flops, geometry
            yield direction + '_av', av, flops, geometry
            def softmax(batch=batch, seq=seq):
                x = af.input((batch, H, seq, seq))
                e = ((x - x.amax(-1)) * float(np.log2(np.e))).exp2()
                return e / (e @ np.ones((seq, 1), np.float16))
            yield direction + '_softmax_exp2_ones', softmax, None, dict(geometry, input_bytes=2*batch*H*seq*seq)
            def core(batch=batch, seq=seq):
                q = af.input((batch, H, seq, dh))
                k = af.input((batch, H, dh, seq))
                v = af.input((batch, H, seq, dh))
                sim = (q @ k) * (dh ** -0.5)
                e = ((sim - sim.amax(-1)) * float(np.log2(np.e))).exp2()
                return (e @ v) / (e @ np.ones((seq, 1), np.float16))
            yield direction + '_attention_core', core, None, dict(geometry, gemm_flops_only=2*flops)
    if 'controls' in suite:
        # A compute-dense control and a same-width chained control. These do not
        # predict model speed: they test whether this API/device can approach peak.
        C, S, depth = 1024, 4096, 8
        conv_weights = [rng.normal(0, 0.01, (C, C, 1, 1)).astype(np.float16) for _ in range(depth)]
        def conv_chain():
            x = af.input((1, C, 1, S))
            for w in conv_weights:
                x = af.conv(x, w)
            return x
        yield 'dense_conv1x1_chain8', conv_chain, 2 * depth * S * C * C, dict(C=C, spatial=S, chain=depth)
        K = N = 512
        dense_weights = [rng.normal(0, 0.02, (N, K)).astype(np.float16) for _ in range(4)]
        def mm_chain():
            x = af.input((M, K))
            for w in dense_weights:
                x = x.linear(w)
            return x
        yield 'dense_linear512_chain4', mm_chain, 2 * 4 * M * K * N, dict(M=M, K=K, N=N, chain=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--suite', default='projections,attention,controls')
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_utilization_T256.json')
    args = ap.parse_args()
    cfg, weights, _ = hp.load_all()
    rows = []
    rng = np.random.default_rng(83)
    print(f'Actual model: dim={cfg["dim"]} heads={cfg["heads"]} dim_head={cfg["dim_head"]}; T={args.T}', flush=True)
    for name, build, flops, geometry in factories(args.T, cfg, weights, set(args.suite.split(','))):
        net = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                start = time.perf_counter()
                graph = build()
                net = af.compile(graph)
                compile_s = time.perf_counter() - start
            feeds = []
            for port, shape in net._inputs:
                a = rng.normal(0, 0.1, shape).astype(np.float16)
                feeds.append(a)
                np.copyto(net.input_view(port), a)
            for _ in range(args.warmup):
                net.execute()
            before = memory_counters()
            execute_samples = []
            for _ in range(args.reps):
                start = time.perf_counter_ns()
                net.execute()
                execute_samples.append((time.perf_counter_ns() - start) / 1e6)
            after = memory_counters()
            expected = net.output_view().copy()
            assert np.isfinite(expected).all(), name
            call_samples = []
            for _ in range(min(5, args.reps)):
                start = time.perf_counter_ns()
                actual = net(*feeds)
                call_samples.append((time.perf_counter_ns() - start) / 1e6)
            np.testing.assert_array_equal(expected, actual.astype(np.float16))
            ex, call = stat(execute_samples), stat(call_samples)
            row = dict(name=name, geometry=geometry, compile_s=compile_s, n_ops=net.n_ops,
                       execute=ex, public_call=call, samples_ms=execute_samples,
                       paging_delta={key: after[key] - before[key] for key in before},
                       public_vs_execute_bitwise_equal=True, flops=flops)
            if flops is not None:
                row['execute_tflops'] = flops / (ex['median_ms'] / 1000) / 1e12
                row['public_call_tflops'] = flops / (call['median_ms'] / 1000) / 1e12
            rows.append(row)
            throughput = f'{row["execute_tflops"]:.2f} TFLOPS' if flops else 'non-MAC / mixed'
            print(f'{name}: execute min/median/max={ex["min_ms"]:.3f}/{ex["median_ms"]:.3f}/{ex["max_ms"]:.3f}ms; '
                  f'public={call["median_ms"]:.3f}ms; {throughput}; paging={row["paging_delta"]}', flush=True)
        except Exception as e:
            rows.append(dict(name=name, error=f'{type(e).__name__}: {str(e)[:400]}'))
            print(f'{name}: FAIL {type(e).__name__}: {str(e)[:160]}', flush=True)
        finally:
            if net is not None:
                net.release()
            gc.collect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(T=args.T, cfg={k:cfg[k] for k in ['dim','heads','dim_head','depth']}, rows=rows), indent=2))
    print(f'Raw samples: {args.output}', flush=True)


if __name__ == '__main__':
    main()
