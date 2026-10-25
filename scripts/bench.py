"""Benchmark: full mel-band-roformer-deux ANE program vs timing budget.

Measures: dispatch latency across T (stft frames), TFLOPS estimate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_graph as bg


def model_flops(cfg, T):
    """Analytic FLOPs (2*MACs) of the ANE program at stft frames T."""
    dim, depth = cfg["dim"], cfg["depth"]
    heads, dh = cfg["heads"], cfg["dim_head"]
    hd = heads * dh
    F = 60
    flops = 0
    # bandsplit: per band d_in->dim matmul, over T frames
    dims_total = 7916
    flops += 2 * dims_total * dim * T
    for d in range(depth):
        # time transformer: batch F=60, seq T
        # attn: qkv (dim->3hd), out (hd->dim), gates(dim->heads), sim+av
        seq = T
        b = F
        flops += 2 * (dim * 3 * hd + dim * heads + hd * dim) * b * seq
        flops += 2 * 2 * heads * seq * seq * dh * b
        # ffn: dim->4dim->dim
        flops += 2 * (dim * 4 * dim + 4 * dim * dim) * b * seq
        # freq transformer: batch T, seq F
        seq = F
        b = T
        flops += 2 * (dim * 3 * hd + dim * heads + hd * dim) * b * seq
        flops += 2 * 2 * heads * seq * seq * dh * b
        flops += 2 * (dim * 4 * dim + 4 * dim * dim) * b * seq
    # mask estimators: S stems x 60 bands x (256->1024->1024->2din)
    S = cfg["num_stems"]
    flops += 2 * S * (256 * 1024 + 1024 * 1024) * 60 * T
    flops += 2 * S * 1024 * dims_total * T
    return flops


def main():
    import aneforge as af

    cfg_full = bg.load_cfg("/Volumes/nfs/data/Downloads/mel-band-roformer-deux/config_deux_becruily.yaml")
    wdir = Path("weights")
    manifest = json.loads((wdir / "manifest.json").read_text())
    w = {k: np.load(wdir / v) for k, v in manifest.items()}
    dims = [w[f"band_split.to_features.{i}.1.weight"].shape[1] for i in range(60)]
    L = sum(dims)

    for T in [64, 128, 256, 512]:
        out = bg.build(T, cfg_full, w, dims)
        t0 = time.time()
        net = af.compile(af.reduce_sum_to_matmul(out))
        ct = time.time() - t0
        rng = np.random.default_rng(0)
        L = sum(dims)
        xin = (rng.standard_normal((1, T, L)) * .5).astype(np.float16)
        if T >= 385:
            # multi-input bandsplit: slice xin along L per group
            feeds = []
            off = 0
            for (s, e, d) in bg.dim_groups(dims):
                w_g = (e - s) * d
                feeds.append(xin[:, :, off:off + w_g].reshape(1, T, e - s, d))
                off += w_g
            args = tuple(feeds)
        else:
            args = (xin,)
        # warmup
        y = net(*args)
        # timed loop
        N = 30
        t0 = time.time()
        for _ in range(N):
            y = net(*args)
        dt = (time.time() - t0) / N
        fl = model_flops(cfg_full, T)
        print(f"T={T:4d}: compile={ct:6.1f}s  latency={dt*1000:8.2f}ms  "
              f"throughput={fl/dt/1e12:6.2f} TFLOPS  ops={net.n_ops}", flush=True)
        net.release()


if __name__ == "__main__":
    main()
