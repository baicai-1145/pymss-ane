"""T>=448 band-split concat compile failure: bisect the wall.

Variables: number of concat groups (dim_groups count), per-group size,
total L. T=512 no-bs compiles => wall is in concat groups, not total length.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import aneforge as af

F16 = np.float16


def probe_concat(n_groups: int, T: int, per_g: int = 128, out_dim: int = 64):
    """Mimic bandsplit: n_groups pieces [1,T,G,dim] -> concat axis=2."""
    x = af.input((1, T, n_groups * per_g), max_abs=16.0)
    outs = []
    for g in range(n_groups):
        off = g * per_g
        xs = x.slice_by_size([0, 0, off], [1, T, per_g]).reshape((1, T, 1, per_g))
        W = np.random.randn(1, per_g, out_dim).astype(F16) * 0.05
        y = xs.bmm_weight(W)                      # [1,T,1,out]
        outs.append(y)                            # concat axis=2 over per_g=1 groups
    z = af.concat(outs, axis=2)
    return z


def try_compile(fn, tag):
    try:
        t0 = time.time()
        net = af.compile(fn(), build_dir=None)
        ok = True
        msg = f"OK {time.time()-t0:.1f}s ops={net.n_ops}"
        net.release()
    except Exception as e:
        ok = False
        msg = f"FAIL {str(e)[:80]}"
    print(f"{tag}: {msg}", flush=True)
    return ok


def main():
    T = 512
    # variable 1: group count at fixed small per-group
    for n in (4, 8, 10, 12, 13, 16, 20, 32):
        try_compile(lambda n=n: probe_concat(n, T), f"T={T} groups={n} per_g=128")
    # variable 2: per-group size at 12 vs 13 groups
    for pg in (64, 128, 256):
        try_compile(lambda pg=pg: probe_concat(13, T, per_g=pg), f"T={T} groups=13 per_g={pg}")


if __name__ == "__main__":
    main()
