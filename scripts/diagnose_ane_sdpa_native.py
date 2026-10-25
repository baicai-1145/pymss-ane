"""Direct timing probe of the native ANE fused-attention bridge (Type=SDPA netlist).

The earlier "sdpa bridge never compiles" verdict came from the segmented-compile
path on our production shapes. This probe bypasses MIL entirely: it builds a
one-op netplist and dispatches via the sdpa_invoker binary. Numerics are gated
against a NumPy fp32 reference; timing is steady-state repeats inside a single
invoker process (not Python call overhead).

Our production time-direction attention is B=60,H=8,S=256,D=64. The netlist path
accepts B=1 only, so the probe folds batch into heads: [1, 256, 480, 64]
(ANE layout seq-in-C, heads-in-H). That is mathematically identical attention
per head: softmax((q@k^T)*scale)@v computed independently for each of the 480
head instances. freq direction stays [1, 60, 8, 64] reshaped likewise.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aneforge._bridges.ane_sdpa_fused import sdpa_fused, numpy_reference


def bench(label, Q, K, V, rng, repeats=30, warmup=8):
    ref = numpy_reference(Q, K, V)
    Y, info = sdpa_fused(Q, K, V, repeats=repeats, warmup=warmup, return_details=True)
    Y = np.asarray(Y)
    assert Y.shape == ref.shape, (Y.shape, ref.shape)
    assert np.isfinite(Y.astype(np.float32)).all()
    cos = float((Y.astype(np.float64).ravel() @ ref.astype(np.float64).ravel())
                / (np.linalg.norm(Y) * np.linalg.norm(ref)))
    rel = float(np.linalg.norm((Y.astype(np.float64) - ref.astype(np.float64)).ravel())
                / np.linalg.norm(ref.astype(np.float64)))
    row = dict(label=label, shape=list(Q.shape),
               cos=cos, rel=rel,
               max_abs=float(np.abs(Y.astype(np.float32) - ref.astype(np.float32)).max()),
               info={k: info.get(k) for k in ('repeats', 'warmup', 'min_us', 'mean_us', 'p50_us', 'max_us') if isinstance(info, dict) and k in info})
    print(f'{label}: shape={Q.shape} cos={cos:.6f} rel={rel:.6f} info={row["info"]}', flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--output', type=Path, default=ROOT / '.tmp/ane_sdpa_native_bridge.json')
    args = ap.parse_args()
    rng = np.random.default_rng(211)
    rows = []
    # Canonical bridge shapes first: sanity against the published cost model.
    for H, S, D in [(8, 64, 64), (8, 128, 64)]:
        Q, K, V = (rng.normal(0, 1, (1, H, S, D)).astype(np.float16) for _ in range(3))
        rows.append(bench(f'canonical_H{H}_S{S}_D{D}', Q, K, V, rng, args.repeats, args.warmup))
    # Production time direction: B=60,H=8,S=256,D=64 folded to B=1,H=480.
    B, H, S, D = 60, 8, 256, 64
    Qf = rng.normal(0, 1, (1, S, B * H, D)).astype(np.float16)  # ANE layout folded
    def unfold_to_bhsh(X):
        # [1, S, B*H, D] -> [B, H, S, D] for the reference path
        return np.ascontiguousarray(X.transpose(0, 2, 1, 3)[0].reshape(B, H, S, D))
    ref = numpy_reference(unfold_to_bhsh(Qf), unfold_to_bhsh(Qf), unfold_to_bhsh(Qf))
    # sdpa_fused expects [B,H,S,D] user layout and transposes internally; feed folded B=1.
    # Fold B into H directly: [1, B*H, S, D]
    Qb = np.ascontiguousarray(Qf.transpose(0, 2, 1, 3))  # [1, B*H, S, D]
    Kb, Vb = Qb.copy(), Qb.copy()
    Y, info = sdpa_fused(Qb, Kb, Vb, repeats=args.repeats, warmup=args.warmup, return_details=True)
    # per-head equivalent reference for folded layout
    ref_folded = numpy_reference(Qb, Kb, Vb)
    cos = float((Y.astype(np.float64).ravel() @ ref_folded.astype(np.float64).ravel())
                / (np.linalg.norm(Y) * np.linalg.norm(ref_folded)))
    rel = float(np.linalg.norm(Y.astype(np.float64) - ref_folded.astype(np.float64))
                / np.linalg.norm(ref_folded.astype(np.float64)))
    rows.append(dict(label='folded_time_B60H8S256D64', shape=list(Qb.shape),
                     cos=cos, rel=rel,
                     info={k: info.get(k) for k in ('min_us', 'mean_us', 'p50_us', 'max_us') if isinstance(info, dict) and k in info}))
    print(f'folded time-dir: shape={Qb.shape} cos={cos:.6f} rel={rel:.6f} info={rows[-1]["info"]}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
