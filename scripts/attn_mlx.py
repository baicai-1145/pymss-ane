#!/usr/bin/env python
"""MLX attention layer for the hybrid track (ANE band-split/FFN/mask + GPU attention).

Semantics aligned with pymss-ane/scripts/build_graph.py attn() and
pymss-core mlx_attention.py:
  xn   = rms_norm(x, gamma, eps=1e-12)                 # fp16 in, fp16 out
  qkv  = xn @ W_qkv                                     # no bias, dim -> 3*heads*dh
  rope = interleaved-pair rotary, theta=10000, per-head freqs on dim_head
  attn = softmax(q k^T * dh^-0.5) v                    # [B,H,S,dh]
  out  = (attn * sigmoid(xn @ W_g + b_g)) @ W_out      # gates per head, no out bias

Variants timed (single layer + 12-layer chained pipeline, both directions):
  naive      : handwritten max/exp/sum softmax, all fp16 tensors
  sdpa       : mx.fast.scaled_dot_product_attention, fp16 (internal softmax fp32)
  sdpa32     : SDPA with fp32-cast q/k/v (fp32 QK^T/AV math), fp16 out
  opt        : fused metal rope kernel (reads contiguous qkv, emits [B,H,S,dh])
               + SDPA fp16 + broadcast gate-mul; projections as plain GEMMs
  core-*     : rope+SDPA+gate-mul only (qkv assumed produced on ANE side) --
               the hybrid-track GPU-side budget.

Numerics: every variant checked vs an fp64 numpy reference (cos > 0.9999).

Run:  ../../pymss-core/.venv/bin/python scripts/attn_mlx.py [--T 256] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

# --------------------------------------------------------------------------- #
# rope metal kernel: one thread per output element, reads contiguous qkv
# [B,S,3,H,dh], writes q/k already in [B,H,S,dh] (SDPA-ready, no transpose).
# cos/sin are interleaved-dup [S,dh] (repeat_interleave(2) layout).
# --------------------------------------------------------------------------- #
_ROPE_SRC = """
    uint elem = thread_position_in_grid.x;            // over qo [B,H,S,D]
    uint d = elem % D;
    uint pair = d >> 1;
    uint sd = elem / D;                               // (b*H + h)*S + s
    uint seq = sd % N;
    uint bh = sd / N;
    uint b = bh / H;
    uint h = bh % H;
    uint qbase = ((b * N + seq) * 3 + 0) * H * D + h * D;
    uint kbase = ((b * N + seq) * 3 + 1) * H * D + h * D;
    T qe = qkv_in[qbase + 2 * pair];                  // even lane of the pair
    T qo_ = qkv_in[qbase + 2 * pair + 1];             // odd lane
    T ke = qkv_in[kbase + 2 * pair];
    T ko_ = qkv_in[kbase + 2 * pair + 1];
    T c = cos[seq * D + d];
    T s = sin[seq * D + d];
    if ((d & 1) == 0) { qo[elem] = qe * c - qo_ * s; ko[elem] = ke * c - ko_ * s; }
    else              { qo[elem] = qo_ * c + qe * s; ko[elem] = ko_ * c + ke * s; }
"""
_rope_kernel = None


def _get_rope_kernel():
    global _rope_kernel
    if _rope_kernel is None:
        _rope_kernel = mx.fast.metal_kernel(
            name="hybrid_rope_qkv",
            input_names=["qkv_in", "cos", "sin"],
            output_names=["qo", "ko"],
            source=_ROPE_SRC,
            ensure_row_contiguous=True,
        )
    return _rope_kernel


def rope_metal(qkv: mx.array, cos: mx.array, sin: mx.array, H: int, dh: int):
    """qkv [B,S,3,H,dh] contiguous fp16 -> (q,k) [B,H,S,dh] contiguous fp16."""
    B, S = qkv.shape[0], qkv.shape[1]
    k = _get_rope_kernel()
    return k(
        inputs=[qkv, cos, sin],
        template=[("T", qkv.dtype), ("N", int(S)), ("H", int(H)), ("D", int(dh))],
        grid=(B * H * S * dh, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, H, S, dh), (B, H, S, dh)],
        output_dtypes=[qkv.dtype, qkv.dtype],
    )


def rope_naive(qkv: mx.array, cos: mx.array, sin: mx.array, H: int, dh: int):
    """Reference rope on strided views (slow: stack+reshape materializes)."""
    B, S = qkv.shape[0], qkv.shape[1]
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)   # [B,H,S,dh] view
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    c = cos[:, 0::2][None, None]                     # [1,1,S,dh/2] one freq per pair
    s = sin[:, 0::2][None, None]

    def rot(t):
        e, o = t[..., 0::2], t[..., 1::2]
        # interleaved-pair layout: (even_k, odd_k) neighbors; stack on LAST axis
        # interleaves them back to (e0,o0,e1,o1,...) — matches build_graph.py
        # `x*cos + (x@P)*sin` semantics.
        return mx.stack((e * c - o * s, o * c + e * s), axis=-1).reshape(t.shape)

    return rot(q), rot(k)


def make_cos_sin(S: int, dh: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float32) / dh))
    ang = np.arange(S, dtype=np.float32)[:, None] * freqs[None, :]
    cos = mx.array(np.repeat(np.cos(ang), 2, axis=1).astype(np.float16))  # [S,dh]
    sin = mx.array(np.repeat(np.sin(ang), 2, axis=1).astype(np.float16))
    return cos, sin, ang  # ang fp32 [S,dh/2] for the fp64 reference


# --------------------------------------------------------------------------- #
# weights (fp16, matching model shapes: dim=256, heads=8, dim_head=64)
# --------------------------------------------------------------------------- #
def make_weights(rng: np.random.Generator, dim=256, heads=8, dh=64):
    hd = heads * dh

    def w(*shape, scale=0.05):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float16))

    return {
        "norm_gamma": mx.array(np.ones(dim, np.float16)),
        "Wq": w(dim, 3 * hd),
        "Wg": w(dim, heads),
        "bg": mx.array((rng.standard_normal(heads) * 0.1).astype(np.float16)),
        "Wo": w(hd, dim),
    }


# --------------------------------------------------------------------------- #
# attention variants. All take x [B,S,dim] fp16 + weights + (cos,sin) and
# return y [B,S,dim] fp16 (residual NOT included).
# --------------------------------------------------------------------------- #
def _proj_qkv(xn, w, B, S, H, dh):
    return (xn @ w["Wq"]).reshape(B, S, 3, H, dh)


def attn_naive(x, w, cos, sin, H, dh):
    """Handwritten softmax: max / exp / sum, everything fp16 tensors."""
    B, S, D = x.shape
    xn = mx.fast.rms_norm(x, w["norm_gamma"], eps=1e-12)
    qkv = _proj_qkv(xn, w, B, S, H, dh)
    q, k = rope_naive(qkv, cos, sin, H, dh)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    scale = mx.array(dh ** -0.5, dtype=mx.float16)
    sim = (q @ k.transpose(0, 1, 3, 2)) * scale          # [B,H,S,S] fp16
    m = sim.max(-1, keepdims=True)
    e = mx.exp(sim - m)
    s = e.sum(-1, keepdims=True)
    o = (e @ v) / s
    gates = mx.sigmoid(xn @ w["Wg"] + w["bg"])
    out = o.transpose(0, 2, 1, 3).reshape(B, S, H, dh)
    return (out * gates[..., None]).reshape(B, S, H * dh) @ w["Wo"]


def attn_sdpa(x, w, cos, sin, H, dh, rope_fn=rope_naive, fp32_math=False):
    B, S, D = x.shape
    xn = mx.fast.rms_norm(x, w["norm_gamma"], eps=1e-12)
    qkv = _proj_qkv(xn, w, B, S, H, dh)
    q, k = rope_fn(qkv, cos, sin, H, dh)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    if fp32_math:
        o = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=dh ** -0.5
        ).astype(mx.float16)
    else:
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5)
    gates = mx.sigmoid(xn @ w["Wg"] + w["bg"])
    out = o.transpose(0, 2, 1, 3).reshape(B, S, H, dh)
    return (out * gates[..., None]).reshape(B, S, H * dh) @ w["Wo"]


def attn_opt(x, w, cos, sin, H, dh):
    """Fastest full-GPU layer: metal rope (qkv->BSHD direct) + SDPA fp16."""
    B, S, D = x.shape
    xn = mx.fast.rms_norm(x, w["norm_gamma"], eps=1e-12)
    qkv = _proj_qkv(xn, w, B, S, H, dh)
    qkv_c = qkv.reshape(B, S, 3, H, dh)  # already contiguous from GEMM reshape
    q, k = rope_metal(qkv_c, cos, sin, H, dh)
    v = qkv_c[:, :, 2].transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5)
    gates = mx.sigmoid(xn @ w["Wg"] + w["bg"])
    out = o.transpose(0, 2, 1, 3).reshape(B, S, H, dh)
    return (out * gates[..., None]).reshape(B, S, H * dh) @ w["Wo"]


# hybrid-split core: qkv arrives from ANE; GPU does rope+SDPA+gate-mul only
def attn_core_opt(qkv, x_norm, w, cos, sin, H, dh):
    """Hybrid split: qkv [B,S,3,H,dh] arrives from ANE; GPU does rope+SDPA+gate."""
    B, S = qkv.shape[0], qkv.shape[1]
    q, k = rope_metal(qkv, cos, sin, H, dh)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5)
    gates = mx.sigmoid(x_norm @ w["Wg"] + w["bg"])
    out = o.transpose(0, 2, 1, 3).reshape(B, S, H, dh)
    return (out * gates[..., None]).reshape(B, S, H * dh)


# --------------------------------------------------------------------------- #
# fp64 numpy reference (build_graph attn semantics)
# --------------------------------------------------------------------------- #
def attn_ref_fp64(x, w, ang, H, dh):
    B, S, D = x.shape
    xn = np.asarray(x, np.float64)
    xn = xn * (1.0 / np.sqrt((xn ** 2).mean(-1, keepdims=True) + 1e-12)) * np.asarray(w["norm_gamma"], np.float64)
    qkv = xn @ np.asarray(w["Wq"], np.float64)
    qkv = qkv.reshape(B, S, 3, H, dh)
    cosd = np.repeat(np.cos(ang), 2, axis=1).astype(np.float64)   # [S,dh]
    sind = np.repeat(np.sin(ang), 2, axis=1).astype(np.float64)

    def rot(t):
        e, o = t[..., 0::2], t[..., 1::2]
        c, s = cosd[:, 0::2][None, None], sind[:, 0::2][None, None]  # one freq per pair
        return np.stack((e * c - o * s, o * c + e * s), axis=-1).reshape(t.shape)  # interleave pairs
    q = rot(qkv[:, :, 0].transpose(0, 2, 1, 3))
    k = rot(qkv[:, :, 1].transpose(0, 2, 1, 3))
    v = np.asarray(qkv[:, :, 2].transpose(0, 2, 1, 3), np.float64)
    sim = np.einsum("bhsd,bhtd->bhst", q, k) * (dh ** -0.5)
    sim -= sim.max(-1, keepdims=True)
    e = np.exp(sim)
    o = np.einsum("bhst,bhtd->bhsd", e / e.sum(-1, keepdims=True), v)
    gates = 1.0 / (1.0 + np.exp(-(xn @ np.asarray(w["Wg"], np.float64) + np.asarray(w["bg"], np.float64))))
    out = o.transpose(0, 2, 1, 3).reshape(B, S, H, dh)
    return (out * gates[..., None]).reshape(B, S, H * dh) @ np.asarray(w["Wo"], np.float64)


def cos_sim(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def bench(fn, n=30, warm=8):
    for _ in range(warm):
        out = fn()
    mx.eval(out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000  # p50 ms


def pipeline(vfn, x0, layers=12):
    """Chained layers with real data dependency: x <- attn(x) + x."""

    def run():
        x = x0
        for _ in range(layers):
            x = vfn(x) + x
        return x

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dim, H, dh = 256, 8, 64
    rng = np.random.default_rng(0)
    results = {"T": args.T, "layers": args.layers, "rows": []}

    # rope kernel numeric check first (vs naive, small case inside every run)
    cos_c, sin_c, _ = make_cos_sin(64, dh)
    qkv_small = mx.array(
        (rng.standard_normal((4, 64, 3, H, dh)) * 0.5).astype(np.float16)
    )
    qn, kn = rope_naive(qkv_small, cos_c, sin_c, H, dh)
    qm, km = rope_metal(qkv_small, cos_c, sin_c, H, dh)
    mx.eval(qn, kn, qm, km)
    cq, ck = cos_sim(qm, qn), cos_sim(km, kn)
    print(f"[rope kernel check] cos q={cq:.7f} k={ck:.7f} {'OK' if min(cq, ck) > 0.9999 else 'FAIL'}")
    assert min(cq, ck) > 0.9999, "rope metal kernel mismatch"

    for T in ([args.T] if args.quick else sorted({64, 256, 512, args.T})):
        for (B, S, label) in [(60, T, "time-dir [60批,T序]"), (T, 60, "freq-dir [T批,60序]")]:
            w = make_weights(rng, dim, H, dh)
            cos, sin, ang = make_cos_sin(S, dh)
            x0 = mx.array((rng.standard_normal((B, S, dim)) * 0.5).astype(np.float16))

            variants = {
                "naive": lambda x, fn=attn_naive: fn(x, w, cos, sin, H, dh),
                "sdpa_fp16": lambda x, fn=attn_sdpa: fn(x, w, cos, sin, H, dh),
                "sdpa_fp32math": lambda x, fn=attn_sdpa: fn(x, w, cos, sin, H, dh, fp32_math=True),
                "opt_metal_rope": lambda x, fn=attn_opt: fn(x, w, cos, sin, H, dh),
            }

            # ---- parity vs fp64 (single layer, fresh input) ----
            ref = attn_ref_fp64(x0, w, ang, H, dh)
            parity = {}
            for name, vfn in variants.items():
                y = vfn(x0)
                mx.eval(y)
                parity[name] = cos_sim(y, ref)

            # ---- single-layer timing ----
            single = {name: bench(lambda vfn=vfn: vfn(x0), n=args.iters) for name, vfn in variants.items()}

            # ---- hybrid-split core timing (rope+SDPA+gate only, qkv precomputed) ----
            xn = mx.fast.rms_norm(x0, w["norm_gamma"], eps=1e-12)
            qkv0 = _proj_qkv(xn, w, B, S, H, dh)
            mx.eval(xn, qkv0)
            core = bench(lambda: attn_core_opt(qkv0, xn, w, cos, sin, H, dh), n=args.iters)
            q_r, k_r = rope_metal(qkv0, cos, sin, H, dh)
            v_r = qkv0[:, :, 2].transpose(0, 2, 1, 3)
            mx.eval(q_r, k_r, v_r)
            sdpa_only = bench(
                lambda: mx.fast.scaled_dot_product_attention(q_r, k_r, v_r, scale=dh ** -0.5),
                n=args.iters,
            )

            # ---- 12-layer chained pipeline (real dependency, residual included) ----
            pipe = {}
            for name, vfn in variants.items():
                run = pipeline(vfn, x0, layers=args.layers)
                pipe[name] = bench(run, n=max(8, args.iters // 3))
            row = dict(T=T, dir=label, B=B, S=S, parity=parity, single_ms=single,
                       core_hybrid_ms=core, sdpa_only_ms=sdpa_only, pipeline_ms=pipe)
            results["rows"].append(row)

            ps = "  ".join(f"{k}={v:.6f}" for k, v in parity.items())
            ss = "  ".join(f"{k}={v:.2f}" for k, v in single.items())
            pp = "  ".join(f"{k}={v:.1f}" for k, v in pipe.items())
            print(f"\n=== T={T} {label} (B={B},S={S},M={B*S}) ===")
            print(f"  parity cos : {ps}")
            print(f"  1-layer ms : {ss}")
            print(f"  core(hybrid split: rope+sdpa+gate) = {core:.2f}ms   sdpa-only = {sdpa_only:.2f}ms")
            print(f"  {args.layers}-layer chained ms: {pp}")
            print(f"  GPU budget: all-GPU(opt)={pipe['opt_metal_rope']:.0f}ms  hybrid-split(core)x{args.layers}={core*args.layers:.0f}ms")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nresults -> {args.json_out}")


if __name__ == "__main__":
    main()
