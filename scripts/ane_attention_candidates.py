"""Research-only attention variants; not selected by production build_graph.build.

Baseline semantics copied from build_graph.attn for controlled comparisons.
Head-packing is a measured negative result; kept here only for reproduction.
"""
from __future__ import annotations
import numpy as np
import aneforge as af
from build_graph import rms_last, big_linear, slast
F16 = np.float16


def attn(x: af.Tensor, w: dict, cfg: dict, rope, pfx: str) -> af.Tensor:
    dim, heads, dh = cfg["dim"], cfg["heads"], cfg["dim_head"]
    hd = heads * dh
    seq = x.shape[-2]
    B = x.shape[:-2]                     # e.g. (60,) for [B, seq, dim]
    cos_dup, sin_dup, P = rope
    xn = rms_last(x, w[f"{pfx}.norm.gamma"])
    qkv_w = w[f"{pfx}.to_qkv.weight"]
    halves = cfg.get("rope_half_layout", False)
    if halves:
        perm = np.concatenate([np.arange(0, dh, 2), np.arange(1, dh, 2)])
        qk_w = qkv_w[:2 * hd].reshape(2, heads, dh, dim)[:, :, perm, :].reshape(2 * hd, dim)
        qkv_w = np.concatenate([qk_w, qkv_w[2 * hd:]], axis=0)
    qkv = big_linear(xn, qkv_w)                           # [B, seq, 3hd]
    # [B, seq, heads, dh] views
    q = slast(qkv, 0, hd).reshape(B + (seq, heads, dh))
    k = slast(qkv, hd, 2 * hd).reshape(B + (seq, heads, dh))
    v = slast(qkv, 2 * hd, 3 * hd).reshape(B + (seq, heads, dh))
    # rope (interleaved-pair): out = x*cos + (x@P)*sin, cos/sin [1,seq,1,dh]
    cd = cos_dup[None, :, None, :].copy()
    sd = sin_dup[None, :, None, :].copy()
    if halves:
        c_half, s_half = cos_dup[:, ::2], sin_dup[:, ::2]
        cd = np.concatenate([c_half, c_half], axis=-1)[None, :, None, :].copy()
        sd = np.concatenate([s_half, s_half], axis=-1)[None, :, None, :].copy()
        def rotate_halves(t):
            return af.concat([slast(t, dh // 2, dh) * -1.0, slast(t, 0, dh // 2)], axis=-1)
        q_rot, k_rot = rotate_halves(q), rotate_halves(k)
    else:
        q_rot = q @ P
        k_rot = k @ P
    q = q * cd + q_rot * sd
    k = k * cd + k_rot * sd
    # -> [B, heads, seq, dh]: swap axes -3,-2
    rank = len(B) + 3
    ph = list(range(rank - 3)) + [rank - 2, rank - 3, rank - 1]
    q, k, v = q.transpose(ph), k.transpose(ph), v.transpose(ph)   # [B,h,S,dh]
    # Experimental opt-in: preserve each head's independent product while
    # exposing one batch axis to the compiler. Restore [B,h,S,dh] before ph.
    pack_heads = cfg.get("attn_pack_heads", False)
    if pack_heads == "freq":
        pack_heads = ".1.layers." in pfx
    if pack_heads:
        bh = int(np.prod(B)) * heads
        q, k, v = (t.reshape((bh, seq, dh)) for t in (q, k, v))
        rank = 3
    # fold 1/sqrt(dh) into q: chunk sims come out pre-scaled
    q = q * (dh ** -0.5)
    LOG2E = float(np.log2(np.e))

    # flash-style chunked softmax (exact), KEY-side chunking:
    # slice kt/v along the key axis; per-chunk (m,s,o) merge exactly.
    # (query-side slicing trips an Espresso layout-inference bug after merge)
    CH = cfg.get("attn_chunk", 100000)   # single-block by default: chunked merge loses fp16 accuracy at depth AND runs slower (no tile-resident scheduling in Espresso)
    n_ch = (seq + CH - 1) // CH
    ms, ss, os_ = [], [], []
    kt = k.transpose(list(range(rank - 2)) + [rank - 1, rank - 2])   # [B,h,dh,S]
    for c in range(n_ch):
        a = min(c * CH, seq)
        b = min((c + 1) * CH, seq)
        sz = b - a
        ktc = kt.slice_by_size([0] * (rank - 1) + [a], list(kt.shape[:-1]) + [sz])
        vc = v.slice_by_size([0] * (rank - 2) + [a, 0], list(v.shape[:-2]) + [sz, dh])
        sim = q @ ktc                                          # [B,h,S,sz]
        m = sim.amax(-1)                                      # [B,h,S,1]
        e = ((sim - m) * LOG2E).exp2()
        s = e @ np.ones((sz, 1), F16)   # row sum as matmul: fp16 narrow accumulator over K=512+ loses ~2e-3/layer
        o = e @ vc                                            # [B,h,S,dh]
        ms.append(m); ss.append(s); os_.append(o)
    if n_ch == 1:
        o, s = os_[0], ss[0]
    else:
        M = ms[0]
        for m_ in ms[1:]:
            M = af.maximum(M, m_)
        S = ss[0] * ((ms[0] - M) * LOG2E).exp2()
        O = os_[0] * ((ms[0] - M) * LOG2E).exp2()
        for i in range(1, n_ch):
            wc = ((ms[i] - M) * LOG2E).exp2()
            S = S + ss[i] * wc
            O = O + os_[i] * wc
        o, s = O, S
    o = o / s                                                  # [B,h,S,dh] or [B*h,S,dh]
    if pack_heads:
        o = o.reshape(B + (heads, seq, dh))
    o = o.transpose(ph).reshape(B + (seq, hd))
    gates = big_linear(xn, w[f"{pfx}.to_gates.weight"], bias=w[f"{pfx}.to_gates.bias"]).sigmoid()
    o = o.reshape(B + (seq, heads, dh)) * gates.reshape(B + (seq, heads, 1))
    return big_linear(o.reshape(B + (seq, hd)), w[f"{pfx}.to_out.0.weight"])
