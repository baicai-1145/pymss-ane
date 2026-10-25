"""Parity check: ANEForge graph vs numpy reference (pure fp32/fp64 host sim).

Reference implements the torch forward exactly:
  bandsplit -> 12x(time transformer, freq transformer) -> mask estimators
using the same weight files. Compares the ANE fp16 program output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_graph as bg


def ref_bandsplit(x, w, dims):
    """x [1,T,L] fp32 -> [1,T,60,dim]. gamma-folded like the torch fast path."""
    outs = []
    for b in range(60):
        d = dims[b]
        xs = x[:, :, sum(dims[:b]):sum(dims[:b + 1])]     # [1,T,d]
        g = w[f"band_split.to_features.{b}.0.gamma"].astype(np.float64) * (d ** 0.5)
        xn = xs / np.sqrt((xs ** 2).mean(-1, keepdims=True) + 1e-12)
        W = w[f"band_split.to_features.{b}.1.weight"].astype(np.float64)   # [dim, d]
        B = w[f"band_split.to_features.{b}.1.bias"].astype(np.float64)
        outs.append(xn * g @ W.T + B)                      # [1,T,dim]
    return np.stack(outs, axis=2)                          # [1,T,60,dim]


def ref_rope(freqs, seq):
    ang = np.arange(seq)[:, None] * freqs[None, :]
    return np.cos(ang), np.sin(ang)                        # [seq, dh/2]


def ref_apply_rope(x, cos, sin):
    """x [..., seq, heads, dh] interleaved pairs."""
    e, o = x[..., 0::2], x[..., 1::2]                      # [..., seq, h, dh/2]
    out = np.empty_like(x)
    out[..., 0::2] = e * cos[..., None, :] - o * cos[..., None, :] * 0 - o * sin[..., None, :] + e * 0
    # careful: even' = e*cos - o*sin ; odd' = o*cos + e*sin
    out[..., 0::2] = e * cos[..., None, :] - o * sin[..., None, :]
    out[..., 1::2] = o * cos[..., None, :] + e * sin[..., None, :]
    return out


def ref_attn(x, w, cfg, freqs, pfx):
    dim, heads, dh = cfg["dim"], cfg["heads"], cfg["dim_head"]
    hd = heads * dh
    seq = x.shape[-2]
    g = w[f"{pfx}.norm.gamma"]
    xn = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-12) * g
    qkv = xn @ w[f"{pfx}.to_qkv.weight"].T                 # [..., seq, 3hd]
    q = qkv[..., :hd].reshape(*qkv.shape[:-1], heads, dh)
    k = qkv[..., hd:2 * hd].reshape(*qkv.shape[:-1], heads, dh)
    v = qkv[..., 2 * hd:].reshape(*qkv.shape[:-1], heads, dh)
    cos, sin = ref_rope(freqs, seq)
    q, k = ref_apply_rope(q, cos, sin), ref_apply_rope(k, cos, sin)
    # [.., seq, heads, dh] -> [.., heads, seq, dh]
    q, k, v = [np.swapaxes(t, -3, -2) for t in (q, k, v)]
    sim = q @ np.swapaxes(k, -1, -2)
    sim = sim * (dh ** -0.5)
    sim = sim - sim.max(-1, keepdims=True)
    p = np.exp(sim); p /= p.sum(-1, keepdims=True)
    o = p @ v                                             # [.., heads, seq, dh]
    # gates: [.., seq, heads]
    gates = 1 / (1 + np.exp(-(xn @ w[f"{pfx}.to_gates.weight"].T + w[f"{pfx}.to_gates.bias"])))
    o = np.swapaxes(o, -2, -3)                             # [.., seq, heads, dh]
    o = o * gates[..., None]                               # broadcast over dh
    o = o.reshape(list(x.shape[:-1]) + [hd])               # [..., seq, hd]
    return o @ w[f"{pfx}.to_out.0.weight"].T


def ref_ffn(x, w, pfx):
    g = w[f"{pfx}.net.0.gamma"]
    h = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-12) * g
    h = h @ w[f"{pfx}.net.1.weight"].T + w[f"{pfx}.net.1.bias"]
    # gelu exact
    h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
    return h @ w[f"{pfx}.net.4.weight"].T + w[f"{pfx}.net.4.bias"]


def ref_transformer(x, w, cfg, freqs, pfx, depth):
    for i in range(depth):
        x = ref_attn(x, w, cfg, freqs, f"{pfx}.layers.{i}.0") + x
        x = ref_ffn(x, w, f"{pfx}.layers.{i}.1") + x
    g = w[f"{pfx}.norm.gamma"]
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-12) * g


def ref_mask_heads(x, w, cfg, dims):
    """x [1,T,60,dim] -> [S, T, L] (group-major L, same as ANE output layout)."""
    S = cfg["num_stems"]
    outs = []
    for s in range(S):
        pieces = []
        for (s0, e0, d_in) in bg.dim_groups(dims):
            band_outs = []
            for b in range(s0, e0):
                h = x[:, :, b, :]                          # [1,T,dim]
                h = np.tanh(h @ w[f"mask_estimators.{s}.to_freqs.{b}.0.0.weight"].T
                            + w[f"mask_estimators.{s}.to_freqs.{b}.0.0.bias"])
                h = np.tanh(h @ w[f"mask_estimators.{s}.to_freqs.{b}.0.2.weight"].T
                            + w[f"mask_estimators.{s}.to_freqs.{b}.0.2.bias"])
                y = h @ w[f"mask_estimators.{s}.to_freqs.{b}.0.4.weight"].T \
                    + w[f"mask_estimators.{s}.to_freqs.{b}.0.4.bias"]     # [1,T,2*d_in]
                a, bb = y[..., :d_in], y[..., d_in:]
                band_outs.append(a * (1 / (1 + np.exp(-bb))))             # GLU
            pieces.append(np.concatenate(band_outs, axis=-1))  # [1,T,g*din]
        outs.append(np.concatenate(pieces, axis=-1))          # [1,T,L]
    return np.stack(outs, axis=0)                             # [S,T,L]


def ref_forward(x, w, cfg, dims):
    x = ref_bandsplit(x, w, dims)
    T, F_ = x.shape[1], 60
    dim = cfg["dim"]
    t_freqs = w["layers.0.0.layers.0.0.rotary_embed.freqs"]
    f_freqs = w["layers.0.1.layers.0.0.rotary_embed.freqs"]
    for d in range(cfg["depth"]):
        xt = x.transpose(0, 2, 1, 3).reshape(F_, T, dim)
        xt = ref_transformer(xt, w, cfg, t_freqs, f"layers.{d}.0", cfg["time_depth"])
        x = xt.reshape(1, F_, T, dim).transpose(0, 2, 1, 3)
        xf = x.reshape(T, F_, dim)
        xf = ref_transformer(xf, w, cfg, f_freqs, f"layers.{d}.1", cfg["freq_depth"])
        x = xf.reshape(1, T, F_, dim)
    return ref_mask_heads(x, w, cfg, dims)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--depth", type=int, default=None)
    args = ap.parse_args()

    cfg_full = bg.load_cfg("/Volumes/nfs/data/Downloads/mel-band-roformer-deux/config_deux_becruily.yaml")
    wdir = Path("weights")
    manifest = json.loads((wdir / "manifest.json").read_text())
    w = {k: np.load(wdir / v) for k, v in manifest.items()}
    dims = [w[f"band_split.to_features.{i}.1.weight"].shape[1] for i in range(60)]

    cfg = dict(cfg_full)
    if args.depth:
        cfg["depth"] = args.depth

    import aneforge as af
    rng = np.random.default_rng(42)
    L = sum(dims)
    xin = rng.standard_normal((1, args.T, L)).astype(np.float32) * 0.5

    # reference (fp64 via upcast)
    w64 = {k: v.astype(np.float64) for k, v in w.items() if v.dtype == np.float16}
    ref = ref_forward(xin.astype(np.float64), w64, cfg, dims)      # [S,T,L]

    # ANE
    out = bg.build(args.T, cfg, w, dims)
    out = af.reduce_sum_to_matmul(out)   # narrow fp16 reduce_sum -> lossless matmul
    net = af.compile(out)
    y = net(xin.astype(np.float16))                                  # [T, L*S?]

    print("ANE out shape:", y.shape, " ref shape:", ref.shape)
    # ANE layout: pieces concat [T, L] per group, groups band-major, stems?? — check mask_heads
    # mask_heads concat S copies on band axis → h rows = [stem0 groups.., stem1 groups..]
    # final pieces transpose → [T, g*din]; concat axis=-1 → [T, L_g0+L_g1+...]
    # rows h: S*60 in order (stem0 band0..59, stem1 band0..59) but groups are dim-grouped
    # Actually h = [G=S*60, T, hidden] with G ordered stem-major THEN band-major,
    # then per dim-group slice [s0:e0] which cuts across the same dim in both stems.
    # pieces p[g_idx, T, d_in] → transpose [T, g, d_in] reshape [T, g*d_in]
    # concat over groups → [T, L_groupmajor_sameorderasref]
    # ref: [S, T, L] → reshape to compare
    S = cfg["num_stems"]
    # ANE y: [T, L] but which stem? both stems' groups interleaved by dim-group:
    # group (s0,e0,d) covers bands s0..e0 for BOTH stems (stem-major rows)
    # ref L layout: concat over groups of (bands×d_in) per stem, stems separate.
    # → reshape ref to [T, L] per stem and compare against y slices per group.
    # Simpler: rebuild y into [S, T, L] using the same group-major order:
    G_layout = []  # list of (stem, band_start, band_end, d_in)
    dims_all = []
    for s in range(S):
        dims_all += dims
    y_parts = []
    ptr = 0
    for (s0, e0, d_in) in bg.dim_groups(dims_all):
        y_parts.append(y[:, ptr:ptr + (e0 - s0) * d_in])
        ptr += (e0 - s0) * d_in
    assert ptr == y.shape[1], f"layout mismatch {ptr} vs {y.shape[1]}"
    # ref per stem pieces
    ref_parts = {s: [] for s in range(S)}
    ptr = 0
    for (s0, e0, d_in) in bg.dim_groups(dims):
        for s in range(S):
            ref_parts[s].append(ref.reshape(S, args.T, -1)[s, :, ptr:ptr + (e0 - s0) * d_in])
        ptr += (e0 - s0) * d_in
    ok = True
    for gi, yp in enumerate(y_parts):
        s = gi % S if False else None  # groups over dims_all are stem-major contiguous
    # dims_all groups: for each stem-major dim-group covering both stems? No —
    # dims_all = dims+dims, dim_groups cuts where dim changes; stems share dims
    # so groups span across the stem boundary identically.
    # y_parts[gi] covers rows s0..e0 of G where G = stem*60+band.
    # ref order: stem0 group g, stem1 group g, stem0 group g+1...
    # But y groups are ordered by dim value, spanning both stems.
    # Rebuild ref in EXACTLY y's group order:
    # ref is [S, T, L_ref] where L_ref = concat over dim_groups(dims) of (bands×d_in)
    # band b's slot in ref[s]: offset = sum over PRECEDING bands in ref order (band-major)
    # = band_offsets[b] = sum(dims[:b]) — because ref concatenates groups in band order.
    L_full = S * ref.shape[-1]
    ref_y = np.zeros((args.T, L_full), dtype=np.float64)
    ref2 = ref.reshape(cfg["num_stems"], args.T, ref.shape[-1])  # [S,T,L]
    ptr = 0
    for (s0, e0, d_in) in bg.dim_groups(dims_all):
        for grow in range(s0, e0):
            s, b = grow // 60, grow % 60
            b_off = sum(dims[:b])
            ref_y[:, ptr:ptr + d_in] = ref2[s, :, b_off:b_off + d_in]
            ptr += d_in
    assert ptr == L_full == y.shape[1], f"{ptr} vs {y.shape[1]}"

    cos = []
    for s_i in range(S):
        a = ref_y.flatten()
        b_ = y.flatten().astype(np.float32)
    a = ref_y.flatten()
    b_ = y.flatten().astype(np.float64)
    cos_sim = (a @ b_) / (np.linalg.norm(a) * np.linalg.norm(b_))
    rel = np.abs(a - b_) / (np.abs(a) + 1e-3)
    print(f"T={args.T} depth={cfg['depth']}: cosine={cos_sim:.6f} "
          f"rel_err_med={np.median(rel):.4f} p99={np.quantile(rel,0.99):.4f}")


if __name__ == "__main__":
    main()
