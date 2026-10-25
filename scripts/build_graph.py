"""mel-band-roformer-deux (becruily) -> ANEForge: graph + host pipeline.

Layout contracts (HOST <-> ANE):
  ANE input  x: [1, T, L]  L=7916, group-major band layout (see below), fp16
  ANE output : [S, 60, T, L] masks (stems-major), fp16

  L layout = concat over dim-groups of (bands in group × dim_in each):
    group g covers bands [s_g, e_g), each band contributes dim_in slots,
    ordered band-major inside the group. Same order as torch's
    _forward_grouped_mlp cat(dim=-1) — matches band_split input layout too
    (band_split output order is band-major via cat(dim=-2) over groups).
    INPUT side band_split reads x[..., offset_s:offset_e].reshape(G, d) — so
    input L layout is also group-major. freq_indices gather must produce this.

  torch mask output [b, S, t, L_group_major]; scatter uses freq_indices
  aligned to the same L order. We keep identical order everywhere.

Numerics (fp16): bandsplit gamma includes sqrt(d); transformer RMSNorm
eps=1e-12 with fp32 gamma bake; rope interleaved-pair via baked cos/sin/P.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import aneforge as af

F16, F32 = np.float16, np.float32


def load_cfg(path: str) -> dict:
    import yaml
    class Ldr(yaml.SafeLoader):
        pass
    Ldr.add_constructor('tag:yaml.org,2002:python/tuple',
                        lambda l, n: tuple(l.construct_sequence(n)))
    with open(path) as f:
        y = yaml.load(f, Loader=Ldr)
    m, a = y["model"], y["audio"]
    return dict(
        dim=m["dim"], depth=m["depth"],
        time_depth=m.get("time_transformer_depth", 1),
        freq_depth=m.get("freq_transformer_depth", 1),
        num_bands=m.get("num_bands", 60),
        dim_head=m.get("dim_head", 64), heads=m.get("heads", 8),
        num_stems=len(y["training"]["instruments"]),
        audio_channels=2 if m.get("stereo", True) else 1,
        stft_n_fft=m.get("stft_n_fft", a.get("n_fft", 2048)),
        stft_hop=m.get("stft_hop_length", a.get("hop_length", 441)),
        sample_rate=m.get("sample_rate", a.get("sample_rate", 44100)),
        dim_freqs_in=m.get("dim_freqs_in", a.get("dim_f", 1024) + 1),
    )


def mel_layout(cfg: dict):
    """Returns (band_dims [60], freq_indices_flat [L] (group-major, stereo),
    num_bands_per_channel_freq [1,1,F*C,1] divisor)."""
    import librosa
    mb = librosa.filters.mel(sr=cfg["sample_rate"], n_fft=cfg["stft_n_fft"],
                             n_mels=cfg["num_bands"])
    mb[0][0], mb[-1][-1] = 1.0, 1.0          # roformer corner fill
    fpb = mb > 0                              # [60, F]
    C = cfg["audio_channels"]
    # band b covers freq bins fpb[b] -> freqs list; band dim = 2*count*C
    dims = [2 * int(fpb[b].sum()) * C for b in range(cfg["num_bands"])]
    # freq_indices per band (freq-major), stereo-interleaved then flattened:
    # torch: fi = arange(F).expand(60,-1)[fpb]; stereo: (fi[:,None]*2 + [0,1]).flatten()
    fi_bands = [np.arange(cfg["dim_freqs_in"])[fpb[b]] for b in range(cfg["num_bands"])]
    freq_indices = []
    for b, fi in enumerate(fi_bands):
        if C == 2:
            fi = (fi[:, None] * 2 + np.arange(2)).reshape(-1)
        freq_indices.append(fi)
    # group-major flatten
    groups = dim_groups(dims)
    flat = []
    for (s, e, d) in groups:
        for b in range(s, e):
            flat.append(freq_indices[b])
    fi_flat = np.concatenate(flat)            # [L]
    # divisor: num_bands_per_freq repeat_interleave(C) — counts covering bands
    nbpf = fpb.sum(axis=0)                    # [F]
    nbpf_c = np.repeat(nbpf, C)               # [F*C]
    return dims, fi_flat, nbpf_c


def dim_groups(dims: list[int]) -> list[tuple[int, int, int]]:
    out, start = [], 0
    for i in range(1, len(dims) + 1):
        if i == len(dims) or dims[i] != dims[start]:
            out.append((start, i, dims[start]))
            start = i
    return out


def group_band_order(dims: list[int]) -> list[int]:
    """Band indices in group-major order (input L layout order)."""
    order = []
    for (s, e, d) in dim_groups(dims):
        order += list(range(s, e))
    return order


# ------------------------------------------------------------- primitives --
def rms_last(x: af.Tensor, gamma_f32: np.ndarray | None, eps: float = 1e-12):
    """RMSNorm over last dim, gamma [D] fp32 (or None)."""
    shp = x.shape
    D = shp[-1]
    x2 = x.reshape((int(np.prod(shp[:-1])), D))
    g = np.ones(D, F32) if gamma_f32 is None else np.asarray(gamma_f32, F32)
    x2 = x2.rms_norm(g, eps=eps)
    return x2.reshape(shp)


def big_linear(x: af.Tensor, W, bias=None) -> af.Tensor:
    """Flatten-to-2D linear as ONE big GEMM (matmul path).

    (A conv1x1 variant was tried: it needs [1,K,1,M] channel-first layout, but
    reshape alone reinterprets [M,K] row-major memory as [K,M] — scrambling
    the layout — and the two extra transposes cost more than conv saves.)
    """
    shp = x.shape
    x2 = x.reshape((int(np.prod(shp[:-1])), shp[-1]))
    x2 = x2.linear(W, bias=bias)
    return x2.reshape(shp[:-1] + (x2.shape[-1],))


def slast(t: af.Tensor, a: int, b: int) -> af.Tensor:
    beg = [0] * len(t.shape); beg[-1] = a
    sz = list(t.shape); sz[-1] = b - a
    return t.slice_by_size(beg, sz)


def rope_bakes(freqs: np.ndarray, seq: int):
    dh = 2 * freqs.shape[0]
    ang = np.arange(seq, dtype=F32)[:, None] * freqs[None, :].astype(F32)
    cos_dup = np.repeat(np.cos(ang), 2, axis=1).astype(F16)     # [seq, dh]
    sin_dup = np.repeat(np.sin(ang), 2, axis=1).astype(F16)
    P = np.zeros((dh, dh), F16)
    for i in range(0, dh, 2):
        P[i + 1, i] = -1.0
        P[i, i + 1] = 1.0
    return cos_dup, sin_dup, P


def attn(x: af.Tensor, w: dict, cfg: dict, rope, pfx: str) -> af.Tensor:
    dim, heads, dh = cfg["dim"], cfg["heads"], cfg["dim_head"]
    hd = heads * dh
    seq = x.shape[-2]
    B = x.shape[:-2]                     # e.g. (60,) for [B, seq, dim]
    cos_dup, sin_dup, P = rope
    xn = rms_last(x, w[f"{pfx}.norm.gamma"])
    qkv = big_linear(xn, w[f"{pfx}.to_qkv.weight"])          # [B, seq, 3hd]
    # [B, seq, heads, dh] views
    q = slast(qkv, 0, hd).reshape(B + (seq, heads, dh))
    k = slast(qkv, hd, 2 * hd).reshape(B + (seq, heads, dh))
    v = slast(qkv, 2 * hd, 3 * hd).reshape(B + (seq, heads, dh))
    # rope (interleaved-pair): out = x*cos + (x@P)*sin, cos/sin [1,seq,1,dh]
    cd = cos_dup[None, :, None, :].copy()
    sd = sin_dup[None, :, None, :].copy()
    q_rot = q @ P                     # 4D const-matmul (flatten would exceed 65536 axis cap)
    k_rot = k @ P
    q = q * cd + q_rot * sd
    k = k * cd + k_rot * sd
    # -> [B, heads, seq, dh]: swap axes -3,-2
    rank = len(B) + 3
    ph = list(range(rank - 3)) + [rank - 2, rank - 3, rank - 1]
    q, k, v = q.transpose(ph), k.transpose(ph), v.transpose(ph)   # [B,h,S,dh]
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
        s = e.sum(-1)
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
    o = o / s                                                  # [B,h,S,dh]
    o = o.transpose(ph).reshape(B + (seq, hd))
    gates = big_linear(xn, w[f"{pfx}.to_gates.weight"], bias=w[f"{pfx}.to_gates.bias"]).sigmoid()
    o = o.reshape(B + (seq, heads, dh)) * gates.reshape(B + (seq, heads, 1))
    return big_linear(o.reshape(B + (seq, hd)), w[f"{pfx}.to_out.0.weight"])


def ffn(x: af.Tensor, w: dict, pfx: str) -> af.Tensor:
    h = rms_last(x, w[f"{pfx}.net.0.gamma"])
    h = big_linear(h, w[f"{pfx}.net.1.weight"], bias=w[f"{pfx}.net.1.bias"]).gelu()
    return big_linear(h, w[f"{pfx}.net.4.weight"], bias=w[f"{pfx}.net.4.bias"])


def transformer(x: af.Tensor, w: dict, cfg: dict, rope, pfx: str, depth: int) -> af.Tensor:
    for i in range(depth):
        x = attn(x, w, cfg, rope, f"{pfx}.layers.{i}.0") + x
        x = ffn(x, w, f"{pfx}.layers.{i}.1") + x
    return rms_last(x, w[f"{pfx}.norm.gamma"])


# --------------------------------------------------------------- blocks ----
_MI_T: list[int] = []   # bandsplit multi-input mode: T passed via this side channel


def bandsplit(x: af.Tensor | None, w: dict, cfg: dict, dims: list[int],
              multi_in: bool = False) -> af.Tensor:
  """x: [1,T,L] single input (T<=384), or multi_in=True with x=None using per-group
  af.input() tensors (T>=385 workaround: one input sliced by >=13 consumers trips an
  Espresso compile wall at T>=385; 13+ independent inputs compile fine at any T)."""
  dim = cfg["dim"]
  T = x.shape[1] if x is not None else _MI_T[0]
  outs = []
  for (s, e, d) in dim_groups(dims):
    if multi_in:
      xs = af.input((1, T, e - s, d), max_abs=16.0)
    else:
      off = sum(dims[:s])
      xs = x.slice_by_size([0, 0, off], [1, T, (e - s) * d]).reshape((1, T, e - s, d))
    # per-band gamma folds into W's input axis: (x_norm*gamma)@W.T == x_norm@(W*gamma).T
    gamma = np.stack([np.asarray(w[f"band_split.to_features.{i}.0.gamma"], F32)
                      for i in range(s, e)])[:, None, :] * (d ** 0.5)  # [G,1,D]
    shp = xs.shape
    x2 = xs.reshape((int(np.prod(shp[:-1])), d)).rms_norm(np.ones(d, F32), eps=1e-12).reshape(shp)
    x2t = x2.transpose([2, 0, 1, 3]).reshape((e - s, T, d))   # [G, T, d] normalized
    W = np.stack([w[f"band_split.to_features.{i}.1.weight"] for i in range(s, e)])  # [G,dim,d]
    B = np.stack([w[f"band_split.to_features.{i}.1.bias"] for i in range(s, e)])
    W_scaled = (W.astype(F32) * gamma).astype(F16)
    xg = x2t                                                        # [G, T, d]
    y = xg.bmm_weight(np.ascontiguousarray(W_scaled.transpose(0, 2, 1), F16))
    y = y + B.astype(F16).reshape(e - s, 1, dim)
    # y [G,T,dim] -> [1,T,G,dim]
    outs.append(y.transpose([1, 0, 2]).reshape((T, e - s, dim)).expand_dims(0))
  return af.concat(outs, axis=2)          # [1,T,60,dim]


def mask_heads(x: af.Tensor, w: dict, cfg: dict, dims: list[int]) -> af.Tensor:
    """[1,T,60,dim] -> [S, 60, T, L] masks, band-major inside groups (matches torch)."""
    T = x.shape[1]
    dim, S = cfg["dim"], cfg["num_stems"]
    G = 60 * S
    xs = af.concat([x] * S, axis=2)                              # [1,T,60S,dim]
    W0 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.0.weight"]
                   for s in range(S) for b in range(60)]).transpose(0, 2, 1)
    B0 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.0.bias"]
                   for s in range(S) for b in range(60)])
    W2 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.2.weight"]
                   for s in range(S) for b in range(60)]).transpose(0, 2, 1)
    B2 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.2.bias"]
                   for s in range(S) for b in range(60)])
    h = xs.transpose([2, 0, 1, 3]).reshape((G, T, dim))
    h = h.bmm_weight(W0.astype(F16)) + B0.astype(F16).reshape(G, 1, 1024)
    h = h.tanh()
    h = h.bmm_weight(W2.astype(F16)) + B2.astype(F16).reshape(G, 1, 1024)
    h = h.tanh()                                                 # [G, T, 1024]
    dims_all: list[int] = []
    for s in range(S):
        dims_all += dims
    outs = []
    for (s0, e0, d_in) in dim_groups(dims_all):
        g = e0 - s0
        W4 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.4.weight"]
                       for s in range(S) for b in range(60)
                       if s0 <= s * 60 + b < e0]).transpose(0, 2, 1)
        B4 = np.stack([w[f"mask_estimators.{s}.to_freqs.{b}.0.4.bias"]
                       for s in range(S) for b in range(60)
                       if s0 <= s * 60 + b < e0])
        hs = h.slice_by_size([s0, 0, 0], [g, T, 1024])
        y = hs.bmm_weight(W4.astype(F16)) + B4.astype(F16).reshape(g, 1, 2 * d_in)
        glu = slast(y, 0, d_in) * slast(y, d_in, 2 * d_in).sigmoid()
        outs.append(glu)                                         # [g, T, d_in]
    # concat last dim impossible (group dims differ) -> concat along axis 0 as
    # [S*60, T, group_pieces]: we need ONE output. Emit group-major L like
    # torch: cat(dim=-1) over groups of flatten(G*d_in) — shapes differ on
    # axis0 too. SOLUTION: reshape each [g,T,din] -> [1,T,g*din] and concat
    # on axis 2 band axis? No — final L order must be group-major on last dim.
    # Use [g,T,din] -> transpose -> [T, g*din] pieces, concat axis=-1.
    pieces = [p.transpose([1, 0, 2]).reshape((T, p.shape[0] * p.shape[2])) for p in outs]  # [T, g*din]
    out = af.concat(pieces, axis=-1)                             # [T, L]
    # [T, L] -> [S, 60, T, L]: rows of G=S*60 interleaved in pieces. Host reshapes.
    return out


def build(T: int, cfg: dict, w: dict, dims: list[int]) -> af.Tensor:
    dim = cfg["dim"]
    L = sum(dims)
    # T>=385: single input sliced by 39 groups trips an Espresso compile wall;
    # use per-group independent inputs instead (host pre-slices along L).
    if T >= 385:
        _MI_T[:] = [T]
        x = bandsplit(None, w, cfg, dims, multi_in=True)
    else:
        x = af.input((1, T, L), max_abs=16.0)
        x = bandsplit(x, w, cfg, dims)
    t_rope = rope_bakes(w["layers.0.0.layers.0.0.rotary_embed.freqs"], T)
    f_rope = rope_bakes(w["layers.0.1.layers.0.0.rotary_embed.freqs"], 60)
    for d in range(cfg["depth"]):
        xt = x.transpose([0, 2, 1, 3]).reshape((60, T, dim))
        xt = transformer(xt, w, cfg, t_rope, f"layers.{d}.0", cfg["time_depth"])
        x = xt.reshape((1, 60, T, dim)).transpose([0, 2, 1, 3])
        xf = x.reshape((T, 60, dim))
        xf = transformer(xf, w, cfg, f_rope, f"layers.{d}.1", cfg["freq_depth"])
        x = xf.reshape((1, T, 60, dim))
    return mask_heads(x, w, cfg, dims)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--config",
                    default="/Volumes/nfs/data/Downloads/mel-band-roformer-deux/config_deux_becruily.yaml")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--build-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    wdir = Path(args.weights)
    manifest = json.loads((wdir / "manifest.json").read_text())
    w = {k: np.load(wdir / v) for k, v in manifest.items()}
    dims = [w[f"band_split.to_features.{i}.1.weight"].shape[1] for i in range(60)]
    print(f"dim={cfg['dim']} depth={cfg['depth']} L={sum(dims)} T={args.T} stems={cfg['num_stems']}")

    t0 = time.time()
    out = build(args.T, cfg, w, dims)
    net = af.compile(out, build_dir=args.build_dir)
    print(f"compiled {time.time()-t0:.1f}s, {net.n_ops} ops")

    L = sum(dims)
    xin = (np.random.randn(1, args.T, L) * 0.5).astype(F16)
    if args.T >= 385:
        # multi-input bandsplit: per-group slices along L
        feeds, off = [], 0
        for (s, e, d) in dim_groups(dims):
            w_g = (e - s) * d
            feeds.append(xin[:, :, off:off + w_g].reshape(1, args.T, e - s, d))
            off += w_g
        y = net(*feeds)
    else:
        y = net(xin)
    print("out", y.shape, "std", float(y.std()))


if __name__ == "__main__":
    main()
