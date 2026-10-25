"""Hybrid ANE<->GPU runway probe (form-A): feasibility verdict with measured segments.

Design (see .pi/herdr-team/7si1w8/reports/task2-hybrid-runway-probe.md):
  P1 (ANE): bandsplit [1,T,L] -> [1,T,60,dim]
  GPU (MLX): 12x time-attention (rmsnorm+qkv+rope+sdpa+gates+out) at [60,T,dim]
  PM (ANE): per-depth time-ffn + freq transformer (attn+ffn, all-ANE, 0 cut/layer)
  P2 (ANE): mask estimator heads -> [T, S*L]
Segments run in one process (ANE venv, mlx 0.32.4 also installed); bridges are
np fp16 <-> mx.array (unified memory) + ANE input_view/output_view zero-copy.

Usage:
  .venv/bin/python scripts/hybrid_probe.py --mode build   # compile P1/PM/P2 (slow)
  .venv/bin/python scripts/hybrid_probe.py --mode parity  # T=64 depth=2 cos check
  .venv/bin/python scripts/hybrid_probe.py --mode bench   # T=256 3-way timing
  .venv/bin/python scripts/hybrid_probe.py --mode all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_graph as bg

F16, F32 = np.float16, np.float32
HERE = Path(__file__).parent
ROOT = HERE.parent
MLX_VENV_PY = Path("/Volumes/2T/repos/mss/pymss-core/.venv/bin/python")  # mlx 0.31.2 (unused; same-proc mlx below)
CACHE = ROOT / ".tmp" / "hybrid_cache"
CFG_PATH = "/Volumes/nfs/data/Downloads/mel-band-roformer-deux/config_deux_becruily.yaml"


def load_all():
    cfg = bg.load_cfg(CFG_PATH)
    wdir = ROOT / "weights"
    manifest = json.loads((wdir / "manifest.json").read_text())
    w = {k: np.load(wdir / v) for k, v in manifest.items()}
    dims = [w[f"band_split.to_features.{i}.1.weight"].shape[1] for i in range(60)]
    return cfg, w, dims


# --------------------------------------------------------------------------
# MIL segment builders (aneforge) — reuse build_graph primitives verbatim
# --------------------------------------------------------------------------

def build_p1(T, cfg, w, dims):
    """ANE program 1: bandsplit. [1,T,L] -> [1,T,60,dim]. T>=385 uses multi-input."""
    if T >= 385:
        bg._MI_T[:] = [T]
        return bg.bandsplit(None, w, cfg, dims, multi_in=True)
    x = af_input_l(T, sum(dims))
    return bg.bandsplit(x, w, cfg, dims)   # [1,T,60,dim]


def build_pm(T, cfg, w, dims, depth_idx):
    """ANE program mid: input [1,T,60,dim] = GPU attention output (x + attn(x)).
    Runs time-FFN (time_depth layers, attn already on GPU) + full freq transformer
    (freq_depth, all-ANE 0 cut/layer) -> [1,T,60,dim] for next depth / mask heads."""
    dim = cfg["dim"]
    x = af_input_t(T, dim)          # [1,T,60,dim] fp16
    # time-ffn part of layer d (attn already done on GPU)
    xt = x.transpose([0, 2, 1, 3]).reshape((60, T, dim))   # [60,T,dim]
    for i in range(cfg["time_depth"]):
        xt = bg.ffn(xt, w, f"layers.{depth_idx}.0.layers.{i}.1") + xt
    xt = bg.rms_last(xt, w[f"layers.{depth_idx}.0.norm.gamma"])
    x = xt.reshape((1, 60, T, dim)).transpose([0, 2, 1, 3])
    # freq transformer (all ANE)
    f_rope = bg.rope_bakes(w["layers.0.1.layers.0.0.rotary_embed.freqs"], 60)
    xf = x.reshape((T, 60, dim))
    xf = bg.transformer(xf, w, cfg, f_rope, f"layers.{depth_idx}.1", cfg["freq_depth"])
    return xf.reshape((1, T, 60, dim))


def af_input_l(T, L):
    import aneforge as af
    return af.input((1, T, L), max_abs=16.0)


def af_input_t(T, dim):
    import aneforge as af
    return af.input((1, T, 60, dim), max_abs=16.0)


def build_p2(T, cfg, w, dims):
    """ANE program 2: mask estimator. [1,T,60,dim] -> [T, S*L]."""
    x = af_input_t(T, cfg["dim"])
    return bg.mask_heads(x, w, cfg, dims)


# --------------------------------------------------------------------------
# MLX GPU side: time-attention segment (12x, one per depth)
# --------------------------------------------------------------------------

class MLXAttention:
    """Same-process MLX time-attention: x -> x + attn(x) at [60,T,dim] fp16."""

    def __init__(self, w, cfg, T, mlx_python=None):
        import mlx.core as mx
        self.mx = mx
        dim, heads, dh = cfg["dim"], cfg["heads"], cfg["dim_head"]
        self.dim, self.heads, self.dh = dim, heads, dh
        self.hd = heads * dh
        self.T = T
        t_freqs = w["layers.0.0.layers.0.0.rotary_embed.freqs"]
        ang = np.arange(T, dtype=F32)[:, None] * t_freqs[None, :].astype(F32)
        cos = np.repeat(np.cos(ang), 2, axis=1).astype(F16)   # [T, dh]
        sin = np.repeat(np.sin(ang), 2, axis=1).astype(F16)
        # weights: per-depth norm gamma, qkv, gates(+b), out
        self.pfx = [f"layers.{d}.0.layers.0.0" for d in range(cfg["depth"])]
        self.W = {}
        for d, p in enumerate(self.pfx):
            self.W[d] = dict(
                gamma=mx.array(w[f"{p}.norm.gamma"].astype(F32)),
                qkv=mx.array(w[f"{p}.to_qkv.weight"].astype(F16)),
                gates_w=mx.array(w[f"{p}.to_gates.weight"].astype(F16)),
                gates_b=mx.array(w[f"{p}.to_gates.bias"].astype(F16)),
                out=mx.array(w[f"{p}.to_out.0.weight"].astype(F16)),
            )
        self.cos = mx.array(cos[None, :, None, :])   # [1,T,1,dh] = [1,T,1,64]
        self.sin = mx.array(sin[None, :, None, :])
        self.scale = dh ** -0.5

    def _rope(self, x):
        mx = self.mx
        e, o = x[..., 0::2], x[..., 1::2]
        out = mx.zeros_like(x)
        out = mx.concatenate([e * self.cos - o * self.sin,
                              o * self.cos + e * self.sin], axis=-1)
        # interleave back
        B, S, H, D = x.shape
        out = out.reshape(B, S, H, D // 2, 2).transpose(0, 1, 2, 4, 3).reshape(B, S, H, D)
        return out

    def _rope_fast(self, x):
        # interleaved-pair rope; x [B,S,H,64] -> same. cos/sin [1,T,1,64] indexed at pair slots
        mx = self.mx
        c, s = self.cos[..., 0::2], self.sin[..., 0::2]     # [1,T,1,32] per-pair
        e, o = x[..., 0::2], x[..., 1::2]                   # [B,S,H,32]
        ec, oc = e * c, o * c
        es, os_ = e * s, o * s
        out = mx.stack([ec - os_, oc + es], axis=-1)   # [B,S,H,32,2]
        return out.reshape(x.shape)

    def attn(self, x, d):
        mx = self.mx
        Wd = self.W[d]
        B, S, _ = x.shape
        xn = mx.fast.rms_norm(x, Wd["gamma"], 1e-12)
        qkv = xn @ Wd["qkv"].T                        # [B,S,3hd] (W stored [3hd,dim])
        hd = self.hd
        q = qkv[..., :hd].reshape(B, S, self.heads, self.dh)
        k = qkv[..., hd:2 * hd].reshape(B, S, self.heads, self.dh)
        v = qkv[..., 2 * hd:].reshape(B, S, self.heads, self.dh)
        q, k = self._rope_fast(q), self._rope_fast(k)
        q = q.transpose(0, 2, 1, 3)                     # [B,H,S,D]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, hd)   # [B,S,hd]
        gates = mx.sigmoid(xn @ Wd["gates_w"].T + Wd["gates_b"])   # [B,S,H]
        o = o.reshape(B, S, self.heads, self.dh) * gates[..., None]
        o = (o.reshape(B, S, hd)) @ Wd["out"].T
        return o + x

    def run_bridge(self, x_np_f16: np.ndarray, d: int) -> np.ndarray:
        """np [60,T,dim] fp16 -> mx -> attn -> np fp16. Includes all bridge cost."""
        mx = self.mx
        x = mx.array(x_np_f16)
        y = self.attn(x, d)
        mx.eval(y)
        return np.asarray(y)


# --------------------------------------------------------------------------
# FORM A-PRIME: ANE keeps all GEMM (rmsnorm/qkv/out-proj/gates), GPU does
# rope+SDPA only. Weight-fold trick: q/k rows permuted per head to [evens|odds]
# halves so GPU rope is the fast concat impl; 1/sqrt(dh) folded into q weights
# (sdpa scale=1.0). v unfolded (rope-free). Coordinate permutation is orthogonal
# -> numerically identical to the torch interleaved rope.
# --------------------------------------------------------------------------

def _pair_fold(W: np.ndarray, heads: int, dh: int) -> np.ndarray:
    """[heads*dh, in] -> per head [evens|odds] row permutation."""
    parts = []
    for h in range(heads):
        Wh = W[h * dh:(h + 1) * dh]
        parts.append(np.concatenate([Wh[0::2], Wh[1::2]], axis=0))
    return np.concatenate(parts, axis=0)


def build_pa(T, cfg, w, dims, d):
    """ANE program A: x [1,T,60,dim] -> qkv [60,T,3hd] folded-halves, q pre-scaled."""
    dim = cfg["dim"]; heads, dh = cfg["heads"], cfg["dim_head"]; hd = heads * dh
    x = af_input_t(T, dim)
    xt = x.transpose([0, 2, 1, 3]).reshape((60, T, dim))
    xn = bg.rms_last(xt, w[f"layers.{d}.0.layers.0.0.norm.gamma"])
    Wqkv = w[f"layers.{d}.0.layers.0.0.to_qkv.weight"].astype(F32)   # [3hd, dim]
    Wq = _pair_fold(Wqkv[:hd], heads, dh) * (dh ** -0.5)
    Wk = _pair_fold(Wqkv[hd:2 * hd], heads, dh)
    Wv = Wqkv[2 * hd:]
    Wfold = np.concatenate([Wq, Wk, Wv], axis=0).astype(F16)
    qkv = bg.big_linear(xn, Wfold)                                   # [60,T,3hd]
    return qkv


class MLXSDPACore:
    """GPU core: qkv [60,T,1536] folded -> rope(halves) -> sdpa(scale=1) -> o [60,T,512]."""

    def __init__(self, w, cfg, T):
        import mlx.core as mx
        self.mx = mx
        heads, dh = cfg["heads"], cfg["dim_head"]
        self.heads, self.dh = heads, dh
        t_freqs = w["layers.0.0.layers.0.0.rotary_embed.freqs"]
        ang = np.arange(T, dtype=F32)[:, None] * t_freqs[None, :].astype(F32)
        self.c = mx.array(np.cos(ang).astype(F16)[None, :, None, :])   # [1,T,1,dh/2]
        self.s = mx.array(np.sin(ang).astype(F16)[None, :, None, :])

    def core(self, qkv):
        mx = self.mx
        hd = self.heads * self.dh
        B, S, _ = qkv.shape
        q = qkv[..., :hd].reshape(B, S, self.heads, 2, self.dh // 2)
        k = qkv[..., hd:2 * hd].reshape(B, S, self.heads, 2, self.dh // 2)
        v = qkv[..., 2 * hd:].reshape(B, S, self.heads, self.dh)
        c, s = self.c, self.s
        qe, qo = q[..., 0, :], q[..., 1, :]
        ke, ko = k[..., 0, :], k[..., 1, :]
        qr = mx.concatenate([qe * c - qo * s, qo * c + qe * s], axis=-1).reshape(B, S, self.heads, self.dh)
        kr = mx.concatenate([ke * c - ko * s, ko * c + ke * s], axis=-1).reshape(B, S, self.heads, self.dh)
        qt, kt, vt = qr.transpose(0, 2, 1, 3), kr.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=1.0)   # scale folded in W
        return o.transpose(0, 2, 1, 3).reshape(B, S, hd)

    def run_bridge(self, qkv_np_f16: np.ndarray) -> np.ndarray:
        mx = self.mx
        y = self.core(mx.array(qkv_np_f16))
        mx.eval(y)
        return np.ascontiguousarray(np.asarray(y))


def build_pm2(T, cfg, w, dims, d):
    """ANE program M-prime: o [60,T,512] + x [60,T,256] -> gates/out-proj/residual
    + t-ffn + freq transformer -> [1,T,60,dim]."""
    dim = cfg["dim"]; heads, dh = cfg["heads"], cfg["dim_head"]; hd = heads * dh
    o = __import__("aneforge").input((60, T, hd), max_abs=16.0)
    x = __import__("aneforge").input((60, T, dim), max_abs=16.0)
    pfx = f"layers.{d}.0.layers.0.0"
    xn = bg.rms_last(x, w[f"{pfx}.norm.gamma"])
    gates = bg.big_linear(xn, w[f"{pfx}.to_gates.weight"], bias=w[f"{pfx}.to_gates.bias"]).sigmoid()
    og = o.reshape((60, T, heads, dh)) * gates.reshape((60, T, heads, 1))
    y = bg.big_linear(og.reshape((60, T, hd)), w[f"{pfx}.to_out.0.weight"])   # no bias
    x1 = y + x
    for i in range(cfg["time_depth"]):
        x1 = bg.ffn(x1, w, f"layers.{d}.0.layers.{i}.1") + x1
    x1 = bg.rms_last(x1, w[f"layers.{d}.0.norm.gamma"])
    xb = x1.reshape((1, 60, T, dim)).transpose([0, 2, 1, 3])
    f_rope = bg.rope_bakes(w["layers.0.1.layers.0.0.rotary_embed.freqs"], 60)
    xf = xb.reshape((T, 60, dim))
    xf = bg.transformer(xf, w, cfg, f_rope, f"layers.{d}.1", cfg["freq_depth"])
    return xf.reshape((1, T, 60, dim))


def hybrid_forward_a2(nets, core, x_in_f16, cfg, dims, T, timing=None):
    """Form A-prime: P1 -> (Pa -> GPU sdpa -> PM2) x depth -> P2."""
    D = cfg["depth"]
    x = feed_p1(nets["p1"], x_in_f16, T, dims).astype(F16)   # [1,T,60,dim]
    for d in range(D):
        t0 = time.time()
        qkv = nets[f"pa{d}"](x)                               # [60,T,1536] per-depth weights
        if timing is not None: timing[f"ane_pa_{d}"] = time.time() - t0
        t0 = time.time()
        o = core.run_bridge(np.asarray(qkv, dtype=F16))       # GPU rope+sdpa [60,T,512]
        if timing is not None: timing[f"gpu_core_{d}"] = time.time() - t0
        t0 = time.time()
        xt = x[0].transpose(1, 0, 2)                          # [60,T,dim] true transpose
        y = nets[f"pm2_{d}"](o, xt)
        if timing is not None: timing[f"ane_pm2_{d}"] = time.time() - t0
        x = np.asarray(y).astype(F16)
    t0 = time.time()
    out = nets["p2"](x)
    if timing is not None: timing["ane_p2_mask"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# compile / cache
# --------------------------------------------------------------------------

def compile_segments(T, cfg, w, dims, use_cache=True, reduce_fix=True, depth_nets=True):
    """Compile P1, PM(d=0..depth-1), P2. Returns dict of af.Model."""
    import aneforge as af
    CACHE.mkdir(parents=True, exist_ok=True)
    nets = {}
    specs = [("p1", lambda: build_p1(T, cfg, w, dims))]
    if depth_nets:
        for d in range(cfg["depth"]):
            specs.append((f"pm{d}", lambda d=d: build_pm(T, cfg, w, dims, d)))
    else:
        specs.append(("pm0", lambda: build_pm(T, cfg, w, dims, 0)))
    specs.append(("p2", lambda: build_p2(T, cfg, w, dims)))
    for name, fn in specs:
        t0 = time.time()
        out = fn()
        if reduce_fix:
            out = af.reduce_sum_to_matmul(out)
        net = af.compile(out)
        nets[name] = net
        print(f"  compiled {name}: {time.time()-t0:.1f}s, {net.n_ops} ops, "
              f"in={[s for _, s in net._inputs] if hasattr(net, '_inputs') else '?'} out={net_out_shape(net)}", flush=True)
    return nets


def net_out_shape(net):
    try:
        return net._out_shape
    except Exception:
        return "?"


def feed_p1(net, xin, T, dims):
    if T >= 385:
        feeds, off = [], 0
        for (s, e, d) in bg.dim_groups(dims):
            w_g = (e - s) * d
            feeds.append(xin[:, :, off:off + w_g].reshape(1, T, e - s, d))
            off += w_g
        return net(*feeds)
    return net(xin)


# --------------------------------------------------------------------------
# hybrid forward (with segment timing)
# --------------------------------------------------------------------------

def hybrid_forward(nets, mlxattn, x_in_f16, cfg, dims, T, timing=None):
    """Full hybrid forward. x_in_f16: [1,T,L]. Returns [T, S*L] masks + timing dict."""
    D = cfg["depth"]
    x = feed_p1(nets["p1"], x_in_f16, T, dims)              # [1,T,60,dim] f32 (net() casts)
    x = x.astype(F16)
    for d in range(D):
        xt = x[0].transpose(1, 0, 2)                        # [60,T,dim] fp16
        t0 = time.time()
        yt = mlxattn.run_bridge(xt, d)                      # GPU: x+attn(x) [60,T,dim]
        if timing is not None:
            timing[f"gpu_t_attn_{d}"] = time.time() - t0
        t0 = time.time()
        y_in = np.expand_dims(yt.transpose(1, 0, 2), 0)     # [1,T,60,dim] (true transpose!)
        y = nets[f"pm{d}"](y_in)                            # ANE: t-ffn + freq-tf
        if timing is not None:
            timing[f"ane_pm_{d}"] = time.time() - t0
        x = np.asarray(y).astype(F16)
    t0 = time.time()
    out = nets["p2"](x)
    if timing is not None:
        timing["ane_p2_mask"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# parity vs numpy reference (parity.py ref_forward)
# --------------------------------------------------------------------------

def parity(T=64, depth=2):
    import parity as par
    import aneforge as af
    cfg, w, dims = load_all()
    cfg = dict(cfg); cfg["depth"] = depth
    rng = np.random.default_rng(42)
    L = sum(dims)
    xin = (rng.standard_normal((1, T, L)) * 0.5).astype(F32)
    w64 = {k: v.astype(np.float64) for k, v in w.items() if v.dtype == F16}
    ref = par.ref_forward(xin.astype(np.float64), w64, cfg, dims)   # [S,T,L]

    nets = compile_segments(T, cfg, w, dims)
    mlxattn = MLXAttention(w, cfg, T)
    y = hybrid_forward(nets, mlxattn, xin.astype(F16), cfg, dims, T)
    # y: [T, S*L] in dims_all group order -> reuse parity.py rebuild logic
    S = cfg["num_stems"]
    ref2 = ref.reshape(S, T, -1)
    L_full = S * ref.shape[-1]
    ref_y = np.zeros((T, L_full), dtype=np.float64)
    dims_all = []
    for s in range(S):
        dims_all += dims
    ptr = 0
    for (s0, e0, d_in) in bg.dim_groups(dims_all):
        for grow in range(s0, e0):
            s, b = grow // 60, grow % 60
            b_off = sum(dims[:b])
            ref_y[:, ptr:ptr + d_in] = ref2[s, :, b_off:b_off + d_in]
            ptr += d_in
    a = ref_y.flatten(); b_ = y.flatten().astype(np.float64)
    cos = float((a @ b_) / (np.linalg.norm(a) * np.linalg.norm(b_)))
    print(f"PARITY T={T} depth={depth}: cos={cos:.6f} "
          f"({'PASS' if cos > 0.999 else 'FAIL'})")
    return cos


# --------------------------------------------------------------------------
# bench: 3-way @ T
# --------------------------------------------------------------------------

def bench_hybrid(T, N=20, use_view_io=True):
    cfg, w, dims = load_all()
    nets = compile_segments(T, cfg, w, dims)
    mlxattn = MLXAttention(w, cfg, T)
    rng = np.random.default_rng(0)
    L = sum(dims)
    xin = (rng.standard_normal((1, T, L)) * 0.5).astype(F16)
    y = hybrid_forward(nets, mlxattn, xin, cfg, dims, T)    # warmup
    timing = {}
    t_all = []
    for i in range(N):
        t0 = time.time()
        y = hybrid_forward(nets, mlxattn, xin, cfg, dims, T, timing if i >= N // 2 else None)
        t_all.append(time.time() - t0)
    dt = sorted(t_all)[len(t_all) // 2]
    # aggregate
    gpu = sum(v for k, v in timing.items() if k.startswith("gpu"))
    ane = sum(v for k, v in timing.items() if k.startswith("ane"))
    cnt = max(1, N - N // 2)
    print(f"HYBRID T={T}: median={dt*1000:.1f}ms  (gpu_attn={gpu/cnt*1000:.1f}ms "
          f"ane_seg={ane/cnt*1000:.1f}ms  bridge=implicit)")
    segs = {}
    for k, v in sorted(timing.items()):
        segs[k] = v / cnt * 1000
    for k in sorted(segs, key=segs.get, reverse=True)[:8]:
        print(f"    {k}: {segs[k]:.2f}ms")
    return dt, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["build", "parity", "bench", "all"])
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--parity-T", type=int, default=64)
    ap.add_argument("--parity-depth", type=int, default=2)
    ap.add_argument("--N", type=int, default=20)
    args = ap.parse_args()
    if args.mode in ("parity", "all"):
        parity(args.parity_T, args.parity_depth)
    if args.mode in ("bench", "all"):
        bench_hybrid(args.T, args.N)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# FORM A-PRIME bench + parity
# --------------------------------------------------------------------------

def compile_a2(T, cfg, w, dims, reduce_fix=True):
    """Compile P1, Pa(d)/PM2(d) per depth, P2. Per-depth nets carry that depth's
    weights (time-branch weights differ every layer)."""
    import aneforge as af
    nets = {}
    specs = [("p1", lambda: build_p1(T, cfg, w, dims))]
    for d in range(cfg["depth"]):
        specs.append((f"pa{d}", lambda d=d: build_pa(T, cfg, w, dims, d)))
        specs.append((f"pm2_{d}", lambda d=d: build_pm2(T, cfg, w, dims, d)))
    specs.append(("p2", lambda: build_p2(T, cfg, w, dims)))
    for name, fn in specs:
        out = fn()
        if reduce_fix:
            out = af.reduce_sum_to_matmul(out)
        nets[name] = af.compile(out)
        print(f"  compiled {name}: {nets[name].n_ops} ops", flush=True)
    return nets


def parity_a2(T=64, depth=2):
    import parity as par
    cfg, w, dims = load_all()
    cfg = dict(cfg); cfg["depth"] = depth
    rng = np.random.default_rng(42)
    L = sum(dims)
    xin = (rng.standard_normal((1, T, L)) * 0.5).astype(F32)
    w64 = {k: v.astype(np.float64) for k, v in w.items() if v.dtype == F16}
    ref = par.ref_forward(xin.astype(np.float64), w64, cfg, dims)

    nets = compile_a2(T, cfg, w, dims)
    core = MLXSDPACore(w, cfg, T)
    y = hybrid_forward_a2(nets, core, xin.astype(F16), cfg, dims, T)
    S = cfg["num_stems"]
    ref2 = ref.reshape(S, T, -1)
    L_full = S * ref.shape[-1]
    ref_y = np.zeros((T, L_full), dtype=np.float64)
    dims_all = []
    for s in range(S):
        dims_all += dims
    ptr = 0
    for (s0, e0, d_in) in bg.dim_groups(dims_all):
        for grow in range(s0, e0):
            s, b = grow // 60, grow % 60
            b_off = sum(dims[:b])
            ref_y[:, ptr:ptr + d_in] = ref2[s, :, b_off:b_off + d_in]
            ptr += d_in
    a = ref_y.flatten(); b_ = y.flatten().astype(np.float64)
    cos = float((a @ b_) / (np.linalg.norm(a) * np.linalg.norm(b_)))
    print(f"PARITY-A2 T={T} depth={depth}: cos={cos:.6f} "
          f"({'PASS' if cos > 0.999 else 'FAIL'})")
    return cos


def bench_a2(T, N=20):
    cfg, w, dims = load_all()
    nets = compile_a2(T, cfg, w, dims)
    core = MLXSDPACore(w, cfg, T)
    rng = np.random.default_rng(0)
    L = sum(dims)
    xin = (rng.standard_normal((1, T, L)) * 0.5).astype(F16)
    y = hybrid_forward_a2(nets, core, xin, cfg, dims, T)
    timing = {}
    t_all = []
    for i in range(N):
        t0 = time.time()
        y = hybrid_forward_a2(nets, core, xin, cfg, dims, T, timing if i >= N // 2 else None)
        t_all.append(time.time() - t0)
    dt = sorted(t_all)[len(t_all) // 2]
    cnt = max(1, len(timing) and N - N // 2)
    agg = {}
    for pref in ("ane_pa", "gpu_core", "ane_pm2", "ane_p2"):
        keys = [k for k in timing if k.startswith(pref)]
        if keys:
            agg[pref] = sum(timing[k] for k in keys) / (cnt / max(1, len(keys))) * 1000
    print(f"HYBRID-A2 T={T}: median={dt*1000:.1f}ms  " +
          "  ".join(f"{k}={v:.1f}ms/iter" for k, v in agg.items()))
    return dt, timing
