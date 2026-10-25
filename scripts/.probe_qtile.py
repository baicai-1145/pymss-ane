"""Query-tile attention: softmax is per-row => slicing query axis needs NO merge.
Each tile's sim [60,8,CH,256] stays <=2MB working set if Espresso schedules it on-chip.
Variants vs single-block baseline (T=256, 12 layers)."""
import sys
sys.path.insert(0, 'scripts')
import aneforge as af
import numpy as np
import json
import time
from pathlib import Path
import build_graph as bg
from build_graph import rms_last, big_linear, slast, rope_bakes

cfg = bg.load_cfg('/Volumes/nfs/data/Downloads/mel-band-roformer-deux/config_deux_becruily.yaml')
wdir = Path('weights')
manifest = json.loads((wdir / 'manifest.json').read_text())
w = {k: np.load(wdir / v) for k, v in manifest.items()}
rng = np.random.default_rng(0)
dim, heads, dh, hd = 256, 8, 64, 512
seq = T = 256
LOG2E = float(np.log2(np.e))


def attn_qtile(x, pfx, rope, CH):
    """query-tiled attention core, no merge (row-independent softmax)."""
    cos_dup, sin_dup, P = rope
    xn = rms_last(x, w[f'{pfx}.norm.gamma'])
    qkv = big_linear(xn, w[f'{pfx}.to_qkv.weight'])
    B = x.shape[:-2]
    q = slast(qkv, 0, hd).reshape(B + (seq, heads, dh))
    k = slast(qkv, hd, 2 * hd).reshape(B + (seq, heads, dh))
    v = slast(qkv, 2 * hd, 3 * hd).reshape(B + (seq, heads, dh))
    cd = cos_dup[None, :, None, :].copy()
    sd = sin_dup[None, :, None, :].copy()
    q = q * cd + (q @ P) * sd
    k = k * cd + (k @ P) * sd
    q, k, v = (t.transpose([0, 2, 1, 3]) for t in (q, k, v))   # [B,h,S,dh]
    q = q * (dh ** -0.5)
    kt = k.transpose([0, 1, 3, 2])
    n = (seq + CH - 1) // CH
    outs = []
    for c in range(n):
        a = min(c * CH, seq)
        sz = min(CH, seq - a)
        qc = q.slice_by_size([0, 0, a, 0], [B[0], heads, sz, dh])
        sim = qc @ kt                                            # [B,h,sz,S]
        m = sim.amax(-1)
        e = ((sim - m) * LOG2E).exp2()
        s = e.sum(-1)
        o = e @ v                                                # [B,h,sz,dh]
        outs.append(o / s)
    if n == 1:
        o = outs[0]
    else:
        o = af.concat(outs, axis=2)                              # [B,h,S,dh]
    o = o.transpose([0, 2, 1, 3]).reshape(B + (seq, hd))
    gates = big_linear(xn, w[f'{pfx}.to_gates.weight'], bias=w[f'{pfx}.to_gates.bias']).sigmoid()
    o = o.reshape(B + (seq, heads, dh)) * gates.reshape(B + (seq, heads, 1))
    return big_linear(o.reshape(B + (seq, hd)), w[f'{pfx}.to_out.0.weight'])


t_rope = rope_bakes(w['layers.0.0.layers.0.0.rotary_embed.freqs'], T)
for CH in [256, 8, 4, 16]:
    x = af.input((60, T, dim), max_abs=16.0)
    for d in range(12):
        xt = attn_qtile(x, f'layers.{d}.0.layers.0.0', t_rope, CH) + x
        x = bg.ffn(xt, w, f'layers.{d}.0.layers.0.1') + xt
    try:
        net = af.compile(x)
        feed = (rng.standard_normal((60, T, dim)) * .5).astype(np.float16)
        net(feed)
        t0 = time.time()
        for _ in range(20):
            net(feed)
        dt = (time.time() - t0) / 20
        mb = 60 * 8 * CH * T * 2 / 1e6
        print(f'qtile CH={CH:4d} (sim tile {mb:4.1f}MB): {dt*1000:7.2f}ms', flush=True)
        net.release()
    except Exception as e:
        print(f'qtile CH={CH}: FAIL {str(e)[:60]}', flush=True)
        continue
