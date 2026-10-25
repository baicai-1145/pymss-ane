"""MLX segment timing: STFT / bandsplit / time-tf / freq-tf / mask / iSTFT."""
import sys
import time

import numpy as np

sys.path.insert(0, "/Volumes/2T/repos/mss/pymss-core")
import mlx.core as mx
import torch


def main():
    from pymss_core.modules.bs_roformer.mel_band_roformer import MelBandRoformer
    from pymss_core.modules.bs_roformer import mlx_roformer as mr

    cfg = dict(
        dim=256, depth=12, stereo=True, num_stems=2,
        time_transformer_depth=1, freq_transformer_depth=1,
        num_bands=60, dim_head=64, heads=8,
        attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
        sample_rate=44100, stft_n_fft=2048, stft_hop_length=441,
        stft_win_length=2048, stft_normalized=False,
        zero_dc=True, mask_estimator_depth=2, mlp_hidden_layers=1,
        match_input_audio_length=False,
    )
    model = MelBandRoformer(**cfg)
    model.eval()
    model.mps_model_compute_dtype = mx.float16

    # monkey-patch segment timers on the mlx path
    seg = {}
    import pymss_core.modules.bs_roformer.mlx_roformer as m
    orig = {}
    for fn in ["_band_split", "_transformer", "_estimate_masks", "_final_norm"]:
        if hasattr(m, fn):
            orig[fn] = getattr(m, fn)

    def timed(fn, name):
        def wrap(*a, **k):
            t0 = time.time()
            r = fn(*a, **k)
            mx.eval(r)
            seg[name] = seg.get(name, 0.0) + (time.time() - t0)
            return r
        return wrap

    if "_band_split" in orig:
        m._band_split = timed(orig["_band_split"], "band_split")
    if "_transformer" in orig:
        m._transformer = timed(orig["_transformer"], "transformer(time+freq)")
    if "_estimate_masks" in orig:
        m._estimate_masks = timed(orig["_estimate_masks"], "estimate_masks")
    if "_final_norm" in orig:
        m._final_norm = timed(orig["_final_norm"], "final_norm")

    for T in [256, 512]:
        seg.clear()
        raw = torch.from_numpy(
            np.random.default_rng(0).standard_normal((1, 2, T * 441)).astype(np.float32) * 0.1
        )
        out = mr.mlx_forward_roformer(model, raw)   # warmup
        seg.clear()
        N = 5
        t0 = time.time()
        for _ in range(N):
            out = mr.mlx_forward_roformer(model, raw)
        total = (time.time() - t0) / N
        parts = "  ".join(f"{k}={v/N*1000:.0f}ms" for k, v in seg.items())
        other = total - sum(seg.values()) / N
        print(f"T={T}: total={total*1000:.0f}ms  {parts}  stft/istft+other={other*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
