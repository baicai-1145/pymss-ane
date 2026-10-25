"""MLX baseline: same model, same T, mlx_roformer full path, on M4 Air.

Uses pymss-core's MLX backend for an apples-to-apples transformer-stack latency.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/Volumes/2T/repos/mss/pymss-core")

import mlx.core as mx


def main():
    from pymss_core.modules.bs_roformer.mel_band_roformer import MelBandRoformer
    from pymss_core.modules.bs_roformer import mlx_roformer
    from pymss_core.modules.bs_roformer.common import init_roformer_layers, roformer_transformer_kwargs

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

    for T in [64, 256, 512]:
        raw = np.random.default_rng(0).standard_normal((1, 2, T * 441)).astype(np.float32) * 0.1
        import torch
        raw_t = torch.from_numpy(raw)
        # warmup
        try:
            out = mlx_roformer.mlx_forward_roformer(model, raw_t)
        except Exception as e:
            print(f"T={T}: mlx forward FAIL {e!r}", flush=True)
            continue
        N = 5
        t0 = time.time()
        for _ in range(N):
            out = mlx_roformer.mlx_forward_roformer(model, raw_t)
        dt = (time.time() - t0) / N
        print(f"T={T}: mlx_full latency={dt*1000:.1f}ms (incl STFT/iSTFT+gather)", flush=True)


if __name__ == "__main__":
    main()
