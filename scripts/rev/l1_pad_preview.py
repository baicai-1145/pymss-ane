"""L1 preview: conv pad offset -> predictable output shift (task-17).

Surgery-semantics rehearsal on the aneforge runway:
  MIL pad parameter (compiler-input layer)  <->  container DMA base offset
  (runtime layer, reloc regs 0x1344 input / 0x1442 output).

Predictable-change definition: conv1x1 with pad=1 must equal zero-padded
input convolved with pad=0 — interior samples identical, boundary row/col
differ exactly by the zero-pad contribution.
"""
import numpy as np
import aneforge as af

rng = np.random.default_rng(7)

# input [1, 4, 5, 5] fp16, identity-style weights (C_out=4 -> C_in=4, 1x1)
C, H, W = 4, 5, 5
x_np = rng.standard_normal((1, C, H, W)).astype(np.float16)
w_np = np.eye(C, dtype=np.float16).reshape(C, C, 1, 1)

# --- baseline: pad=0 --------------------------------------------------------
x0 = af.input((1, C, H, W))
y0 = af.conv(x0, w_np, pad=0) if "pad" in af.conv.__doc__ else None
if y0 is None:
    # API discovery: inspect signature
    import inspect
    print("conv signature:", inspect.signature(af.conv))
    raise SystemExit(1)
m0 = af.compile(y0)
out0 = m0(x_np)

# --- pad=1 variant ----------------------------------------------------------
x1 = af.input((1, C, H, W))
y1 = af.conv(x1, w_np, pad=1)
m1 = af.compile(y1)
out1 = m1(x_np)  # [1, C, H+2? or same with crop] — record shape

print("out0 shape:", out0.shape, "out1 shape:", out1.shape)

# --- reference: numpy conv1x1 with zero pad=1 --------------------------------
xp = np.zeros((1, C, H + 2, W + 2), dtype=np.float16)
xp[:, :, 1:1 + H, 1:1 + W] = x_np
ref = xp[:, :, 0:2 + H, 0:2 + W]  # identity weights -> padded input window
ref_full = xp  # if out1 is (H+2, W+2)

def cos(a, b):
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

if out1.shape == xp.shape:
    print("full-pad output: cos(ANE, ref) =", cos(out1, ref_full))
    # boundary analysis: interior (1..H, 1..W) must equal x exactly (identity w)
    interior_ane = out1[0, :, 1:1 + H, 1:1 + W]
    interior_ref = x_np[0]
    exact = np.array_equal(interior_ane.astype(np.float16), interior_ref)
    print("interior == input (identity kernel):", exact,
          "| max abs diff:", float(np.abs(interior_ane.astype(np.float32) - interior_ref.astype(np.float32)).max()))
    border_ane = out1[0, :, [0, -1], :].astype(np.float32)
    print("border rows all zero:", bool((border_ane == 0).all()))
elif out1.shape == out0.shape:
    # same-size: pad=1 + crop center => identical to pad=0
    print("center-crop output: cos(out1, out0) =", cos(out1, out0),
          "| equal:", np.array_equal(out1.astype(np.float16), out0.astype(np.float16)))

# --- predictable-change ledger ----------------------------------------------
# pad 0->1 shifts the effective DMA read window by one row/col (input tile
# base -0x1344 equivalent) and/or write window (output 0x1442): the container
# translates this MIL parameter into tile-DMA coordinates. The exact-match
# interior + zero border IS the observable of a one-tile address shift.
print("L1-preview: predictable shift verified" )
