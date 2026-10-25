#!/usr/bin/env python3
"""Differential-automation pipeline: compile single-variable MIL graph variants,
parse the H16G.e5 descriptors, align-diff them, and emit a parameter->field map.

Usage:
  .venv/bin/python scripts/rev/diff_matrix.py --out .tmp/diff_matrix.json
  .venv/bin/python scripts/rev/diff_matrix.py --cases conv_k,conv_c --dry   # no compile

Each case = (name, builder, variant params). The pipeline:
  1. build aneforge graph variants (one parameter changed per pair)
  2. af.compile -> materializes ~/Models/.aneforge-cache/<sha>/ (content-addressed)
  3. parse each H16G.e5 with e5_parser (TD blocks + section table)
  4. pair-wise structural diff -> {param: {td_idx: {slot: (valA, valB)}}}
"""
from __future__ import annotations
import argparse, json, sys, difflib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5_parser as EP

import aneforge as af


# ---------------- graph builders (single-variable families) ----------------

def g_conv(cin, cout, w, k=1, stride=1, pad=0, act=False):
    x = af.input((1, cin, 1, w))
    W = np.zeros((cout, cin, 1, k), dtype=np.float16)
    W[0, 0, 0, 0] = 1.0
    y = af.conv(x, W, stride=stride, pad=pad)
    if act:
        y = y.relu()
    return y


def g_matmul(n, m, k):
    x = af.input((n, m, k))
    W = np.zeros((k, k), dtype=np.float16)
    W[0, 0] = 1.0
    return x @ W


def g_reduce_mean(w, axes=(2,)):
    x = af.input((1, 8, 1, w))
    return x.mean(axes)


def g_softmax(w):
    x = af.input((1, 8, 1, w))
    return x.softmax(axis=-1)


def g_batchnorm(w):
    x = af.input((1, 8, 1, w))
    g = np.ones(8, dtype=np.float16)
    b = np.zeros(8, dtype=np.float16)
    mu = np.zeros(8, dtype=np.float16)
    var = np.ones(8, dtype=np.float16)
    return af.batch_norm(x, g, b, mu, var)


# (case, baseline-args, variant-args, human label)
CASES = [
    # conv family
    ("conv_k",       g_conv, dict(cin=32, cout=32, w=64, k=1),  dict(cin=32, cout=32, w=64, k=3),  "conv kernel 1x1 -> 3x1"),
    ("conv_k5",      g_conv, dict(cin=32, cout=32, w=64, k=3),  dict(cin=32, cout=32, w=64, k=5),  "conv kernel 3 -> 5"),
    ("conv_stride",  g_conv, dict(cin=32, cout=32, w=64, stride=1), dict(cin=32, cout=32, w=64, stride=2), "conv stride 1 -> 2"),
    ("conv_pad",     g_conv, dict(cin=32, cout=32, w=64, pad=0), dict(cin=32, cout=32, w=64, pad=1), "conv pad 0 -> 1"),
    ("conv_cin",     g_conv, dict(cin=32, cout=32, w=64),  dict(cin=64, cout=32, w=64),  "conv Cin 32 -> 64"),
    ("conv_cout",    g_conv, dict(cin=32, cout=32, w=64),  dict(cin=32, cout=64, w=64),  "conv Cout 32 -> 64"),
    ("conv_w",       g_conv, dict(cin=32, cout=32, w=64),  dict(cin=32, cout=32, w=128), "conv W 64 -> 128"),
    ("conv_act",     g_conv, dict(cin=32, cout=32, w=64, act=False), dict(cin=32, cout=32, w=64, act=True), "conv + relu"),
    # matmul family
    ("mm_n",         g_matmul, dict(n=32, m=32, k=64), dict(n=64, m=32, k=64), "matmul N 32 -> 64"),
    ("mm_m",         g_matmul, dict(n=32, m=32, k=64), dict(n=32, m=64, k=64), "matmul M 32 -> 64"),
    ("mm_k",         g_matmul, dict(n=32, m=32, k=64), dict(n=32, m=32, k=128), "matmul K 64 -> 128"),
    # attention-relevant families
    ("rm_w",         g_reduce_mean, dict(w=64), dict(w=128), "reduce_mean W 64 -> 128"),
    ("rm_axes",      g_reduce_mean, dict(w=64, axes=(2,)), dict(w=64, axes=(1, 2)), "reduce_mean axes (2) -> (1,2)"),
    ("sm_w",         g_softmax, dict(w=64), dict(w=128), "softmax W 64 -> 128"),
    ("bn_w",         g_batchnorm, dict(w=64), dict(w=128), "batchnorm W 64 -> 128"),
]


def find_e5(build_key_hint: str | None = None) -> Path | None:
    root = Path.home() / "Models" / ".aneforge-cache"
    best = None
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        for p in d.rglob("H16G.e5"):
            return p
    return None


def compile_and_locate(builder, kwargs) -> tuple[Path, str]:
    """Compile a graph; return (e5 path, mil text).

    The cache is content-addressed, so we locate our dir by writing the MIL
    ourselves and matching its sha: af.compile re-emits model.mil into
    ~/Models/.aneforge-cache/<sha24>/; we hash the emitted MIL+weights via the
    same scheme (_emit_program_dir) instead of trusting mtimes (a cache hit
    reorders mtimes and breaks naive newest-dir logic)."""
    y = builder(**kwargs)
    import hashlib, os
    from aneforge._compile import _lower_fused_to_dir
    d = _lower_fused_to_dir(y)          # content-addressed emit (no compile)
    mil = (d / "model.mil").read_text()
    e5 = next(d.rglob("H16G.e5"), None)
    if e5 is None:                        # not compiled yet -> compile now
        af.compile(y)
        e5 = next(d.rglob("H16G.e5"))
    return e5, mil


def diff_parsed(pa: dict, pb: dict) -> dict:
    """Structural diff of two parsed e5s."""
    out = {}
    sa, sb = EP.signature(pa), EP.signature(pb)
    out["size"] = (sa["size"], sb["size"])
    # section-level u16 diffs
    secs = {}
    na = {s["name"]: s["u16"] for s in sa["sections"]}
    nb = {s["name"]: s["u16"] for s in sb["sections"]}
    for name in sorted(set(na) | set(nb)):
        ua, ub = na.get(name), nb.get(name)
        if ua != ub:
            secs[name] = {"a": ua, "b": ub}
    out["sections"] = secs
    # TD block diffs
    tds = []
    ta, tb = sa["td_blocks"], sb["td_blocks"]
    for i in range(max(len(ta), len(tb))):
        A = ta[i] if i < len(ta) else None
        B = tb[i] if i < len(tb) else None
        if A is None or B is None:
            tds.append({"idx": i, "a_len": A and A["len"], "b_len": B and B["len"], "changed": True})
            continue
        if A["u16"] == B["u16"]:
            continue
        slots = {}
        for k in range(max(len(A["u16"]), len(B["u16"]))):
            va = A["u16"][k] if k < len(A["u16"]) else None
            vb = B["u16"][k] if k < len(B["u16"]) else None
            if va != vb:
                slots[f"u16[{k}]"] = [va, vb]
        tds.append({"idx": i, "a_len": A["len"], "b_len": B["len"], "slots": slots})
    out["td_blocks"] = tds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/diff_matrix.json")
    ap.add_argument("--cases", default=None, help="comma-separated case names")
    ap.add_argument("--dry", action="store_true", help="parse existing cache only, no compile")
    args = ap.parse_args()

    selected = CASES
    if args.cases:
        names = set(args.cases.split(","))
        selected = [c for c in CASES if c[0] in names]

    results = []
    for name, builder, base_kw, var_kw, label in selected:
        entry = {"case": name, "label": label}
        try:
            if not args.dry:
                e5a, mila = compile_and_locate(builder, base_kw)
                e5b, milb = compile_and_locate(builder, var_kw)
                entry["e5_a"], entry["e5_b"] = str(e5a), str(e5b)
                pa = EP.parse(Path(e5a).read_bytes(), str(e5a))
                pb = EP.parse(Path(e5b).read_bytes(), str(e5b))
            else:
                raise RuntimeError("dry")
            entry["diff"] = diff_parsed(pa, pb)
            entry["ok"] = True
        except Exception as e:
            entry["ok"] = False
            entry["error"] = repr(e)[:300]
        results.append(entry)
        print(f"[{name}] {'OK' if entry.get('ok') else 'FAIL'}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    # human summary
    for r in results:
        if not r.get("ok"):
            print(f"== {r['case']}: FAILED {r.get('error','')[:120]}")
            continue
        d = r["diff"]
        print(f"== {r['case']} ({r['label']}): e5 {d['size'][0]} -> {d['size'][1]} bytes")
        for t in d["td_blocks"]:
            if t.get("slots"):
                n = len(t["slots"])
                first = list(t["slots"].items())[:6]
                print(f"   TD[{t['idx']}] len {t['a_len']}->{t['b_len']}  {n} slot diffs: "
                      + ", ".join(f"{k}:{a}->{b}" for k, (a, b) in first))
        for sname, sdiff in d["sections"].items():
            print(f"   section {sname}: u16 lists differ")


if __name__ == "__main__":
    main()
