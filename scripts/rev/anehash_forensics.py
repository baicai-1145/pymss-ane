#!/usr/bin/env python3
"""anehash forensics summary + regression tool.

Established rules (230/271-cache corpus, task-17):
  cache-dir name     = SHA256(model.mil + weights.bin)[:24]   (271/271)
  anehash layout     = <h1>_<h2>, both 64-hex uppercase
    h1: varies per model (230 distinct), NOT any simple hash of on-disk files
        (mil/weights/e5/bundle-name/all 10 combinations tested = 0 hits).
        Semantics: hash of the ANECIR net.plist (compiler IR, lives only in
        the ANECompilerService XPC process / compile tmpdir /net.anehash).
    h2: varies per op at same shape (c2/c4/c5/c_softmax evidence) = runtime
        container identity hash.
  Disk loads verify bundle integrity via verifyBundleAtPath (TOCTOU check,
  AppleNeuralEngine dsc) -> a patched bundle file needs a recomputed anehash,
  whose inputs are not reconstructible from disk. Conclusion: bundle-file
  patching (channel a) is hash-gated; in-memory interpose (channel b) is the
  only viable surgery channel, consistent with the task-16 correction.
"""
import hashlib
import sys
from pathlib import Path


def cache_dir_hash(mil: bytes, weights: bytes) -> str:
    return hashlib.sha256(mil + weights).hexdigest()[:24]


def verify_cache(cache_root=None) -> int:
    cache = Path(cache_root or (Path.home() / "Models/.aneforge-cache"))
    n = hit = 0
    for d in sorted(cache.iterdir()):
        mil, w = d / "model.mil", d / "weights.bin"
        if not (mil.exists() and w.exists()):
            continue
        n += 1
        if cache_dir_hash(mil.read_bytes(), w.read_bytes()) == d.name:
            hit += 1
    print(f"cache-dir == SHA256(mil+weights)[:24]: {hit}/{n}")
    return hit == n


if __name__ == "__main__":
    sys.exit(0 if verify_cache(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
