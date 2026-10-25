import os, sys, subprocess, hashlib, json

BASE = "/Volumes/2T/repos/mss/pymss-ane"
TEST = r'''
import aneforge as af, numpy as np, os, hashlib
from pathlib import Path
x = af.input((1,4,1,4)); w = np.zeros((8,4,1,1), dtype=np.float16); w[0,0,0,0] = 1.0
m = af.compile(af.conv(x,w))
out = m(np.zeros((1,4,1,4), dtype=np.float16))
print("OK compiled+ran")
# report cache bundle hashes
cache = Path.home() / "Models/.aneforge-cache"
best = None
for d in sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime):
    if not (d/"cache").exists(): continue
    anehs = list((d/"cache").rglob("model.anehash"))
    if anehs: best = d
if best:
    print("newest cache:", best.name)
    for f in best.rglob("*"):
        if f.is_file():
            print("  ", f.name, f.stat().st_size, hashlib.md5(f.read_bytes()).hexdigest()[:12])
'''

def run_case(name, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env["ANE_CACHE_TAG"] = name  # informational only
    r = subprocess.run([BASE + "/.venv/bin/python", "-W", "ignore", "-c", TEST],
                       capture_output=True, text=True, env=env, timeout=300, cwd=BASE)
    out = r.stdout.strip().splitlines()
    tail = [l for l in out if not l.startswith("  ")][-1:]
    files = [l for l in out if l.startswith("  ")]
    print(f"[{name}] rc={r.returncode} {' '.join(tail)}")
    for f in files[:6]:
        print("   ", f.strip())
    if r.returncode != 0:
        print("   stderr:", r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else "-")
    return r.returncode, files

print("=== baseline ===")
run_case("baseline", {})
print("=== IGNORE_ANE_PRECOMPILED=1 ===")
run_case("ignore_precompiled", {"ESPRESSO_PRIVATE_IGNORE_ANE_PRECOMPILED": "1"})
print("=== ALWAYS_COMPRESSED=1 ===")
run_case("always_compressed", {"ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED": "1"})
print("=== TUNELOG=1 ===")
run_case("tunelog", {"ESPRESSO_TUNELOG": "1"})
