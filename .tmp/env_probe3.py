import os, subprocess, hashlib
BASE = "/Volumes/2T/repos/mss/pymss-ane"
CH = 96  # same as baseline fresh compile
TEST = f'''
import aneforge as af, numpy as np, hashlib
from pathlib import Path
x = af.input((1,{CH},1,4)); w = np.zeros((8,{CH},1,1), dtype=np.float16); w[0,0,0,0] = 1.0
m = af.compile(af.conv(x,w))
out = m(np.zeros((1,{CH},1,4), dtype=np.float16))
print("OK")
cache = Path.home() / "Models/.aneforge-cache"
newest = max((d for d in cache.iterdir() if (d/"cache").exists() and any((d/"cache").rglob("model.anehash"))), key=lambda p: p.stat().st_mtime)
for f in sorted(newest.rglob("H16G.e5")):
    print("E5", f.stat().st_size, hashlib.md5(f.read_bytes()).hexdigest()[:12])
print("CACHEDIR", newest.name)
'''
env = dict(os.environ)
for k in os.environ:
    if k.startswith("ESPRESSO"): del env[k]
r = subprocess.run([BASE+"/.venv/bin/python","-W","ignore","-c",TEST],
                   capture_output=True, text=True, env=env, timeout=300, cwd=BASE)
print("no-env rerun:", r.stdout.strip().splitlines()[-2:], r.returncode)
