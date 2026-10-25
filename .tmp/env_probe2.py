import os, subprocess, hashlib, json, time
BASE = "/Volumes/2T/repos/mss/pymss-ane"
import sys
SHAPE_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ENVK = sys.argv[2] if len(sys.argv) > 2 else ""
ENVV = sys.argv[3] if len(sys.argv) > 3 else "1"
# shape varies per invocation to force fresh compile
CH = [96, 104, 112, 120][SHAPE_IDX]
TEST = f'''
import aneforge as af, numpy as np, hashlib
from pathlib import Path
x = af.input((1,{CH},1,4)); w = np.zeros((8,{CH},1,1), dtype=np.float16); w[0,0,0,0] = 1.0
m = af.compile(af.conv(x,w))
out = m(np.zeros((1,{CH},1,4), dtype=np.float16))
print("OK CH={CH}")
cache = Path.home() / "Models/.aneforge-cache"
newest = max((d for d in cache.iterdir() if (d/"cache").exists() and any((d/"cache").rglob("model.anehash"))), key=lambda p: p.stat().st_mtime)
print("cache:", newest.name)
for f in sorted(newest.rglob("*")):
    if f.is_file() and f.name in ("H16G.e5", "model.anehash", "weights.bin"):
        print(f.name, f.stat().st_size, hashlib.md5(f.read_bytes()).hexdigest()[:12])
'''
env = dict(os.environ)
if ENVK: env[ENVK] = ENVV
r = subprocess.run([BASE+"/.venv/bin/python","-W","ignore","-c",TEST],
                   capture_output=True, text=True, env=env, timeout=300, cwd=BASE)
print(r.stdout.strip())
if r.returncode: print("ERR", r.stderr.strip().splitlines()[-1][:200])
