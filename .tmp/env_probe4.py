import os, subprocess, hashlib
BASE = "/Volumes/2T/repos/mss/pymss-ane"
CH = 96
TEST = f'''
import aneforge as af, numpy as np, hashlib
from pathlib import Path
x = af.input((1,{CH},1,4)); w = np.zeros((8,{CH},1,1), dtype=np.float16); w[0,0,0,0] = 1.0
m = af.compile(af.conv(x,w))
out = m(np.zeros((1,{CH},1,4), dtype=np.float16))
print("OK")
cache = Path.home() / "Models/.aneforge-cache"
d = cache / "5c6611e8392a75a610c983d3"  # CH=96 cache dir
for f in sorted(d.rglob("H16G.e5")):
    print("E5", f.stat().st_size, hashlib.md5(f.read_bytes()).hexdigest()[:12])
for f in sorted(d.rglob("model.anehash")):
    print("ANEH", open(f).read()[:32])
'''
env = dict(os.environ)
for k in list(os.environ):
    if k.startswith("ESPRESSO"): del env[k]
env["ESPRESSO_PRIVATE_IGNORE_ANE_PRECOMPILED"] = "1"
r = subprocess.run([BASE+"/.venv/bin/python","-W","ignore","-c",TEST],
                   capture_output=True, text=True, env=env, timeout=300, cwd=BASE)
print("IGNORE_PRECOMPILED on same-input:", r.stdout.strip().splitlines()[-3:], r.returncode)
print("baseline E5 was: 3864 815d113c8043 (dir 5c6611e8)")
