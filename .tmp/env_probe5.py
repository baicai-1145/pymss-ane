import os, subprocess, hashlib
BASE = "/Volumes/2T/repos/mss/pymss-ane"
CH = 104
TEST = f'''
import aneforge as af, numpy as np, hashlib
from pathlib import Path
x = af.input((1,{CH},1,4)); w = np.zeros((8,{CH},1,1), dtype=np.float16); w[0,0,0,0] = 1.0
m = af.compile(af.conv(x,w))
out = m(np.zeros((1,{CH},1,4), dtype=np.float16))
print("OK")
cache = Path.home() / "Models/.aneforge-cache"
d = cache / "63a262371474380ba956007e"
for f in sorted(d.rglob("H16G.e5")):
    print("E5", hashlib.md5(f.read_bytes()).hexdigest()[:12])
for f in sorted(d.rglob("model.anehash")):
    print("ANEH", open(f).read().strip()[:64])
'''
env = dict(os.environ)
for k in list(os.environ):
    if k.startswith("ESPRESSO"): del env[k]
env["ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED"] = "1"
r = subprocess.run([BASE+"/.venv/bin/python","-W","ignore","-c",TEST],
                   capture_output=True, text=True, env=env, timeout=300, cwd=BASE)
print("ALWAYS_COMPRESSED same-input:", r.stdout.strip().splitlines()[-2:])
print("baseline: E5 f739c513a6f8, ANEH b0c78183...  (dir 63a26237)")
