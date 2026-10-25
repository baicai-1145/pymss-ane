#!/bin/bash
# aneforge MIL runway: [60,8,256,256]-equiv softmax variants
run_mil_variant() {
  local NAME="$1" FLAGS="$2"
  log stream --predicate 'eventMessage CONTAINS "zin_ane_compiler"' --style compact --debug > /tmp/t20m_l_$NAME.txt 2>&1 &
  LP=$!
  for i in $(seq 1 30); do grep -q Filtering /tmp/t20m_l_$NAME.txt && break; done
  ANE_ANEC_FLAGS="$FLAGS" /Volumes/2T/repos/mss/pymss-ane/.venv/bin/python -W ignore - <<PYEOF > /tmp/t20m_b_$NAME.log 2>&1
import aneforge as af, numpy as np, time
x = af.input((1,480,256,256))
y = x.softmax()
m = af.compile(y)
inp = np.zeros((1,480,256,256), dtype=np.float16)
inp[0,:,0,0] = 2.0  # deterministic non-trivial row
out = m(inp)
for i in range(5): m(inp)
ts = []
for i in range(20):
    t0 = time.perf_counter(); m(inp); ts.append((time.perf_counter()-t0)*1000)
ts.sort()
print(f"RESULT p50={ts[10]:.3f} p90={ts[18]:.3f} chk={float(out[0,0,0,0]):.6f}")
PYEOF
  RC=$?
  kill $LP 2>/dev/null
  local LINE=$(grep -oE 'RESULT.*' /tmp/t20m_b_$NAME.log | head -1)
  local LAND=$(grep -oE 'fspatial-split=[a-z-]+|memcache-size=[0-9]+|fl2-cache-mode=[a-z]+|fkernel-rewind=[a-z]+' /tmp/t20m_l_$NAME.txt | sort -u | tr '\n' ' ')
  echo "$NAME | $LINE | $LAND"
}
run_mil_variant base        ""
run_mil_variant split_auto  "SpatialSplitMode=Auto"
run_mil_variant split_mem   "SpatialSplitMode=Memory"
run_mil_variant split_dag   "SpatialSplitMode=GenericDAG"
run_mil_variant rewind_off  "KernelRewind=false"
run_mil_variant optred      "OptimizeReductionPerf=1"
