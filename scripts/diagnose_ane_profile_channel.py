"""Test this machine's opt-in profiling/log observability, without sudo.

A successful compile does not prove stats were emitted. Absent log data does
not prove the hardware lacks counters. Stores raw stream and child outputs.
No throughput claims are made under instrumentation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def child(mode, root):
    import ctypes
    import time
    import numpy as np
    import aneforge as af
    import aneforge._runtime as rt
    framework = ctypes.CDLL('/System/Library/PrivateFrameworks/Espresso.framework/Espresso')
    symbol = hasattr(framework, 'e5rt_e5_compiler_options_set_enable_profiling')
    q = af.input((1,8,64,64))
    k = af.input((1,8,64,64))
    v = af.input((1,8,64,64))
    score = (q @ k) * (64 ** -0.5)
    e = ((score-score.amax(-1))*float(np.log2(np.e))).exp2()
    out = (e @ v)/(e @ np.ones((64,1),np.float16))
    net = af.compile(out, build_dir=str(root/mode))
    for port, _ in net._inputs:
        net.input_view(port).fill(np.float16(0.1))
    samples = []
    for _ in range(3):
        t = time.perf_counter_ns()
        net.execute()
        samples.append((time.perf_counter_ns()-t)/1e6)
    print(json.dumps(dict(mode=mode, profiling_requested=os.environ.get('ANEFORGE_PROFILE'),
                          profiling_symbol_present=symbol, runtime_library=str(rt._lib._name),
                          pid=os.getpid(), compile_execute_ok=True,
                          output_finite=bool(np.isfinite(net.output_view()).all()),
                          execute_samples_ms=samples)),flush=True)
    net.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child', choices=['control','profile'])
    ap.add_argument('--root', type=Path, default=ROOT/'.tmp/ane_profile_channel')
    args = ap.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    if args.child:
        child(args.child,args.root)
        return
    logpath = args.root/'ane_events.ndjson'
    cmd = ['/usr/bin/log','stream','--style','ndjson','--level','debug','--signpost',
           '--predicate','subsystem CONTAINS "com.apple.ane"']
    results = []
    with logpath.open('w') as f:
        stream = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:
            for mode in ['control','profile']:
                env = dict(os.environ)
                env.pop('ANEFORGE_PROFILE',None)
                if mode=='profile':
                    env['ANEFORGE_PROFILE']='1'
                try:
                    p = subprocess.run([sys.executable,__file__,'--child',mode,'--root',str(args.root)],
                                       env=env,capture_output=True,text=True,timeout=45)
                    (args.root/(mode+'.stdout')).write_text(p.stdout)
                    (args.root/(mode+'.stderr')).write_text(p.stderr)
                    row = dict(mode=mode,returncode=p.returncode,stdout=p.stdout[-2200:],stderr=p.stderr[-1200:])
                except subprocess.TimeoutExpired:
                    row = dict(mode=mode,error='Child exceeded bounded 45s observation window')
                results.append(row)
                print(json.dumps(row),flush=True)
        finally:
            if stream.poll() is None:
                stream.send_signal(signal.SIGINT)
                try:
                    stream.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stream.kill()
                    stream.wait()
    events, types, stat_fields = 0, Counter(), []
    errors = []
    for line in logpath.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if any(s in line.lower() for s in ['error','denied','not permitted']):
                errors.append(line[:300])
            continue
        events += 1
        types[event.get('eventType','unknown')] += 1
        matched = {k:v for k,v in event.items() if re.search(r'performance|perfStats|statsMask|statistics|counterValues',k,re.I)}
        msg = str(event.get('eventMessage',''))
        if matched or re.search(r'perfStatsMask|statsMask|stats.descriptor|initStatsBuffer',msg,re.I):
            stat_fields.append(dict(fields=matched,message=msg[:1500],processID=event.get('processID')))
    summary = dict(children=results,log_command=cmd,log_exit_code=stream.returncode,
                   event_count=events,event_types=dict(types),stat_records=stat_fields[:40],
                   log_errors=errors[:10],raw_log=str(logpath))
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Log events={events}; event types={dict(types)}; stats-related records={len(stat_fields)}')
    print(f'Results: {args.root/"summary.json"}')


if __name__=='__main__':
    main()
