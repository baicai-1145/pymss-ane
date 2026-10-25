"""Verify whether ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR actually fires on this machine.

Compile-only probe (plus one tiny execute). Espresso compile work may run
inside a root XPC service; if so, client-side env vars cannot reach it without
sudo. A silent result is a boundary observation, not proof the channel never
exists. No throughput claims. Control child runs with the var unset.
"""
from __future__ import annotations
import argparse
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRUNE = {'Library', 'Pictures', 'Music', 'Movies', 'Applications', '.Trash',
         'Developer', 'Public', 'node_modules', '.git'}

SEARCH_ROOTS = None  # set in main after scratch creation


def find_candidates(roots, depth=4):
    hits = []
    for root in (roots or []):
        root = Path(root)
        if not root.exists():
            continue
        name = root.name.lower()
        if 'anecir' in name or name == 'net.plist':
            hits.append(str(root))
        stack = [(root, 0)]
        while stack:
            directory, level = stack.pop()
            if level >= depth:
                continue
            try:
                entries = list(directory.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                lowered = entry.name.lower()
                if 'anecir' in lowered or lowered == 'net.plist':
                    hits.append(str(entry))
                if (entry.is_dir() and not entry.is_symlink()
                        and entry.name not in PRUNE
                        and not str(entry).startswith(('/System', '/usr', '/Volumes'))):
                    stack.append((entry, level + 1))
    return sorted(set(hits))


def child(scratch: Path):
    import warnings
    import numpy as np
    import aneforge as af
    pre = find_candidates(SEARCH_ROOTS)
    q = af.input((1, 8, 64, 64))
    kt = af.input((1, 8, 64, 64))
    v = af.input((1, 8, 64, 64))
    score = (q @ kt) * (64 ** -0.5)
    e = ((score - score.amax(-1)) * float(np.log2(np.e))).exp2()
    out = (e @ v) / (e @ np.ones((64, 1), np.float16))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = af.compile(out, opt=0, build_dir=str(scratch / 'program'))
    for name, _ in net._inputs:
        net.input_view(name).fill(np.float16(0.1))
    net.execute()
    time.sleep(0.5)  # give any async dumper a moment
    post = find_candidates(SEARCH_ROOTS)
    print(json.dumps(dict(pid=os.getpid(),
                          env_value=os.environ.get('ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR'),
                          pre=pre, post=post,
                          new=[p for p in post if p not in pre])), flush=True)
    net.release()


def brief(value, depth=0):
    if isinstance(value, dict):
        if depth >= 4:
            return f'<dict:{len(value)}>'
        return {str(k): brief(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        if depth >= 4:
            return f'<list:{len(value)}>'
        out = [brief(v, depth + 1) for v in value[:40]]
        if len(value) > 40:
            out.append(f'<+{len(value) - 40} more>')
        return out
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    return text if len(text) < 100 else text[:100] + '…'


def main():
    global SEARCH_ROOTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT / '.tmp/ane_compiler_ir')
    args = ap.parse_args()
    if args.child:
        child(Path.cwd())
        return
    args.root.mkdir(parents=True, exist_ok=True)
    children = {}
    for mode in ['control', 'dump']:
        scratch = args.root / mode
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
    SEARCH_ROOTS = [args.root, Path('/tmp'), Path(tempfile.gettempdir()), Path.home(), ROOT / '.tmp']
    logpath = args.root / 'compile_events.ndjson'
    cmd = ['/usr/bin/log', 'stream', '--style', 'ndjson', '--level', 'debug', '--signpost',
           '--predicate', 'eventMessage CONTAINS[c] "anecir" OR eventMessage CONTAINS[c] "in-memory" '
           'OR processImagePath CONTAINS "ANECompiler"']
    with logpath.open('w') as f:
        stream = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            for mode in ['control', 'dump']:
                env = dict(os.environ)
                env.pop('ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR', None)
                if mode == 'dump':
                    env['ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR'] = '1'
                try:
                    proc = subprocess.run([sys.executable, __file__, '--child'], env=env,
                                          capture_output=True, text=True, timeout=90,
                                          cwd=str(args.root / mode))
                    children[mode] = dict(returncode=proc.returncode,
                                          stdout=proc.stdout[-2000:], stderr=proc.stderr[-1500:])
                except subprocess.TimeoutExpired:
                    children[mode] = dict(error='child exceeded 90s window')
                print(json.dumps({mode: children[mode]}), flush=True)
        finally:
            if stream.poll() is None:
                stream.send_signal(signal.SIGINT)
                try:
                    stream.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stream.kill()
                    stream.wait()
    plists = {}
    for path in find_candidates(SEARCH_ROOTS, depth=6):
        p = Path(path)
        if p.is_file() and p.name == 'net.plist':
            try:
                with p.open('rb') as fh:
                    plists[str(p)] = brief(plistlib.load(fh))
            except Exception as ex:
                plists[str(p)] = f'parse error: {type(ex).__name__}: {str(ex)[:200]}'
        elif p.is_dir() and 'anecir' in p.name.lower():
            plists[str(p)] = [q.name for q in p.iterdir()][:30]
    events = []
    for line in logpath.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = str(event.get('eventMessage', ''))
        if 'anecir' in message.lower() or 'in-memory' in message.lower():
            events.append(dict(pid=event.get('processID'), message=message[:500]))
    summary = dict(children=children, discovered=plists, anecir_log_events=events,
                   log_path=str(logpath), note='control must precede dump; new entries are the signal')
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Discovered entries: {list(plists)}')
    print(f'anecir-related log events: {len(events)}')
    print(f'Results: {args.root / "summary.json"}')


if __name__ == '__main__':
    main()
