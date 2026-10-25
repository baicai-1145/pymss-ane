"""Calibration and output-observability perturbation for ANE attention.

All core variants use raw lowering (compile opt=0 or compile_multi); one-output
multi validates that the multi-output wrapper is not itself the comparison.
Additional outputs can change allocation, scheduling and native optimization.
Their time delta is NOT a DRAM price or proof of SRAM residency. Identity timing
is a calibration workload, not a pure measurement of an invariant dispatch tax.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import aneforge as af
from aneforge._compile import compile_multi
from diagnose_ane_utilization import memory_counters, stat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rev'))
from e5_parser import parse as parse_e5


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--T',type=int,default=256)
    ap.add_argument('--rounds',type=int,default=15)
    ap.add_argument('--output',type=Path,default=ROOT/'.tmp/ane_fusion_ports_T256.json')
    args=ap.parse_args()
    rng=np.random.default_rng(151)
    rows=[]
    for direction,B,S in [('time',60,args.T),('freq',args.T,60)]:
        H,D=8,64
        q,kt,v=[af.input(shape) for shape in [(B,H,S,D),(B,H,D,S),(B,H,S,D)]]
        score=(q@kt)*(D**-0.5)
        e=((score-score.amax(-1))*float(np.log2(np.e))).exp2()
        out=(e@v)/(e@np.ones((S,1),np.float16))
        tiny=q.slice_by_size([0,0,0,0],[1,1,1,1])
        identity_tiny=af.input((1,1,1,1))*1.0
        identity_score=af.input((B,H,S,S))*1.0
        specs=[('mono_raw',[out],False),('multi_single',[out],True),
               ('multi_tiny_port',[out,tiny],True),('multi_score_port',[out,score],True),
               ('identity_tiny',[identity_tiny],False),('identity_score',[identity_score],False)]
        q_np=rng.normal(0,0.1,(B,H,S,D)).astype(np.float16)
        kt_np=rng.normal(0,0.1,(B,H,D,S)).astype(np.float16)
        v_np=rng.normal(0,0.1,(B,H,S,D)).astype(np.float16)
        identity_np=rng.normal(0,0.1,(B,H,S,S)).astype(np.float16)
        handles={}
        models=[]
        artifacts={}
        outputs={}
        try:
            for label,outs,multi in specs:
                directory=ROOT/f'.tmp/ane_fusion_ports/T{args.T}_{direction}/{label}'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model=compile_multi(outs,build_dir=str(directory)) if multi else af.compile(outs[0],opt=0,build_dir=str(directory))
                models.append(model)
                if multi:
                    prog=model.prog
                    inputs=[(name,t.shape) for t,name in model.input_ports]
                    output_ports=[(name,t.shape) for t,name in model.output_ports]
                else:
                    prog=model._prog
                    inputs=list(model._inputs)
                    output_ports=[(model._out_name,model._out_shape)]
                feeds=[np.array([[[[0.25]]]],np.float16)] if label=='identity_tiny' else ([identity_np] if label=='identity_score' else [q_np,kt_np,v_np])
                assert len(inputs)==len(feeds)
                for (name,shape),array in zip(inputs,feeds,strict=True):
                    assert tuple(shape)==array.shape,(label,shape,array.shape)
                    np.copyto(prog.input_view(name),array)
                handles[label]=(prog,inputs,output_ports)
                for _ in range(4):
                    prog.execute()
                outputs[label]=prog.output_view(output_ports[0][0]).copy()
                e5s=[]
                for path in directory.rglob('H16G.e5'):
                    parsed=parse_e5(path.read_bytes(),str(path))
                    e5s.append(dict(path=str(path),sha256=parsed['sha256'],
                                     parser_td_blocks=len(parsed['td_blocks']),
                                     section_names=[s['name'] for s in parsed['sections']]))
                artifacts[label]=dict(build_dir=str(directory),
                                      mil_sha256=hashlib.sha256((directory/'model.mil').read_bytes()).hexdigest(),
                                      input_shapes=[list(shape) for _,shape in inputs],
                                      output_shapes=[list(shape) for _,shape in output_ports],
                                      output_bytes=sum(int(np.prod(shape))*2 for _,shape in output_ports),
                                      output_input_same_virtual_pointer=bool(prog.output_view(output_ports[0][0]).ctypes.data==prog.input_view(inputs[0][0]).ctypes.data),
                                      e5=e5s)
                print(f'Compiled {direction}/{label}',flush=True)
            parity={}
            for label in ['multi_single','multi_tiny_port','multi_score_port']:
                np.testing.assert_array_equal(outputs['mono_raw'],outputs[label])
                parity[label]=dict(main_output_bitwise_equal=True)
            np.testing.assert_array_equal(outputs['identity_tiny'],np.array([[[[0.25]]]],np.float16))
            np.testing.assert_array_equal(outputs['identity_score'],identity_np)
            print(f'{direction} all main-output/identity bitwise gates PASS',flush=True)
            # Later compilations may page out state warmed earlier. Re-warm
            # every case only after the complete comparison set is compiled.
            for _ in range(3):
                for prog, _, _ in handles.values():
                    prog.execute()
            before=memory_counters()
            samples=[]
            order_rng=random.Random(163)
            for r in range(args.rounds):
                order=list(handles)
                order_rng.shuffle(order)
                for label in order:
                    t=time.perf_counter_ns()
                    handles[label][0].execute()
                    samples.append(dict(round=r,label=label,execute_ms=(time.perf_counter_ns()-t)/1e6))
            after=memory_counters()
            summary={label:stat([s['execute_ms'] for s in samples if s['label']==label]) for label in handles}
            paired={}
            for label in ['multi_single','multi_tiny_port','multi_score_port']:
                reference='mono_raw' if label=='multi_single' else 'multi_single'
                differences=[next(s['execute_ms'] for s in samples if s['round']==r and s['label']==label)-next(s['execute_ms'] for s in samples if s['round']==r and s['label']==reference) for r in range(args.rounds)]
                paired[label+'_minus_'+reference]=stat(differences)
            row=dict(direction=direction,B=B,H=H,S=S,D=D,summary=summary,paired_deltas=paired,
                     parity=parity,artifacts=artifacts,samples=samples,
                     paging_delta={k:after[k]-before[k] for k in before})
            rows.append(row)
            for label,values in summary.items():
                print(f'{direction}/{label}: min={values["min_ms"]:.3f} median={values["median_ms"]:.3f} max={values["max_ms"]:.3f}ms',flush=True)
            print(f'{direction}: paired deltas={paired}; paging={row["paging_delta"]}',flush=True)
        finally:
            for model in models:
                model.release()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(T=args.T,raw_lowering=True,post_compile_warmup_passes=3,rows=rows),indent=2))
    print(f'Raw samples: {args.output}',flush=True)


if __name__=='__main__':
    main()
