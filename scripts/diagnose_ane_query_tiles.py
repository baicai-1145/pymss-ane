"""Query-row tiling as a graph-expression experiment, not proof of FlashAttention.

Every query tile retains the full key axis (same softmax semantics); no online
key merge. Compare full-core execute at opt=0, no host copies during timing.
Smaller individual score tensors do not establish lower peak active memory or
SRAM residency: native scheduling is not observed. Production graph is unchanged.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import aneforge as af
from diagnose_ane_utilization import memory_counters, stat
from diagnose_ane_fusion_boundary import cosine, numpy_lane

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rev'))
from e5_parser import parse as parse_e5


def graph_for(B,H,S,D,query_tile):
    q=af.input((B,H,S,D))
    kt=af.input((B,H,D,S))
    v=af.input((B,H,S,D))
    outputs=[]
    for start in range(0,S,query_tile):
        size=min(query_tile,S-start)
        qt=q if size==S else q.slice_by_size([0,0,start,0],[B,H,size,D])
        score=(qt@kt)*(D**-0.5)
        e=((score-score.amax(-1))*float(np.log2(np.e))).exp2()
        outputs.append((e@v)/(e@np.ones((S,1),np.float16)))
    return outputs[0] if len(outputs)==1 else af.concat(outputs,axis=2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq',type=int,default=256)
    ap.add_argument('--batches',default='1,8,60')
    ap.add_argument('--tiles',default='64,128')
    ap.add_argument('--rounds',type=int,default=15)
    ap.add_argument('--output',type=Path,default=ROOT/'.tmp/ane_query_tiles_S256.json')
    args=ap.parse_args()
    batches=[int(b) for b in args.batches.split(',')]
    requested=[int(t) for t in args.tiles.split(',')]
    assert args.seq>0 and all(b>0 for b in batches) and all(t>0 for t in requested)
    H,D,S=8,64,args.seq
    rng=np.random.default_rng(179)
    rows=[]
    for B in batches:
        specs=[('baseline',S),*[(f'q{t}',t) for t in requested if t<S]]
        feeds=[rng.normal(0,1,shape).astype(np.float16) for shape in [(B,H,S,D),(B,H,D,S),(B,H,S,D)]]
        nets,outputs,artifacts,failures={},{},{},{}
        try:
            for label,tile in specs:
                net=None
                directory=ROOT/f'.tmp/ane_query_tiles/B{B}_S{S}_D{D}_{label}'
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        net=af.compile(graph_for(B,H,S,D,tile),opt=0,build_dir=str(directory))
                    for (name,shape),data in zip(net._inputs,feeds,strict=True):
                        assert tuple(shape)==data.shape
                        np.copyto(net.input_view(name),data)
                    net.execute()
                    out=net.output_view().copy()
                    assert np.isfinite(out).all()
                    outputs[label]=out
                    e5=[]
                    for p in directory.rglob('H16G.e5'):
                        parsed=parse_e5(p.read_bytes(),str(p))
                        e5.append(dict(path=str(p),sha256=parsed['sha256'],parser_td_blocks=len(parsed['td_blocks'])))
                    artifacts[label]=dict(build_dir=str(directory),
                                          mil_sha256=hashlib.sha256((directory/'model.mil').read_bytes()).hexdigest(),
                                          frontend_nodes=net.n_ops,
                                          query_tile=tile,
                                          largest_individual_score_node_bytes=2*B*H*min(tile,S)*S,
                                          score_node_count=(S+tile-1)//tile,e5_candidates=e5)
                    nets[label]=net
                    print(f'Compiled B={B} {label}: {net.n_ops} frontend nodes',flush=True)
                except Exception as ex:
                    failures[label]=f'{type(ex).__name__}: {str(ex)[:350]}'
                    print(f'B={B} {label} FAIL: {failures[label]}',flush=True)
                    if net is not None:
                        net.release()
                    if label=='baseline':
                        raise
            parity={}
            reference=numpy_lane(*feeds)
            for label,out in outputs.items():
                rel=float(np.linalg.norm((out.astype(np.float64)-outputs['baseline'].astype(np.float64)).ravel())/
                          np.linalg.norm(outputs['baseline'].astype(np.float64).ravel()))
                c=cosine(out,outputs['baseline'])
                lane=cosine(out[:1,:1],reference)
                parity[label]=dict(cosine=c,relative_l2=rel,bitwise_equal=bool(np.array_equal(out,outputs['baseline'])),
                                   numpy_lane_cosine=lane)
                assert c>0.9999 and rel<0.003 and lane>0.999,parity
            for _ in range(3):
                for net in nets.values():
                    net.execute()
            before=memory_counters()
            samples=[]
            order_rng=random.Random(181)
            for r in range(args.rounds):
                order=list(nets)
                order_rng.shuffle(order)
                for label in order:
                    t=time.perf_counter_ns()
                    nets[label].execute()
                    samples.append(dict(round=r,label=label,execute_ms=(time.perf_counter_ns()-t)/1e6))
            after=memory_counters()
            summary={label:stat([s['execute_ms'] for s in samples if s['label']==label]) for label in nets}
            paired={}
            for label in nets:
                if label=='baseline':
                    continue
                delta=[next(s['execute_ms'] for s in samples if s['round']==r and s['label']==label)-next(s['execute_ms'] for s in samples if s['round']==r and s['label']=='baseline') for r in range(args.rounds)]
                paired[label]=dict(**stat(delta),negative_pairs=sum(d<0 for d in delta),pairs=len(delta))
            row=dict(B=B,H=H,S=S,D=D,parity=parity,summary=summary,paired_delta_ms=paired,
                     samples=samples,artifacts=artifacts,failures=failures,
                     paging_delta={k:after[k]-before[k] for k in before})
            rows.append(row)
            print(f'B={B} numerical gates PASS; timing={summary}; paired={paired}',flush=True)
            print(f'B={B} paging={row["paging_delta"]}',flush=True)
        finally:
            for net in nets.values():
                net.release()
            gc.collect()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(seq=S,head_dim=D,heads=H,raw_lowering=True,rows=rows),indent=2))
    print(f'Raw samples: {args.output}',flush=True)


if __name__=='__main__':
    main()
