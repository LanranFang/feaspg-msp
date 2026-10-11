"""Paired old-data diagnostics and fresh frozen validation of tighter bounds."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .scaling import ROOT,CoupledHydro,Grid,ScalingEngine,policy_intervals
from .scaling_run import OUT as OLD_OUT,BUDGETS,N_EV,draw,codes,write
from .tightening import TightenedEngine

OUT=ROOT/'outputs/tightened_v3'
VARIANTS=('original','calibration_only','evaluation_only','tightened','centered_hoeffding_eb')
CHECKPOINTS=(40,200,400)


def load(R,T):
    source=OLD_OUT/f'r{R}_t{T}'
    old_protocol=json.loads((source/'protocol.json').read_text())
    for name,expected in old_protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen v2 source changed')
    with np.load(source/'fit.npz') as data:
        v=data['v'];parameters={k:data[f'checkpoint_{k}'] for k in CHECKPOINTS}
    h=CoupledHydro(R,T);g=Grid(h)
    engines={name:TightenedEngine(ScalingEngine(g,value)) for name,value in [('zero',np.zeros_like(v)),('pwl',v)]}
    reference=json.loads((source/'reference.json').read_text())
    return h,engines,parameters,reference


def evaluate_block(h,engines,parameters,reference,calibrated,evaluation,rep,old_npz=None):
    rows=[];payload={};evalcodes=codes(evaluation)
    for name,e in engines.items():
        traces={k:policy_intervals(h,e.base.v,p,evaluation) for k,p in parameters.items()}
        for n in BUDGETS:
            cal=calibrated[name,n]
            if old_npz is not None:
                inner=old_npz[f'{name}_{n}_inner'];seconds=0.
            else:inner,seconds=e.base.inner(cal['counts'],n,evalcodes)
            payload[f'{name}_{n}_inner']=inner
            for checkpoint,trace in traces.items():
                variants,samples=e.evaluate(cal,evaluation,trace,inner)
                ref=reference['policies'][str(checkpoint)]
                for variant,r in variants.items():
                    rows.append(dict(reservoirs=h.reservoirs,stages=h.stages,repetition=rep,
                        checkpoint=checkpoint,generator=name,n_calibration=n,n_evaluation=len(evaluation),
                        variant=variant,inner_seconds=seconds,policy_cost=ref['cost_midpoint'],
                        loss_lower=ref['loss_lower'],loss_upper=ref['loss_upper'],
                        covers_reference_upper=r['certificate']>=ref['loss_upper'],
                        below_reference_lower=r['certificate']<ref['loss_lower'],**r))
                for key,val in samples.items():payload[f'{name}_{n}_{checkpoint}_{key}']=val
    return rows,payload


def pilot():
    dest=OUT/'pilot';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for R in (1,2,3):
        for T in (6,12,18):
            h,engines,parameters,ref=load(R,T)
            for rep in range(3):
                with np.load(OLD_OUT/f'r{R}_t{T}/rep_{rep}.npz') as old:
                    calpaths=old['calibration_paths']
                    calibrated={(name,n):e.calibrate(calpaths[:n].sum(0),n) for name,e in engines.items() for n in BUDGETS}
                    block,_=evaluate_block(h,engines,parameters,ref,calibrated,old['evaluation_paths'],rep,old)
                rows.extend(block)
            print(f'Paired diagnostic R={R} T={T} done',flush=True)
    pd.DataFrame(rows).to_csv(dest/'results.csv',index=False)
    write(dest/'DESIGN.json',dict(classification='paired old-data engineering diagnostic; not fresh coverage validation',
        variants=VARIANTS,rows=len(rows),source='frozen scaling_v2 raw trajectories; unchanged fitting and inner problems'))


def formal(repetitions=200):
    h,engines,parameters,reference=load(3,18)
    dest=OUT/'formal';dest.mkdir(parents=True,exist_ok=True)
    if not (OUT/'pilot/results.csv').exists():raise RuntimeError('Complete implementation diagnostics before formal validation')
    source_names=('scaling.py','scaling_inner.cpp','scaling_run.py','tightening.py','tightening_run.py')
    protocol=dict(classification='fresh fixed-protocol validation conditional on the unchanged fitted policies',
        reservoirs=3,stages=18,stagewise_independent_binary_shocks=True,
        repetitions=repetitions,calibration_budgets=BUDGETS,n_evaluation=N_EV,
        checkpoints=CHECKPOINTS,generators=['zero','pwl'],variants=VARIANTS,
        primary_variant='tightened',primary_checkpoint=400,alpha=.025,delta=.025,
        confidence='95% marginal per fixed setting; no selection of the smallest reported bound',
        calibration='joint independent-coordinate subGaussian Euclidean ball; querywise paired contrast correction',
        evaluation='Maurer-Pontil 2009 Theorem4 on full raw+random correction; exact dyadic sample moments',
        envelope='minimum of three deterministic bounds, including adversarial residual DP, frozen before evaluation',
        seeds='SeedSequence([95731,3,18,block,rep]); block0 calibration, block1 evaluation',
        independent_from_previous_seeds=True,reference='retained verified continuous-loss interval; no exact loss point claim',
        fitting_sha256=hashlib.sha256((OLD_OUT/'r3_t18/fit.npz').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names})
    normalized=json.loads(json.dumps(protocol))
    if (dest/'protocol.json').exists():
        if json.loads((dest/'protocol.json').read_text())!=normalized:raise RuntimeError('Frozen formal protocol changed')
    else:
        write(dest/'protocol.json',protocol)
        write(dest/'freeze_timestamp.json',dict(utc=datetime.now(timezone.utc).isoformat()))
    start=time.perf_counter()
    for rep in range(repetitions):
        if (dest/f'rep_{rep:04d}.json').exists():continue
        calpaths=draw(h,max(BUDGETS),np.random.SeedSequence([95731,3,18,0,rep]))
        calibrated={(name,n):e.calibrate(calpaths[:n].sum(0),n) for name,e in engines.items() for n in BUDGETS}
        # No evaluation path is drawn or opened until all envelopes are frozen.
        evaluation=draw(h,N_EV,np.random.SeedSequence([95731,3,18,1,rep]))
        rows,payload=evaluate_block(h,engines,parameters,reference,calibrated,evaluation,rep)
        payload.update(calibration_paths=calpaths,evaluation_paths=evaluation)
        np.savez_compressed(dest/f'rep_{rep:04d}.npz',**payload)
        write(dest/f'rep_{rep:04d}.json',rows)
        if rep%10==0 or rep==repetitions-1:
            print(f'Fresh validation {rep+1}/{repetitions}: {time.perf_counter()-start:.1f}s',flush=True)
    rows=[]
    for rep in range(repetitions):rows.extend(json.loads((dest/f'rep_{rep:04d}.json').read_text()))
    pd.DataFrame(rows).to_csv(dest/'results.csv',index=False)
    write(dest/'COMPLETED.json',dict(repetitions=repetitions,rows=len(rows),failures=0,
        seconds_this_invocation=time.perf_counter()-start,utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['pilot','formal'])
    args=parser.parse_args();{'pilot':pilot,'formal':formal}[args.action]()
