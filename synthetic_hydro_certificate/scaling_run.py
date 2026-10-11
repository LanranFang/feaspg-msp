"""Run all predeclared cells of the coupled-hydro scaling engineering pilot."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .scaling import (ROOT,SCALE,CoupledHydro,Grid,ScalingEngine,fit_checkpoints,
    policy_intervals,paths_from_codes,oracle_probabilities,exact_path_masses,
    continuous_tree_lp,continuous_inner_audit,UP,DOWN,isum)

OUT=ROOT/'outputs/scaling_v2'
BUDGETS=(2048,8192,32768)
N_EV=8192
REPETITIONS=3


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def draw(h,n,seed):
    return (np.random.default_rng(seed).random((n,h.stages))<oracle_probabilities(h)/20).astype(np.uint8)


def codes(paths):return paths@(2**np.arange(paths.shape[1]-1,-1,-1))


def reference(h,g,checkpoints,v,dest):
    """References use oracle probabilities; none enter actual certification."""
    p=oracle_probabilities(h);weights=exact_path_masses(h)
    all_codes=np.arange(2**h.stages,dtype=np.int64)
    start=time.perf_counter()
    integer=g.bellman(p,20,exact=True)
    upper=Fraction(int(integer[0][g.initial]),SCALE*20**h.stages)
    engine=ScalingEngine(g,v)
    inner,seconds=engine.inner(p,20,all_codes)
    lower=Fraction(sum(int(a)*int(b) for a,b in zip(inner,weights)),SCALE*20**(h.stages+1))
    if lower>upper:raise RuntimeError('Reference weak duality violated')
    oracle=dict(optimum_lower=float(DOWN(float(lower))),optimum_upper=float(UP(float(upper))),
        optimum_lower_exact=str(lower),optimum_upper_exact=str(upper),
        reference_interval_width=float(upper-lower),inner_seconds=seconds,
        reference_type='exact-expectation information-relaxation lower bound; feasible integer-state policy upper bound')
    if h.stages<=6:
        oracle['continuous_tree_lp']=continuous_tree_lp(h)
        val=oracle['continuous_tree_lp']['value']
        if not float(lower)-1e-7<=val<=float(upper)+1e-7:raise RuntimeError('Continuous oracle outside reference bracket')
        # Independent numerical LP benchmark for small instances; retain the
        # rigorous rational bracket separately rather than relabelling it exact.
    policy_costs={}
    for checkpoint,parameters in checkpoints.items():
        total_low=0.;total_high=0.;max_width=0.
        for start_i in range(0,len(all_codes),16384):
            sl=slice(start_i,start_i+16384)
            paths=paths_from_codes(all_codes[sl],h.stages)
            cost,_=policy_intervals(h,np.zeros_like(v),parameters,paths)
            bounds=isum((cost[0],cost[1]))
            probability=np.asarray(weights[sl]/(20**h.stages),dtype=float)
            total_low+=float(probability@np.maximum(bounds[0],0))
            total_high+=float(probability@bounds[1])
            max_width=max(max_width,float(np.max(bounds[1]-bounds[0])))
        # Standard floating-point dot/sum bound, using positive weights and a
        # deterministic total-cost cap; factor 4 dominates gamma_n for n<=2**18.
        cost_cap=float(sum(h.prices)*3*h.reservoirs+h.stages*(h.release@h.operating)+12*h.reservoirs)
        integration_guard=4*np.finfo(float).eps*(2**h.stages+2*h.stages+20)*cost_cap
        lo=max(0,float(DOWN(total_low-integration_guard)));hi=float(UP(total_high+integration_guard))
        policy_costs[checkpoint]=dict(cost_lower=lo,cost_upper=hi,cost_midpoint=(lo+hi)/2,
            loss_lower=max(0,lo-oracle['optimum_upper']),loss_upper=hi-oracle['optimum_lower'],
            integration_guard=integration_guard,max_trace_interval=max_width)
    oracle['policies']=policy_costs;oracle['seconds']=time.perf_counter()-start
    np.savez_compressed(dest/'reference_paths.npz',inner_numerators=inner,all_codes=all_codes)
    write(dest/'reference.json',oracle)
    return oracle


def run_cell(R,T):
    h=CoupledHydro(R,T);dest=OUT/f'r{R}_t{T}';dest.mkdir(parents=True,exist_ok=True)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scaling.py','scaling_inner.cpp','scaling_run.py')}
    protocol=dict(classification='engineering pilot, not a formal coverage experiment',model=asdict(h),
        reservoirs=list(range(1,4)),stages=[6,12,18],n_fit=2048,n_validation=1024,
        training_updates=400,checkpoints=[40,200,400],continuations=['zero','pwl'],
        calibration_budgets=BUDGETS,n_evaluation=N_EV,repetitions=REPETITIONS,alpha=.025,delta=.025,
        envelope='sum of within-shock stage-objective ranges on the full relaxed domain; no path enumeration',
        reference='continuous optimum bracket, with independent small-instance continuous LP',
        seeds='SeedSequence([84620,R,T,block,rep]); fit=0, validation=1, calibration=2, evaluation=3, audit=4',
        source_sha256=hashes)
    normalized=json.loads(json.dumps(protocol))
    if (dest/'protocol.json').exists():
        if json.loads((dest/'protocol.json').read_text())!=normalized:raise RuntimeError('Frozen scaling pilot changed; use a new version')
    else:write(dest/'protocol.json',protocol)
    if (dest/'COMPLETED.json').exists():
        print(f'Already complete R={R} T={T}',flush=True);return
    start=time.perf_counter();g=Grid(h)
    seed=lambda block,rep=0:np.random.SeedSequence([84620,R,T,block,rep])
    if (dest/'fit.npz').exists():
        with np.load(dest/'fit.npz') as d:
            fit=d['fit'];validation=d['validation'];v=d['v'];checkpoints={str(k):d[f'checkpoint_{k}'] for k in (40,200,400)}
    else:
        fit=draw(h,2048,seed(0));validation=draw(h,1024,seed(1))
        checkpoints,history=fit_checkpoints(h,fit,validation,84621+R*100+T)
        v,errors=g.fit_continuation(fit)
        np.savez_compressed(dest/'fit.npz',fit=fit,validation=validation,v=v,
                            **{f'checkpoint_{k}':val for k,val in checkpoints.items()})
        write(dest/'fitting.json',dict(checkpoints=history,value_grid_rmse=errors,seconds=time.perf_counter()-start))
    print(f'R={R} T={T}: fitted; {len(g.states)} states, {len(g.a)} state/action vertices',flush=True)
    engines={'zero':ScalingEngine(g,np.zeros_like(v)),'pwl':ScalingEngine(g,v)}
    audit_paths=draw(h,4,seed(4));audit_counts=np.full(T,1024,dtype=np.int64)
    audits=[]
    for name,e in engines.items():
        exact,_=e.inner(audit_counts,2048,codes(audit_paths))
        for i,path in enumerate(audit_paths):
            audit=continuous_inner_audit(e,audit_counts,2048,path)
            error=max(abs(audit['value']-exact[i]/(2048*SCALE)),abs(audit['bound']-exact[i]/(2048*SCALE)))
            if error>1e-7:raise RuntimeError('Continuous global audit disagreement')
            audits.append(dict(generator=name,path=path.tolist(),absolute_error=error,**audit))
    write(dest/'continuous_audits.json',audits)
    oracle=json.loads((dest/'reference.json').read_text()) if (dest/'reference.json').exists() else reference(h,g,checkpoints,v,dest)
    print(f'R={R} T={T}: V* in [{oracle["optimum_lower"]:.4f}, {oracle["optimum_upper"]:.4f}], reference {oracle["seconds"]:.1f}s',flush=True)
    rows=[]
    for rep in range(REPETITIONS):
        calpaths=draw(h,max(BUDGETS),seed(2,rep))
        calibrated={(name,n):e.calibrate(calpaths[:n].sum(axis=0),n) for name,e in engines.items() for n in BUDGETS}
        # All calibration objects and envelopes frozen before evaluation draw.
        evaluation=draw(h,N_EV,seed(3,rep));evalcodes=codes(evaluation)
        payload=dict(calibration_paths=calpaths,evaluation_paths=evaluation)
        for name,e in engines.items():
            traces={k:policy_intervals(h,e.v,par,evaluation) for k,par in checkpoints.items()}
            for n in BUDGETS:
                cal=calibrated[name,n];inner,seconds=e.inner(cal['counts'],n,evalcodes)
                payload[f'{name}_{n}_inner']=inner
                for checkpoint,trace in traces.items():
                    report=e.evaluate(cal,evaluation,trace,inner)
                    ref=oracle['policies'][checkpoint]
                    row=dict(reservoirs=R,stages=T,repetition=rep,generator=name,checkpoint=int(checkpoint),
                        n_calibration=n,n_evaluation=N_EV,inner_seconds=seconds,
                        policy_cost=ref['cost_midpoint'],loss_lower=ref['loss_lower'],loss_upper=ref['loss_upper'],
                        covered_reference_upper=report['certificate']>=ref['loss_upper'],
                        below_reference_lower=report['certificate']<ref['loss_lower'],
                        **report)
                    row['certificate_fraction_cost']=row['certificate']/row['policy_cost']
                    row['excess_lower']=row['certificate']-row['loss_upper']
                    row['excess_upper']=row['certificate']-row['loss_lower']
                    rows.append(row)
        np.savez_compressed(dest/f'rep_{rep}.npz',**payload)
        pd.DataFrame(rows).to_csv(dest/'results.csv',index=False)
        print(f'R={R} T={T}: repetition {rep+1}/{REPETITIONS}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    write(dest/'COMPLETED.json',dict(rows=len(rows),elapsed_seconds=time.perf_counter()-start,
        utc=datetime.now(timezone.utc).isoformat(),max_continuous_error=max(a['absolute_error'] for a in audits)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reservoirs',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--stages',type=int,nargs='+',default=[6,12,18])
    args=parser.parse_args()
    cells=[(r,t) for r in args.reservoirs for t in args.stages]
    cells.sort(key=lambda c:(c!=(3,18),c))
    for r,t in cells:run_cell(r,t)


if __name__=='__main__':main()
