"""Prepare, pilot, and run the frozen small-hydro certificate experiment."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .model import Hydro, draw_paths, fit_policy, path_indices, scenario_tree_oracle, support
from .certificate import CertificateEngine, bellman_values, generators, report_certificate
from .validation import audit_engines

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/certificate_v1'
# These probabilities belong to the sampling/oracle driver, never the certificate engine.
ORACLE_P = tuple(Fraction(x,20) for x in (7,12,9,13,8,11))
BUDGETS = (128,512,2048,8192,32768,131072)
PAIRS = tuple(sorted({*((n,32768) for n in BUDGETS),
                       *((131072,n) for n in BUDGETS)}))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    dest=OUT/'frozen'
    if (dest/'manifest.json').exists():
        return load_frozen()
    h=Hydro()
    fit=draw_paths([float(p) for p in ORACLE_P],2048,73101)
    validation=draw_paths([float(p) for p in ORACLE_P],1024,73202)
    parameters=fit_policy(fit,validation,h,dest)
    np.savez_compressed(dest/'fitting_paths.npz',fit=fit,validation=validation)
    gen=generators(h,fit)
    write_json(dest/'continuations.json',[g.payload() for g in gen])
    write_json(dest/'model.json',asdict(h))
    write_json(dest/'manifest.json',dict(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        fit_seed=73101,validation_seed=73202,n_fit=2048,n_validation=1024,
        policy='projected stagewise affine A1, full-horizon pathwise Adam',
        updates_per_candidate=800,batch_size=64,training_seeds=[73001,73002],
        learning_rates=[0.01,0.03],validation_every=40,
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    return h,parameters,gen


def load_frozen():
    dest=OUT/'frozen'
    manifest=json.loads((dest/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(dest/name)!=expected:
            raise RuntimeError(f'Frozen fitting artifact changed: {name}')
    h=Hydro()
    parameters=np.asarray(json.loads((dest/'policy.json').read_text())['parameters'])
    with np.load(dest/'fitting_paths.npz') as data:gen=generators(h,data['fit'])
    if [g.payload() for g in gen]!=json.loads((dest/'continuations.json').read_text()):
        raise RuntimeError('The frozen value generators did not reproduce.')
    return h,parameters,gen


def oracle(h,parameters,engines):
    v=bellman_values(h,ORACLE_P,'oracle_only')
    lp=scenario_tree_oracle(h,ORACLE_P)
    exact_optimum=v.values[0][h.initial_storage]
    if abs(lp['value']-float(exact_optimum))>1e-8:
        raise RuntimeError('Scenario-tree LP and exact Bellman oracle disagree.')
    probabilities=[]
    for path in support(h):
        mass=Fraction(1)
        for p,bit in zip(ORACLE_P,path):mass*=p if bit else 1-p
        probabilities.append(mass)
    assert sum(probabilities)==1
    costs=[sum(t['costs']) for t in engines[0].traces]
    exact_policy=sum(p*c for p,c in zip(probabilities,costs))
    true_engine=CertificateEngine(h,v,parameters)
    inner,_=true_engine.inner_exact([p*20 for p in ORACLE_P],20)
    if any(Fraction(int(y),20*true_engine.denominator)!=exact_optimum for y in inner):
        raise RuntimeError('Exact-value pathwise strong-duality audit failed.')
    references=[]
    for e in engines:
        inner,_=e.inner_exact([p*20 for p in ORACLE_P],20)
        population_lower=sum(p*Fraction(int(y),20*e.denominator) for p,y in zip(probabilities,inner))
        approximation_error=sum(max(abs(a-b) for a,b in zip(e.continuation.values[t],v.values[t]))
                                for t in range(1,h.T))
        references.append(dict(generator=e.continuation.name,exact_centering_lower=float(population_lower),
                               ideal_population_gap=float(exact_policy-population_lower),
                               summed_value_sup_error=float(approximation_error)))
    result=dict(optimum=float(exact_optimum),optimum_exact=str(exact_optimum),
                policy_cost=float(exact_policy),true_policy_loss=float(exact_policy-exact_optimum),
                loss_exact=str(exact_policy-exact_optimum),scenario_tree_lp=lp,
                support_size=len(probabilities),probabilities=[float(p) for p in probabilities],
                exact_value_strong_duality_all_paths=True,reference_gaps=references)
    write_json(OUT/'oracle.json',result)
    return result


def pilot():
    h,parameters,gen=prepare()
    engines=[CertificateEngine(h,g,parameters) for g in gen]
    truth=oracle(h,parameters,engines)
    sample=draw_paths([float(p) for p in ORACLE_P],max(BUDGETS),73303)
    evaluation=draw_paths([float(p) for p in ORACLE_P],32768,73404)
    evaluation_counts=np.bincount(path_indices(evaluation),minlength=2**h.T)
    rows=[]
    start=time.perf_counter()
    for n in BUDGETS:
        counts=sample[:n].sum(axis=0)
        for e in engines:
            c=e.calibrate(counts,n)
            rows.append(dict(generator=e.continuation.name,n_calibration=n,
                             **report_certificate(c,evaluation_counts)))
    audits=audit_engines(engines,sample[:2048].sum(axis=0),2048)
    dest=OUT/'pilot';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'certificates.csv',index=False)
    pd.DataFrame(audits).to_csv(dest/'continuous_inner_audit.csv',index=False)
    write_json(dest/'summary.json',dict(classification='engineering pilot, excluded from coverage experiment',
               calibration_seed=73303,evaluation_seed=73404,elapsed_seconds=time.perf_counter()-start,
               max_continuous_inner_error=max(r['absolute_error'] for r in audits),
               n_continuous_inner_checks=len(audits),oracle=truth))
    print(json.dumps(truth,indent=2),flush=True)
    print(pd.DataFrame(rows)[['generator','n_calibration','certificate','raw_gap',
                             'calibration_correction','evaluation_correction']].to_string(index=False),flush=True)


def formal(repetitions=200):
    h,parameters,gen=load_frozen()
    if not (OUT/'pilot/summary.json').exists():
        raise RuntimeError('Complete the separate engineering/solver pilot first.')
    engines=[CertificateEngine(h,g,parameters) for g in gen]
    truth=oracle(h,parameters,engines)
    dest=OUT/'formal';dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(
        repetitions=repetitions,calibration_budgets=BUDGETS,evaluation_budgets=BUDGETS,
        budget_pairs=PAIRS,alpha=0.025,delta=0.025,nominal_coverage=0.95,
        random_seed_rule='SeedSequence([73505, repetition, block]); block 0 calibration, 1 evaluation',
        paired_design='Nested prefixes within repetition; independent complete paths between blocks and repetitions',
        comparisons=['zero','affine','pwl'],conditioning_feature='singleton; stagewise independent binary shocks',
        band='Bernoulli Hoeffding moment band, union over six stages; uniform coefficient supremum',
        envelope='Minimum of two predeclared bounds: exact full-support supremum of the paired gap, and summed stage ranges; both plus centering and conversion caps. No evaluation observations or oracle probabilities enter either bound.',
        structural_envelope_comparison='Also report the certificate using only summed stage ranges',
        inner_solver='Exact arbitrary-precision integer DP; continuous PWL optimum by integral cell vertices',
        rounding_cap=1e-8,policy_fitting_manifest_sha256=sha(OUT/'frozen/manifest.json'),
        source_sha256={name:sha(ROOT/name) for name in ('model.py','certificate.py','run.py','validation.py')})
    # Write and verify the full protocol before drawing any formal calibration/evaluation paths.
    frozen=json.loads(json.dumps(protocol))
    if (dest/'protocol.json').exists():
        if json.loads((dest/'protocol.json').read_text())!=frozen:
            raise RuntimeError('Protocol/code changed; use a new output version rather than mixing runs.')
    else:
        write_json(dest/'protocol.json',protocol)
        write_json(dest/'freeze_timestamp.json',{'utc':datetime.now(timezone.utc).isoformat()})
    start=time.perf_counter()
    for rep in range(repetitions):
        rep_path=dest/f'rep_{rep:04d}.json'
        if rep_path.exists():continue
        calibration_paths=draw_paths([float(p) for p in ORACLE_P],max(BUDGETS),np.random.SeedSequence([73505,rep,0]))
        calibrated={}
        cal_counts=np.asarray([calibration_paths[:n].sum(axis=0) for n in BUDGETS])
        for j,n in enumerate(BUDGETS):
            for e in engines:
                calibrated[e.continuation.name,n]=e.calibrate(cal_counts[j],n)
        # All envelopes are frozen before this independent evaluation block is opened.
        evaluation_paths=draw_paths([float(p) for p in ORACLE_P],max(BUDGETS),np.random.SeedSequence([73505,rep,1]))
        ev_counts=np.asarray([np.bincount(path_indices(evaluation_paths[:n]),minlength=2**h.T) for n in BUDGETS])
        rows=[]
        for n_cal,n_ev in PAIRS:
            for e in engines:
                c=calibrated[e.continuation.name,n_cal]
                r=report_certificate(c,ev_counts[BUDGETS.index(n_ev)])
                counts=cal_counts[BUDGETS.index(n_cal)]
                rows.append(dict(repetition=rep,generator=e.continuation.name,n_calibration=n_cal,
                                 **r,covered=r['certificate']>=truth['true_policy_loss'],
                                 band_covered=bool(np.max(np.abs(counts/n_cal-np.asarray([float(p) for p in ORACLE_P])))<=c['rho']),
                                 true_policy_loss=truth['true_policy_loss'],policy_cost=truth['policy_cost']))
        # Compressed raw primitive paths and all memoized pathwise statistics are retained.
        payload=dict(calibration_paths=calibration_paths,evaluation_paths=evaluation_paths,
                     calibration_counts=cal_counts,evaluation_counts=ev_counts)
        for name in ('raw','corrected','inner_lower','policy_upper','beta'):
            payload[name]=np.asarray([[calibrated[e.continuation.name,n][name] for n in BUDGETS] for e in engines])
        np.savez_compressed(dest/f'rep_{rep:04d}.npz',**payload)
        write_json(rep_path,rows)
        if rep % 20 == 0 or rep == repetitions-1:
            print(f'Completed {rep+1}/{repetitions} repetitions in {time.perf_counter()-start:.1f}s',flush=True)
    all_rows=[]
    for rep in range(repetitions):all_rows.extend(json.loads((dest/f'rep_{rep:04d}.json').read_text()))
    frame=pd.DataFrame(all_rows)
    frame.to_csv(dest/'per_repetition.csv',index=False)
    write_json(dest/'COMPLETED.json',dict(repetitions=repetitions,reported_certificates=len(frame),
               memoized_inner_paths=repetitions*len(BUDGETS)*len(gen)*2**h.T,
               elapsed_this_invocation=time.perf_counter()-start,failures=0,
               completed_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','pilot','formal'))
    args=parser.parse_args()
    {'prepare':prepare,'pilot':pilot,'formal':formal}[args.action]()
