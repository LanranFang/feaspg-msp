"""Independent arithmetic, finite-sample, and pathwise refinement checks."""
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.stats import binom

from synthetic_hydro_certificate.scaling import (
    SCALE,CoupledHydro,Grid,ScalingEngine,paths_from_codes,policy_intervals,
)
from synthetic_hydro_certificate.tightening import TightenedEngine,exact_float_moments,bernstein


def fixture(R=2,T=3):
    h=CoupledHydro(R,T);g=Grid(h)
    fit=(np.random.default_rng(38).random((256,T))<.5).astype(np.uint8)
    v,_=g.fit_continuation(fit)
    e=TightenedEngine(ScalingEngine(g,v,threads=2))
    parameters=np.zeros((T,R,R+1));parameters[:,:,0]=.3
    for r in range(R):parameters[:,r,r+1]=.61
    return e,parameters


def test_exact_moments_against_fraction_arithmetic():
    for values in ([0.,0.,0.],[.125,.5,1.,9.],
                   [0.,np.nextafter(0.,1.),1e-100,.1,1.,1e3]):
        x=np.asarray(values)
        fractions=list(map(Fraction,values));mean=sum(fractions)/len(x)
        variance=sum((a-mean)**2 for a in fractions)/(len(x)-1)
        assert exact_float_moments(x)==(mean,variance)
    base=np.asarray([.125,.5,1.,9.]);a=bernstein(base,10)
    b=bernstein(base+16,10)
    assert a['evaluation']==b['evaluation']
    assert a['variance']==b['variance']


def test_joint_ball_by_exact_binomial_enumeration():
    # Direct distribution-level check, independent of the implemented formula
    # derivation; no normal/chi-square approximation.
    e,_=fixture(T=3)
    for n in (4,8,16):
        for delta in (.025,.1,.25):
            radius=e.calibrate(np.full(3,n//2),n,delta)['radius']
            p=np.asarray([.2,.5,.8]);failure=0.
            for counts in product(range(n+1),repeat=3):
                if np.linalg.norm(np.asarray(counts)/n-p)>radius:
                    failure+=float(np.prod(binom.pmf(counts,n,p)))
            assert failure<=delta+1e-13


def test_adversarial_range_matches_exhaustive_small_problem():
    e,_=fixture(R=2,T=2);h=e.h;counts=[19,43];n=64
    b=e.structural_bounds(counts,n);values=[]
    def value(t,state):
        return sum(Fraction(int(e.base.v[t,r,int(s)]),SCALE) for r,s in enumerate(state))
    def recurse(t,state,total):
        if t==h.stages:values.append(total);return
        for action in product(*[range(min(int(s),int(c))+1) for s,c in zip(state,h.release)]):
            q=Fraction(0)
            next_states=[]
            for wet in (0,1):
                nxt=np.minimum(h.capacity,np.asarray(state)-action+h.inflow*wet)
                cost=int(h.prices[t])*max(int(h.demand(wet))-sum(action),0)
                cost+=sum(Fraction(r,8)*action[r] for r in range(h.reservoirs))
                if t==h.stages-1:cost+=4*sum(max(3-int(s),0) for s in nxt)
                q+=Fraction(counts[t] if wet else n-counts[t],n)*(cost+value(t+1,nxt))
                next_states.append(nxt)
            residual=q-value(t,state)
            for nxt in next_states:recurse(t+1,nxt,total+residual)
    recurse(0,h.initial,Fraction(0))
    assert min(values)==Fraction(b['residual_min_numerator'],b['denominator'])
    assert max(values)==Fraction(b['residual_max_numerator'],b['denominator'])


def test_paired_perturbation_over_future_aware_paths():
    e,parameters=fixture();paths=paths_from_codes(np.arange(8),3)
    trace=policy_intervals(e.h,e.base.v,parameters,paths)
    cal=e.calibrate(np.asarray([28,32,40]),64)
    inner,_=e.base.inner(cal['counts'],64,np.arange(8))
    rows,samples=e.evaluate(cal,paths,trace,inner)
    assert samples['corrected'].max()<=cal['total_bound']
    assert samples['corrected'].min()>=cal['calibration_floor']
    assert rows['tightened']['corrected_sample_variance']==bernstein(samples['corrected'],cal['range_width'])['variance']
    assert abs(rows['tightened']['corrected_sample_variance']-rows['tightened']['raw_sample_variance'])>1e-5
    checked=0
    for counts in product((8,12,16,20,24),repeat=3):
        if np.linalg.norm(np.asarray(counts)/32-cal['counts']/64)>cal['radius']:continue
        true_inner,_=e.base.inner(counts,32,np.arange(8))
        reference={**cal,'counts':np.asarray(counts),'n':32,'raw_bound':np.inf}
        true_lower,_=e.raw_interval(reference,paths,trace,true_inner)
        assert np.all(true_lower<=samples['corrected'])
        checked+=len(paths)
    assert checked>=100
