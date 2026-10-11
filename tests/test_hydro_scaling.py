"""Independent audits for the continuous coupled-hydro scaling pilot."""
from fractions import Fraction

import numpy as np

from synthetic_hydro_certificate.scaling import (
    CoupledHydro, Grid, ScalingEngine, SCALE, paths_from_codes,
    continuous_inner_audit, continuous_tree_lp, policy_intervals,
    exact_path_masses, oracle_probabilities,
)


def setup(R=2,T=3):
    h=CoupledHydro(R,T);g=Grid(h)
    fit=(np.random.default_rng(817).random((256,T))<.5).astype(np.uint8)
    v,_=g.fit_continuation(fit)
    return h,g,v


def test_continuous_coupled_inner_and_suffix_cache():
    for R in (1,2,3):
        h,g,v=setup(R)
        for continuation in (v,np.zeros_like(v)):
            e=ScalingEngine(g,continuation,threads=2)
            counts=np.array([80,170,120]);codes=np.arange(8)
            exact,_=e.inner(counts,256,codes)
            subset,_=e.inner(counts,256,np.array([7,0,3,3]))
            np.testing.assert_array_equal(subset,exact[[7,0,3,3]])
            for i,path in enumerate(paths_from_codes(codes,3)):
                audit=continuous_inner_audit(e,counts,256,path)
                assert abs(audit['value']-exact[i]/(256*SCALE))<1e-7
                assert abs(audit['bound']-exact[i]/(256*SCALE))<1e-7


def test_reference_bracket_contains_continuous_tree_optimum():
    h,g,v=setup()
    p=oracle_probabilities(h)
    values=g.bellman(p,20,exact=True)
    upper=Fraction(int(values[0][g.initial]),SCALE*20**h.stages)
    inner,_=ScalingEngine(g,v,threads=2).inner(p,20,np.arange(2**h.stages))
    lower=Fraction(sum(int(a)*int(b) for a,b in zip(inner,exact_path_masses(h))),
                   20*SCALE*20**h.stages)
    lp=continuous_tree_lp(h)
    assert float(lower)-1e-7<=lp['value']<=float(upper)+1e-7


def test_interval_policy_matches_exact_fractional_replay():
    h,g,v=setup()
    params=np.random.default_rng(10).uniform(-.15,.4,(3,2,3))
    params[:,:,0]+=.8
    paths=paths_from_codes(np.arange(8),3)
    c,d=policy_intervals(h,v,params,paths)
    for i,path in enumerate(paths):
        s=[Fraction(int(x)) for x in h.initial]
        for t,wet in enumerate(path):
            a=[min(max(Fraction(float(params[t,r,0]))+sum(Fraction(float(params[t,r,k+1]))*s[k] for k in range(h.reservoirs)),0),s[r],int(h.release[r])) for r in range(h.reservoirs)]
            cost=[];response=[];successors=[]
            for w in (0,1):
                nxt=[min(int(h.capacity[r]),s[r]-a[r]+int(h.inflow[r])*w) for r in range(h.reservoirs)]
                val=int(h.prices[t])*max(int(h.demand(w))-sum(a),0)+sum(Fraction(r,8)*a[r] for r in range(h.reservoirs))
                if t==h.stages-1:val+=4*sum(max(3-z,0) for z in nxt)
                cost.append(val);future=Fraction(0)
                for r,z in enumerate(nxt):
                    k=min(z.numerator//z.denominator,int(h.capacity[r])-1)
                    future+=Fraction(int(v[t+1,r,k]),SCALE)+(z-k)*Fraction(int(v[t+1,r,k+1]-v[t+1,r,k]),SCALE)
                response.append(val+future);successors.append(nxt)
            assert Fraction(float(c[0,i,t]))<=cost[wet]<=Fraction(float(c[1,i,t]))
            difference=response[1]-response[0]
            assert Fraction(float(d[0,i,t]))<=difference<=Fraction(float(d[1,i,t]))
            s=successors[wet]
    e=ScalingEngine(g,v,threads=2);cal=e.calibrate(np.array([80,170,120]),256)
    exact,_=e.inner(cal['counts'],256,np.arange(8))
    report=e.evaluate(cal,paths,(c,d),exact)
    assert report['max_arithmetic_interval']<1e-6
    assert report['certificate']>0
