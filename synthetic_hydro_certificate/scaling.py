"""Coupled continuous-control hydro: an independent scaling pilot, version 2.

No v1 artifact or frozen source is modified. Finite integer vertices are used
only when solving a continuous PWL inner problem with an integral network-cell
matrix. The stochastic integer-grid Bellman solution is explicitly an UPPER
reference for the continuous stochastic optimum, never asserted to be exact.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
SCALE = 65536
DOWN = lambda a: np.nextafter(a, -np.inf)
UP = lambda a: np.nextafter(a, np.inf)


@dataclass(frozen=True)
class CoupledHydro:
    reservoirs: int
    stages: int

    @property
    def capacity(self): return np.arange(4, 4 + self.reservoirs, dtype=np.int64)
    @property
    def release(self): return np.arange(2, 2 + self.reservoirs, dtype=np.int64)
    @property
    def inflow(self): return np.arange(1, 1 + self.reservoirs, dtype=np.int64)
    @property
    def initial(self): return np.full(self.reservoirs, 3, dtype=np.int64)
    @property
    def operating(self): return np.arange(self.reservoirs) / 8
    @property
    def prices(self): return np.resize([1, 4, 2, 6, 3, 8], self.stages)
    def demand(self, wet): return self.reservoirs * (3 - np.asarray(wet))


class Grid:
    def __init__(self, h):
        self.h = h
        self.states = np.asarray(list(product(*[range(k+1) for k in h.capacity])), dtype=np.int64)
        self.shape = tuple(h.capacity + 1)
        states, actions, offsets = [], [], [0]
        for s in self.states:
            a = list(product(*[range(min(k, int(z))+1) for k,z in zip(h.release,s)]))
            actions.extend(a); states.extend([s] * len(a)); offsets.append(len(actions))
        self.s = np.asarray(states); self.a = np.asarray(actions)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.next = np.asarray([np.minimum(h.capacity, self.s-self.a+h.inflow*w) for w in (0,1)])
        self.next_index = np.asarray([np.ravel_multi_index(s.T,self.shape) for s in self.next], dtype=np.int64)
        self.initial = int(np.ravel_multi_index(h.initial,self.shape))
        self.cost = np.empty((h.stages,2,len(self.a)), dtype=np.int64)
        for t,p in enumerate(h.prices):
            for w in (0,1):
                c = p * np.maximum(h.demand(w)-self.a.sum(axis=1),0) + self.a@h.operating
                if t == h.stages-1: c += 4 * np.maximum(3-self.next[w],0).sum(axis=1)
                self.cost[t,w] = np.rint(c*SCALE).astype(np.int64)

    def bellman(self, counts, n, exact=False):
        """Integer-action policy optimum: feasible reference for continuous MSP."""
        values = [None] * (self.h.stages+1)
        values[-1] = np.zeros(len(self.states), dtype=object if exact else float)
        for t in reversed(range(self.h.stages)):
            if exact:
                # Common denominator SCALE * n**(T-t) at stage t.
                factor = n**(self.h.stages-t-1)
                scores = sum(int(n-counts[t] if w==0 else counts[t]) *
                    (self.cost[t,w].astype(object)*factor + values[t+1][self.next_index[w]]) for w in (0,1))
            else:
                p = counts[t]/n
                scores = sum(prob*(self.cost[t,w]/SCALE+values[t+1][self.next_index[w]])
                             for w,prob in ((0,1-p),(1,p)))
            values[t] = np.minimum.reduceat(scores,self.offsets[:-1])
        return values

    def fit_continuation(self, fit):
        """Fit an additive convex PWL surrogate to coupled empirical DP targets.

        v_t(s)=b_t+sum_{r,k} a_{t,r,k} max(k-s_r,0), a>=0.
        Store dyadic knot values; no true probabilities enter this operation.
        """
        target = self.bellman(fit.sum(axis=0),len(fit))
        basis = [np.ones(len(self.states))]
        for r,kmax in enumerate(self.h.capacity):
            basis.extend(np.maximum(k-self.states[:,r],0) for k in range(1,kmax+1))
        design = np.asarray(basis).T
        v = np.zeros((self.h.stages+1,self.h.reservoirs,max(self.h.capacity)+1),dtype=np.int64)
        errors=[]
        for t in range(self.h.stages):
            coef,_=nnls(design,target[t],maxiter=10000)
            errors.append(float(np.sqrt(np.mean((design@coef-target[t])**2))))
            j=1
            for r,kmax in enumerate(self.h.capacity):
                row=np.full(kmax+1,coef[0]/self.h.reservoirs)
                for k in range(1,kmax+1):
                    row += coef[j]*np.maximum(k-np.arange(kmax+1),0);j+=1
                v[t,r,:kmax+1]=np.rint(row*SCALE).astype(np.int64)
                v[t,r,kmax+1:]=v[t,r,kmax]
        return v,errors


def torch_cost(params, paths, h):
    s=torch.as_tensor(h.initial,dtype=torch.float64).repeat(len(paths),1)
    limits=torch.as_tensor(h.release); capacities=torch.as_tensor(h.capacity)
    inflow=torch.as_tensor(h.inflow); operating=torch.as_tensor(h.operating)
    total=torch.zeros(len(paths),dtype=torch.float64)
    for t,p in enumerate(h.prices):
        proposal=params[t,:,0]+s@params[t,:,1:].T
        a=torch.minimum(torch.relu(proposal),torch.minimum(s,limits))
        total=total+p*torch.relu(h.reservoirs*(3-paths[:,t])-a.sum(1))+(a*operating).sum(1)
        s=torch.minimum(capacities,s-a+paths[:,t,None]*inflow)
    return total+4*torch.relu(3-s).sum(1)


def fit_checkpoints(h, fit, validation, seed, steps=400):
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    initial=np.zeros((h.stages,h.reservoirs,h.reservoirs+1))
    initial[:,:,0]=.3
    for r in range(h.reservoirs):initial[:,r,r+1]=.5
    initial+=rng.normal(0,.01,initial.shape)
    params=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64))
    optimizer=torch.optim.Adam([params],lr=.02)
    fit_t=torch.as_tensor(fit,dtype=torch.float64);val_t=torch.as_tensor(validation,dtype=torch.float64)
    checkpoints={};history=[]
    for step in range(steps+1):
        if step in (steps//10,steps//2,steps):
            with torch.no_grad():score=float(torch_cost(params,val_t,h).mean())
            checkpoints[str(step)]=params.detach().numpy().copy()
            history.append(dict(update=step,validation_cost=score))
        if step==steps:break
        optimizer.zero_grad()
        torch_cost(params,fit_t[rng.integers(len(fit),size=64)],h).mean().backward()
        optimizer.step()
    return checkpoints,history


def iadd(a,b):return DOWN(a[0]+b[0]),UP(a[1]+b[1])
def isub(a,b):return DOWN(a[0]-b[1]),UP(a[1]-b[0])
def imul(a,b):
    x=[a[i]*b[j] for i in (0,1) for j in (0,1)]
    return DOWN(np.minimum.reduce(x)),UP(np.maximum.reduce(x))
def isum(a,axis=-1):
    # Explicit outward accumulation: do not rely on unspecified BLAS reduction.
    lo=np.zeros(np.delete(a[0].shape,axis));hi=lo.copy()
    for k in range(a[0].shape[axis]):
        lo=DOWN(lo+np.take(a[0],k,axis=axis));hi=UP(hi+np.take(a[1],k,axis=axis))
    return lo,hi


def value_interval(v,t,state,h):
    total=(np.zeros(len(state[0])),np.zeros(len(state[0])))
    for r,kmax in enumerate(h.capacity):
        lo=np.clip(state[0][:,r],0,kmax);hi=np.clip(state[1][:,r],0,kmax)
        # Rigorous PWL range via endpoints and every contained integer knot.
        ends=[]
        for z in (lo,hi):
            k=np.minimum(np.floor(z).astype(int),kmax-1)
            base=v[t,r,k]/SCALE;slope=(v[t,r,k+1]-v[t,r,k])/SCALE
            ends.append(iadd((base,base),imul((slope,slope),(DOWN(z-k),UP(z-k)))))
        low=np.minimum(ends[0][0],ends[1][0]);high=np.maximum(ends[0][1],ends[1][1])
        for k in range(kmax+1):
            inside=(lo<=k)&(hi>=k);val=v[t,r,k]/SCALE
            low=np.where(inside,np.minimum(low,val),low);high=np.where(inside,np.maximum(high,val),high)
        total=iadd(total,(low,high))
    return total


def policy_intervals(h,v,parameters,paths):
    """Outward intervals enclose the exact-real saved-float affine policy."""
    n=len(paths);s0=np.broadcast_to(h.initial,(n,h.reservoirs)).astype(float).copy();s=(s0,s0)
    costs=[];contrasts=[]
    for t,p in enumerate(h.prices):
        lo=np.broadcast_to(parameters[t,:,0],s0.shape).copy();hi=lo.copy()
        for r in range(h.reservoirs):
            coeff=parameters[t,:,r+1]
            term=imul((s[0][:,r,None],s[1][:,r,None]),(coeff,coeff))
            lo,hi=iadd((lo,hi),term)
        a=(np.minimum(np.maximum(lo,0),np.minimum(s[0],h.release)),
           np.minimum(np.maximum(hi,0),np.minimum(s[1],h.release)))
        # Tighten using known physical feasibility before evaluating responses.
        a=(np.maximum(a[0],0),np.maximum(a[1],0))
        y=isub(s,a); y=(np.maximum(y[0],0),np.minimum(y[1],h.capacity))
        release=isum(a);op=isum(imul(a,(h.operating,h.operating)))
        response=[];stage_cost=[];next_states=[]
        for wet in (0,1):
            nxt=iadd(y,(h.inflow*wet,h.inflow*wet))
            nxt=(np.clip(nxt[0],0,h.capacity),np.clip(nxt[1],0,h.capacity));next_states.append(nxt)
            thermal=isub((h.demand(wet),h.demand(wet)),release)
            thermal=(np.maximum(thermal[0],0),np.maximum(thermal[1],0))
            c=iadd(imul(thermal,(p,p)),op)
            if t==h.stages-1:
                short=isub((3,3),nxt)
                short=(np.maximum(short[0],0),np.maximum(short[1],0))
                c=iadd(c,imul(isum(short),(4,4)))
            stage_cost.append(c);response.append(iadd(c,value_interval(v,t+1,nxt,h)))
        w=paths[:,t].astype(bool)
        costs.append(tuple(np.where(w,stage_cost[1][k],stage_cost[0][k]) for k in (0,1)))
        contrasts.append(isub(response[1],response[0]))
        s=tuple(np.where(w[:,None],next_states[1][k],next_states[0][k]) for k in (0,1))
    return np.asarray(costs).transpose(1,2,0),np.asarray(contrasts).transpose(1,2,0)


def compile_inner():
    source=ROOT/'scaling_inner.cpp';dest=ROOT/'outputs/scaling_v2/bin'
    dest.mkdir(parents=True,exist_ok=True)
    library=dest/(hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.so')
    if not library.exists():
        subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.inner_step.argtypes=[ctypes.c_int]*3+[ptr]*9+[ctypes.c_int]
    return lib


class ScalingEngine:
    def __init__(self,grid,v,threads=8):
        self.grid=grid;self.h=grid.h;self.v=v;self.threads=threads;self.lib=compile_inner()
        values=np.zeros((self.h.stages,2,len(grid.a)),dtype=np.int64)
        for t in range(self.h.stages):
            for w in (0,1):
                for r in range(self.h.reservoirs):values[t,w]+=v[t+1,r,grid.next[w,:,r]]
        self.contrast=grid.cost[:,1]+values[:,1]-grid.cost[:,0]-values[:,0]
        self.R=np.max(np.abs(self.contrast),axis=1)/SCALE

    def objectives(self,counts,n):
        a=np.asarray(counts,dtype=np.int64)[:,None]
        worst=self.h.stages*int(np.max(np.abs(self.grid.cost))+np.max(np.abs(self.contrast)))*int(n)
        if worst>=2**62:raise OverflowError('Declared exact-integer accumulator bound exceeded')
        return np.ascontiguousarray(n*self.grid.cost[:,0]+a*self.contrast),np.ascontiguousarray(n*self.grid.cost[:,1]+(a-n)*self.contrast)

    def inner(self,counts,n,codes):
        """Exact global continuous inner values, memoizing shared path suffixes."""
        codes=np.asarray(codes,dtype=np.int64);g=self.grid;f0,f1=self.objectives(counts,n)
        previous=np.zeros((1,len(g.states)),dtype=np.int64);previous_codes=np.array([0],dtype=np.int64)
        start=time.perf_counter()
        for t in reversed(range(self.h.stages)):
            remaining=self.h.stages-t
            unique=np.unique(codes&((1<<remaining)-1))
            wet=np.ascontiguousarray(unique>>(remaining-1));child=np.searchsorted(previous_codes,unique&((1<<(remaining-1))-1)).astype(np.int64)
            current=np.empty((len(unique),1 if t==0 else len(g.states)),dtype=np.int64)
            self.lib.inner_step(len(g.states),len(unique),g.initial if t==0 else -1,g.offsets,
                g.next_index[0],g.next_index[1],f0[t],f1[t],wet,child,previous,current,self.threads)
            previous=current;previous_codes=unique
        return previous[np.searchsorted(previous_codes,codes),0],time.perf_counter()-start

    def calibrate(self,counts,n,delta=.025,rounding_cap=1e-6):
        from math import sqrt,log
        radius=min(1,sqrt(log(2*self.h.stages/delta)/(2*n))+1e-14)
        beta=UP(self.R*radius+1e-13)
        f0,f1=self.objectives(counts,n)
        ranges=np.maximum(f0.max(axis=1)-f0.min(axis=1),f1.max(axis=1)-f1.min(axis=1))
        shift=float(UP(float(2*sum(Fraction(float(b)) for b in beta))))
        range_upper=UP(sum(int(r) for r in ranges)/(n*SCALE))
        envelope=float(UP(UP(range_upper+shift)+rounding_cap))
        return dict(counts=np.asarray(counts),n=n,beta=beta,rho=radius,shift=shift,
                    envelope=envelope,rounding_cap=rounding_cap)

    def evaluate(self,cal,paths,trace,inner_numerators,alpha=.025):
        from math import sqrt,log
        c,d=trace
        # Dyadic n in the pilot: p_hat and p_hat-w are represented exactly.
        p=cal['counts']/cal['n']-paths
        stage=iadd((c[0],c[1]),imul((d[0],d[1]),(p,p)))
        policy=isum(stage)
        lower=DOWN(inner_numerators/(cal['n']*SCALE));upper=UP(inner_numerators/(cal['n']*SCALE))
        rounding=policy[1]-policy[0]+upper-lower
        if rounding.max()>cal['rounding_cap']:raise RuntimeError('Policy arithmetic cap exceeded')
        raw=UP(policy[1]-lower)
        if raw.min() < -cal['rounding_cap']:raise RuntimeError('Policy below global inner optimum')
        corrected=UP(raw+cal['shift'])
        if corrected.max()>cal['envelope']:raise RuntimeError('Predeclared envelope failed')
        # Summation error upper bound, independent of observation values via B_G.
        mean=float(np.mean(raw)+np.finfo(float).eps*len(raw)*cal['envelope']*4)
        outer=float(UP(cal['envelope']*sqrt(log(2/alpha)/(2*len(paths)))))
        return dict(raw_gap=mean,calibration_correction=cal['shift'],evaluation_correction=outer,
                    certificate=float(UP(mean+cal['shift']+outer)),envelope=cal['envelope'],
                    max_arithmetic_interval=float(rounding.max()),exact_inner_gap=0.)


def paths_from_codes(codes,T):
    return ((np.asarray(codes)[:,None]>>np.arange(T-1,-1,-1))&1).astype(np.uint8)


def oracle_probabilities(h):
    """Only the data driver and reference assessment call this function."""
    return np.resize([7,12,9,13,8,11],h.stages)


def exact_path_masses(h):
    weights=np.array([1],dtype=object)
    for p in oracle_probabilities(h):weights=np.stack((weights*(20-int(p)),weights*int(p)),axis=1).ravel()
    assert sum(weights)==20**h.stages
    return weights


def continuous_tree_lp(h):
    """Independent continuous reference. Extra free spill cannot improve optimum:
    costs/terminal shortage are monotone in available water and storing is free.
    A policy can retain extra water and emulate any free-spill policy's releases.
    """
    import gurobipy as gp
    with gp.Env(params={'OutputFlag':0}) as env, gp.Model(env=env) as m:
        m.Params.Threads=2;m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9
        states=[m.addMVar(h.reservoirs,lb=h.initial,ub=h.initial)]
        masses=np.array([1.]);objective=gp.LinExpr()
        for t,price in enumerate(h.prices):
            nxt_states=[];nxt_masses=[];p=oracle_probabilities(h)[t]/20
            for s,mass in zip(states,masses):
                a=m.addMVar(h.reservoirs,lb=0,ub=h.release);m.addConstr(a<=s)
                for wet,prob in ((0,1-p),(1,p)):
                    nxt=m.addMVar(h.reservoirs,lb=0,ub=h.capacity)
                    m.addConstr(nxt<=s-a+h.inflow*wet)
                    thermal=m.addVar(lb=0);m.addConstr(thermal>=h.demand(wet)-a.sum())
                    term=price*thermal+(h.operating@a)
                    if t==h.stages-1:
                        deficit=m.addMVar(h.reservoirs,lb=0);m.addConstr(deficit>=3-nxt)
                        term+=4*deficit.sum()
                    objective+=mass*prob*term.item()
                    nxt_states.append(nxt);nxt_masses.append(mass*prob)
            states=nxt_states;masses=np.asarray(nxt_masses)
        m.setObjective(objective);m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:raise RuntimeError('Continuous reference LP failed')
        return dict(value=m.ObjVal,bound=m.ObjBound,variables=m.NumVars,constraints=m.NumConstrs)


def continuous_inner_audit(engine,counts,n,path):
    import gurobipy as gp
    h,v=engine.h,engine.v
    with gp.Env(params={'OutputFlag':0}) as env,gp.Model(env=env) as m:
        m.Params.Threads=1;m.Params.MIPGap=0;m.Params.MIPGapAbs=1e-8
        m.Params.FeasibilityTol=1e-9;m.Params.OptimalityTol=1e-9;m.Params.TimeLimit=30
        m.Params.IntFeasTol=1e-9
        previous=None
        for t,wet in enumerate(path):
            a=m.addMVar(h.reservoirs,lb=0,ub=h.release,obj=h.operating)
            post=m.addMVar(h.reservoirs,lb=0,ub=h.capacity)
            m.addConstr(post+a==(h.initial if t==0 else previous))
            q=m.addVar(lb=0,ub=int(h.release.sum()));m.addConstr(q==a.sum())
            p=counts[t]/n;knots=list(range(int(h.release.sum())+1))
            m.setPWLObj(q,knots,[h.prices[t]*((1-p)*max(h.demand(0)-x,0)+p*max(h.demand(1)-x,0)) for x in knots])
            nxt=m.addMVar(h.reservoirs,lb=0,ub=h.capacity)
            for r,K in enumerate(h.capacity):
                xs=list(range(K+1));vals=[]
                for y in xs:
                    expected=0
                    for donor,prob in ((0,1-p),(1,p)):
                        z=min(K,y+h.inflow[r]*donor)
                        response=v[t+1,r,z]/SCALE
                        if t==h.stages-1:response+=4*max(3-z,0)
                        expected+=prob*response
                    vals.append(expected-v[t+1,r,min(K,y+h.inflow[r]*int(wet))]/SCALE)
                m.setPWLObj(post[r].item(),xs,vals)
                m.addGenConstrPWL(post[r].item(),nxt[r].item(),xs,[min(K,y+h.inflow[r]*int(wet)) for y in xs])
            previous=nxt
        m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:raise RuntimeError(f'Continuous inner audit status {m.Status}')
        return dict(value=m.ObjVal,bound=m.ObjBound,mip=bool(m.IsMIP),nodes=m.NodeCount)
