"""Proved structured refinements of Algorithm 2 for independent binary stages.

Keep frozen v1/v2 code intact. The joint calibration theorem, paired contrast
comparison, and conditional empirical Bernstein argument are documented in
TIGHTENING_PROTOCOL.md and independently reviewed in the accompanying notes.
"""
from fractions import Fraction
from math import log, sqrt

import numpy as np

from .scaling import SCALE, UP, DOWN, iadd, imul, isum


def exact_float_moments(values):
    """Exact mean and unbiased variance of finite nonnegative binary64 values.

    All mantissas are aligned to one dyadic denominator; Python integers avoid
    overflow in the sum of squares. Thus variance rounding cannot invalidate
    the empirical Bernstein radius.
    """
    values=np.ascontiguousarray(values,dtype=np.float64)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected at least two finite nonnegative observations')
    bits=values.view(np.uint64)
    exponent=((bits>>52)&2047).astype(np.int64)
    mantissa=(bits&((1<<52)-1))|np.where(exponent>0,np.uint64(1<<52),np.uint64(0))
    power=np.where(exponent>0,exponent-1075,-1074)
    nonzero=mantissa>0
    if not nonzero.any():return Fraction(0),Fraction(0)
    base=int(power[nonzero].min())
    shifts=np.where(nonzero,power-base,0)
    integers=np.left_shift(mantissa.astype(object),shifts)
    total=int(integers.sum());squares=int((integers*integers).sum());n=len(values)
    unit=Fraction(2**base) if base>=0 else Fraction(1,2**(-base))
    mean=Fraction(total,n)*unit
    variance=Fraction(n*squares-total*total,n*(n-1))*unit*unit
    if variance<0:raise RuntimeError('Exact variance cannot be negative')
    return mean,variance


def bernstein(values,range_width,alpha=.025):
    """Maurer--Pontil (2009), Theorem 4, conditional on frozen calibration."""
    mean,var=exact_float_moments(values)
    mean_upper=float(UP(float(mean)));var_upper=float(UP(float(var)))
    ell=UP(log(2/alpha))
    variance_part=UP(sqrt(float(UP(UP(2*var_upper)*ell/len(values)))))
    range_part=UP(UP(UP(7*range_width)*ell)/(3*(len(values)-1)))
    return dict(mean=mean_upper,variance=var_upper,
                evaluation=float(UP(variance_part+range_part)),
                variance_part=float(variance_part),range_part=float(range_part))


def norm_bounds(nonnegative_lower,nonnegative_upper):
    low=np.maximum(nonnegative_lower,0);high=np.maximum(nonnegative_upper,0)
    lo=np.zeros(low.shape[:-1]);hi=np.zeros(high.shape[:-1])
    for t in range(low.shape[-1]):
        lo=np.maximum(0,DOWN(lo+np.maximum(0,DOWN(low[...,t]*low[...,t]))))
        hi=UP(hi+UP(high[...,t]*high[...,t]))
    return np.maximum(0,DOWN(np.sqrt(lo))),UP(np.sqrt(hi))


class TightenedEngine:
    def __init__(self,base):
        self.base=base;self.grid=base.grid;self.h=base.h
        self.minimum=base.contrast.min(axis=1)/SCALE
        self.maximum=base.contrast.max(axis=1)/SCALE
        self.span=self.maximum-self.minimum
        self.current=np.zeros((self.h.stages,len(self.grid.a)),dtype=np.int64)
        self.future0=np.zeros_like(self.current)
        for t in range(self.h.stages):
            for r in range(self.h.reservoirs):
                self.current[t]+=base.v[t,r,self.grid.s[:,r]]
                self.future0[t]+=base.v[t+1,r,self.grid.next[0,:,r]]

    def structural_bounds(self,counts,n):
        """Exact residual extrema over controls and shocks, without path listing.

        F(x,w)=v_1(s_1)+sum_t(Qhat_t(s_t,x_t)-v_t(s_t)); the first term
        cancels from a paired gap. Continuous PWL extrema are integral by the
        network-cell proof already used for the inner solver, for max and min.
        """
        g=self.grid;e=self.base
        magnitude=lambda a:max(abs(int(np.min(a))),abs(int(np.max(a))))
        residual_cap=(int(n)*(magnitude(g.cost[:,0])+magnitude(self.future0)+magnitude(self.current))
                      +max(abs(int(c)) for c in counts)*magnitude(e.contrast))
        if self.h.stages*residual_cap>=2**62:
            raise OverflowError('Residual DP input bound exceeds exact accumulator cap')
        old0,old1=e.objectives(counts,n)
        stage_old=int(np.maximum(old0.max(1)-old0.min(1),old1.max(1)-old1.min(1)).sum())
        residual=n*(g.cost[:,0]+self.future0-self.current)+np.asarray(counts,dtype=np.int64)[:,None]*e.contrast
        if self.h.stages*int(np.abs(residual).max())>=2**62:
            raise OverflowError('Residual DP exact accumulator cap exceeded')
        stage_residual=int((residual.max(1)-residual.min(1)).sum())
        low=np.zeros(len(g.states),dtype=np.int64);high=low.copy()
        for t in reversed(range(self.h.stages)):
            low_candidate=residual[t]+np.minimum(low[g.next_index[0]],low[g.next_index[1]])
            high_candidate=residual[t]+np.maximum(high[g.next_index[0]],high[g.next_index[1]])
            low=np.minimum.reduceat(low_candidate,g.offsets[:-1])
            high=np.maximum.reduceat(high_candidate,g.offsets[:-1])
        adversarial=int(high[g.initial])-int(low[g.initial])
        denominator=n*SCALE
        return dict(old=stage_old/denominator,residual=stage_residual/denominator,
            adversarial=adversarial/denominator,
            selected=float(UP(min(stage_old,stage_residual,adversarial)/denominator)),
            residual_min_numerator=int(low[g.initial]),residual_max_numerator=int(high[g.initial]),
            denominator=denominator)

    def calibrate(self,counts,n,delta=.025,rounding_cap=1e-6):
        if n<1 or n&(n-1):raise ValueError('This implementation uses dyadic calibration budgets')
        counts=np.asarray(counts)
        if counts.shape!=(self.h.stages,) or np.any(counts<0) or np.any(counts>n) or np.any(counts!=np.floor(counts)):
            raise ValueError('Calibration counts must be integer frequencies for every stage')
        if not 0<delta<1:raise ValueError('Expected calibration error probability in (0,1)')
        original=self.base.calibrate(counts,n,delta,rounding_cap)
        bounds=self.structural_bounds(counts,n)
        x=UP(log(1/delta));T=self.h.stages
        expression=UP(UP(T+UP(2*UP(sqrt(float(UP(T*x))))))+UP(2*x))
        radius=float(UP(sqrt(float(UP(expression/(4*n))))+1e-13))
        length_low,length_high=norm_bounds(self.span,self.span)
        cap=float(UP(radius*length_high))
        floor=max(0,float(DOWN(DOWN(radius*length_low)/2)))
        raw_bound=float(UP(bounds['selected']+rounding_cap))
        total_bound=float(UP(raw_bound+cap))
        range_width=float(UP(total_bound-floor))
        old_width=float(UP(bounds['old']+rounding_cap))
        centered_shift=float(UP(float(sum(Fraction(float(s)) for s in self.span)*Fraction(float(original['rho'])))))
        return dict(original=original,counts=np.asarray(counts,dtype=np.int64),n=n,
            delta=delta,radius=radius,calibration_cap=cap,calibration_floor=floor,
            raw_bound=raw_bound,total_bound=total_bound,range_width=range_width,
            old_raw_bound=old_width,centered_shift=centered_shift,
            rounding_cap=rounding_cap,bounds=bounds)

    def raw_interval(self,cal,paths,trace,inner_numerators):
        c,d=trace
        factor=cal['counts']/cal['n']-paths
        stages=iadd((c[0],c[1]),imul((d[0],d[1]),(factor,factor)))
        policy=isum(stages)
        inner_low=DOWN(inner_numerators/(cal['n']*SCALE))
        inner_high=UP(inner_numerators/(cal['n']*SCALE))
        lo=np.maximum(0,DOWN(policy[0]-inner_high));hi=np.maximum(0,UP(policy[1]-inner_low))
        if np.any(lo>hi) or float((hi-lo).max())>cal['rounding_cap']:
            raise RuntimeError('Paired-gap numerical enclosure failed')
        if float(hi.max())>cal['raw_bound']:raise RuntimeError('Frozen raw envelope failed')
        return lo,hi

    def correction_interval(self,cal,contrast):
        low=np.maximum(contrast[0],self.minimum)
        high=np.minimum(contrast[1],self.maximum)
        if np.any(low>high):raise RuntimeError('Policy contrast outside certified query bounds')
        amin=np.maximum.reduce([DOWN(low-self.minimum),DOWN(self.maximum-high),
                                np.broadcast_to(self.span/2,low.shape)])
        amax=UP(np.maximum(UP(high-self.minimum),UP(self.maximum-low)))
        norm_low,norm_high=norm_bounds(amin,amax)
        k_low=np.maximum(0,DOWN(cal['radius']*norm_low))
        k_high=np.minimum(cal['calibration_cap'],UP(cal['radius']*norm_high))
        if np.any(k_low>k_high) or float((k_high-k_low).max())>cal['rounding_cap']:
            raise RuntimeError('Calibration correction numerical enclosure failed')
        return k_low,k_high

    def evaluate(self,cal,paths,trace,inner_numerators,alpha=.025):
        raw_low,raw=self.raw_interval(cal,paths,trace,inner_numerators)
        k_low,kappa=self.correction_interval(cal,trace[1])
        corrected=UP(raw+kappa)
        # Numeric allowances are included in raw_bound; every policy calculation
        # is a deterministic upper enclosure. Stop rather than discard failures.
        total_round=(raw-raw_low)+(kappa-k_low)
        if float(total_round.max())>cal['rounding_cap']:
            raise RuntimeError('Combined numerical enclosure cap exceeded')
        if corrected.min()<cal['calibration_floor'] or corrected.max()>cal['total_bound']:
            raise RuntimeError('Frozen corrected envelope failed')
        raw_mom=bernstein(raw,cal['raw_bound'],alpha)
        full_mom=bernstein(corrected,cal['range_width'],alpha)
        raw_mean=float(UP(float(exact_float_moments(raw)[0])))
        k_mean=float(UP(float(exact_float_moments(kappa)[0])))
        old=self.base.evaluate(cal['original'],paths,trace,inner_numerators,alpha)
        ell=UP(log(2/alpha));n=len(paths)
        joint_hoeffding=float(UP(UP(cal['old_raw_bound']+cal['calibration_cap'])*sqrt(float(UP(ell/(2*n))))))
        variants={
            'original':dict(raw_gap=old['raw_gap'],calibration_correction=old['calibration_correction'],
                evaluation_correction=old['evaluation_correction'],certificate=old['certificate']),
            'calibration_only':dict(raw_gap=raw_mean,calibration_correction=k_mean,
                evaluation_correction=joint_hoeffding,certificate=float(UP(full_mom['mean']+joint_hoeffding))),
            'evaluation_only':dict(raw_gap=raw_mean,calibration_correction=cal['original']['shift'],
                evaluation_correction=raw_mom['evaluation'],
                certificate=float(UP(UP(raw_mom['mean']+cal['original']['shift'])+raw_mom['evaluation']))),
            'tightened':dict(raw_gap=raw_mean,calibration_correction=k_mean,
                evaluation_correction=full_mom['evaluation'],certificate=float(UP(full_mom['mean']+full_mom['evaluation']))),
            'centered_hoeffding_eb':dict(raw_gap=raw_mean,calibration_correction=cal['centered_shift'],
                evaluation_correction=raw_mom['evaluation'],
                certificate=float(UP(UP(raw_mom['mean']+cal['centered_shift'])+raw_mom['evaluation']))),
        }
        used_ranges={'original':cal['original']['envelope'],
            'calibration_only':float(UP(cal['old_raw_bound']+cal['calibration_cap'])),
            'evaluation_only':cal['raw_bound'],'tightened':cal['range_width'],
            'centered_hoeffding_eb':cal['raw_bound']}
        used_calibration={'original':'coordinate Hoeffding, absolute contrasts',
            'calibration_only':'independent-stage joint L2, paired contrasts',
            'evaluation_only':'coordinate Hoeffding, absolute contrasts',
            'tightened':'independent-stage joint L2, paired contrasts',
            'centered_hoeffding_eb':'coordinate Hoeffding, centered contrasts'}
        for name,row in variants.items():
            row.update(raw_envelope=cal['raw_bound'],joint_corrected_range=cal['range_width'],
                evaluation_range_used=used_ranges[name],calibration_rule_used=used_calibration[name],
                variance_used=(full_mom['variance'] if name=='tightened' else
                               raw_mom['variance'] if name in ('evaluation_only','centered_hoeffding_eb') else None),
                joint_radius=cal['radius'],calibration_cap=cal['calibration_cap'],
                corrected_sample_variance=full_mom['variance'],raw_sample_variance=raw_mom['variance'],
                max_arithmetic_interval=float(total_round.max()))
        return variants,dict(raw=raw,kappa=kappa,corrected=corrected)
