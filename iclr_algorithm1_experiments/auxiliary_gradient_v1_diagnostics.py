"""Finite differences that detach x only inside the auxiliary residual."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .auxiliary_gradient_v1 import OUT, loss_for
from .config import ICLRProtocol
from .projection_distance import projection_distance, regularized_objective
from .projection_distance_run import configure, initial_policy, save_json, sha256
from .projection_plateau import DEFAULT_OUTPUT, fitting_data
from efficiency_switching_hydro.simulator import simulate


def two_stage_check():
    theta=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    x1=torch.clamp(theta,0.,2.)
    proposal=1.+.5*x1
    action=torch.minimum(torch.clamp_min(proposal,0.),x1)
    full=.5*(proposal-action)**2
    detached=.5*(proposal-action.detach())**2
    gf=float(torch.autograd.grad(full,theta,retain_graph=True)[0])
    gd=float(torch.autograd.grad(detached,theta)[0])
    d=lambda t:.5*(1-.5*t)**2
    fd=(d(1.+1e-6)-d(1.-1e-6))/(2e-6)
    assert abs(gf-fd)<1e-9 and gf==-0.25 and gd==0.25
    return {"theta":1.,"full_derivative":gf,"detached_derivative":gd,"finite_difference":fd,
            "distance":d(1.),"distance_after_full_step":d(1.-.1*gf),"distance_after_detached_step":d(1.-.1*gd)}


def main():
    configure()
    p=ICLRProtocol();h,fit=fitting_data("drought_02")
    checkpoint=DEFAULT_OUTPUT/"audit/seed0_fixed_one_update30.pt"
    original=DEFAULT_OUTPUT/"audit/finite_difference.csv"
    actor=initial_policy(p,h,0,"projected_mlp_matched")
    actor.load_state_dict(torch.load(checkpoint,weights_only=True))
    actor.double().eval()
    data=fit.take(np.arange(32),"audit_fit32")
    params=tuple(actor.parameters());base=parameters_to_vector(params).detach().clone()
    tr=simulate(actor,data,h,dtype=torch.float64)
    objectives={"full":regularized_objective(tr,1.),"detach_auxiliary_action":loss_for(tr,1.,True)}
    assert torch.equal(objectives["full"].detach(),objectives["detach_auxiliary_action"].detach())
    def grad(y):
        gs=torch.autograd.grad(y,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(params,gs)])
    gradients={name:grad(y) for name,y in objectives.items()}
    residual=(tr.proposals-tr.actions).detach()
    correction=grad((tr.actions*residual).sum((1,2)).mean())
    torch.testing.assert_close(gradients["detach_auxiliary_action"]-gradients["full"],correction,rtol=1e-8,atol=1e-7)
    rows=[]
    try:
        for index,seed in enumerate((20000,21000,22000)):
            rng=torch.Generator().manual_seed(seed)
            direction=torch.randn(base.shape,generator=rng,dtype=base.dtype)
            direction/=direction.norm().clamp(min=1e-12)
            for step in p.study_protocol().finite_difference_steps:
                with torch.no_grad():
                    vector_to_parameters(base+step*direction,params)
                    plus=float(regularized_objective(simulate(actor,data,h,dtype=torch.float64),1.))
                    vector_to_parameters(base-step*direction,params)
                    minus=float(regularized_objective(simulate(actor,data,h,dtype=torch.float64),1.))
                fd=(plus-minus)/(2*step)
                for mode,g in gradients.items():
                    ad=float(g@direction)
                    rows.append({"direction_index":index,"direction_seed":seed,"gradient_mode":mode,"lambda_value":1.,
                                 "step":step,"autograd_directional_derivative":ad,"finite_difference_directional_derivative":fd,
                                 "relative_error":abs(ad-fd)/max(1.,abs(ad),abs(fd)),"gradient_norm":float(g.norm()),
                                 "max_forward_difference":0.})
    finally:
        with torch.no_grad(): vector_to_parameters(base,params)
    f=pd.DataFrame(rows)
    old=pd.read_csv(original)
    joined=f[f.gradient_mode=="full"].merge(old[old.gradient_mode=="full"],on=["direction_seed","step"],suffixes=("_new","_old"))
    assert len(joined)==21
    for column in ("autograd_directional_derivative","finite_difference_directional_derivative","relative_error"):
        np.testing.assert_allclose(joined[column+"_new"],joined[column+"_old"],rtol=1e-10,atol=1e-10)
    dest=OUT/"diagnostics";dest.mkdir(parents=True,exist_ok=True)
    f.to_csv(dest/"finite_difference.csv",index=False)
    summary=f.groupby(["gradient_mode","step"]).relative_error.agg(["mean","std","max"]).reset_index()
    summary.to_csv(dest/"finite_difference_summary.csv",index=False)
    save_json(dest/"checks.json",{
        "source_checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
        "original_finite_differences":str(original),"original_sha256":sha256(original),
        "reused_complete_derivatives_reproduced":True,"forward_objectives_identical":True,
        "omitted_term_identity_verified":True,"n_fit_paths":32,"direction_seeds":[20000,21000,22000],
        "steps":list(p.study_protocol().finite_difference_steps),"checkpoint":"Seed 0, scarcity q=.02, fixed lambda=1, update 30; same checkpoint as the existing auxiliary-loss diagnostic.",
        "gradient_relative_difference":float((gradients["detach_auxiliary_action"]-gradients["full"]).norm()/gradients["full"].norm()),
        "gradient_cosine":float(torch.nn.functional.cosine_similarity(gradients["full"],gradients["detach_auxiliary_action"],dim=0)),
        "two_stage_sign_check":two_stage_check()})
    print(summary[summary.step==1e-5].to_string(index=False),flush=True)


if __name__=="__main__": main()
