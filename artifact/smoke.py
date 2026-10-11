"""Small CPU fitting checks for every policy benchmark, not paper experiments."""
import json
import numpy as np
import torch
from . import ROOT
from .inputs import load_input
from iclr_algorithm1_experiments import regularized_a1_v2 as core
from iclr_algorithm1_experiments.projection_distance import regularized_objective
from iclr_algorithm1_experiments.projection_distance_run import configure


def main():
    configure();rows=[]
    for name,method in [('controlled_n100','projected_graph_neural_global'),
                        ('ons_subsystems_n250','algorithm1_mlp'),
                        ('camels_br_12_n250','algorithm1_deepsets')]:
        bundle=load_input(name);actor=core.build(bundle,method,0)
        optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
        for step in range(2):
            batch=bundle['fit'].take(np.arange(2*step,2*step+2))
            trace=core.simulate(actor,batch,bundle)
            loss=regularized_objective(trace,.01/np.sqrt(step+1))
            optimizer.zero_grad();loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(actor.parameters(),10);optimizer.step()
        metrics,_=core.evaluate(actor,bundle['validation'].take(np.arange(4)),bundle)
        rows.append(dict(bundle=name,method=method,updates=2,validation_cost=metrics['mean_cost_float64']))
        print(f'CPU fitting smoke passed: {name}',flush=True)
    dest=ROOT/'outputs/smoke';dest.mkdir(parents=True,exist_ok=True)
    (dest/'report.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
