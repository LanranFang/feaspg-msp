"""Portable driver for the paper's approximate linearized-SDDP comparison.

The solver and simulator are the original implementations. This driver replaces
the former manuscript-adjacent launch script, using the settings stated in the
paper: 2 joint outcomes, 50 iterations, validation at 0, 10, ..., 50.
"""
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
import pandas as pd
import torch
from . import ROOT
from .inputs import load_input
from efficiency_switching_hydro.config import SDDPConfig
from efficiency_switching_hydro.sddp import fit_checkpoint
from efficiency_switching_hydro.evaluation import evaluate_with_costs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets',type=int,nargs='+',default=[100,250,500,1000,2500])
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--output',default='outputs/controlled_sddp')
    args=parser.parse_args();torch.set_num_threads(1)
    dest=ROOT/args.output;dest.mkdir(parents=True,exist_ok=True)
    config=SDDPConfig(iterations=50,validation_every=10,noise_scenarios=2)
    rows=[]
    for n in args.budgets:
        bundle=load_input(f'controlled_n{n}')
        for seed in args.seeds:
            folder=dest/f'n{n}'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():
                rows.append(json.loads((folder/'result.json').read_text()));continue
            start=time.perf_counter()
            fit=fit_checkpoint(bundle['fit'],bundle['validation'],bundle['system'],config,seed,folder/'selected.pt')
            metrics,costs=evaluate_with_costs(fit.policy,bundle['test'],bundle['system'])
            row={'n_fit':n,'seed':seed,'selected_iteration':fit.selected_iteration,
                 'best_validation_cost':fit.best_validation_cost,
                 'runtime_seconds':time.perf_counter()-start,'sddp_config':asdict(config),
                 'test':asdict(metrics)}
            (folder/'result.json').write_text(json.dumps(row,indent=2)+'\n')
            pd.DataFrame(fit.history).to_csv(folder/'history.csv',index=False)
            np.savez_compressed(folder/'path_costs.npz',test_costs=costs)
            rows.append(row);print(f'Completed n={n}, seed={seed}',flush=True)
    pd.json_normalize(rows).to_csv(dest/'per_seed.csv',index=False)


if __name__=='__main__':
    main()
