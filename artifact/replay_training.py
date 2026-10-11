"""Recompute one archived A1 training candidate, including checkpoint selection."""
import argparse
import json
import pickle
from . import ROOT
from .inputs import load_input
from iclr_algorithm1_experiments import regularized_a1_v2 as core
from iclr_algorithm1_experiments.projection_distance_run import configure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',default='controlled_n100')
    parser.add_argument('--method',default='projected_graph_neural_global')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',default='outputs/training_replay')
    args=parser.parse_args();configure()
    index=json.loads((ROOT/'reference_checkpoints/index.json').read_text())
    selected=[r for r in index if r['bundle']==args.bundle and r['method']==args.method and
              r['seed']==args.seed and r['mode']=='full' and not r['zero']]
    if len(selected)!=1:parser.error('No unique complete-gradient positive-weight reference checkpoint.')
    reference=json.loads((ROOT/selected[0]['evaluation']).read_text())['selection']
    bundle=load_input(args.bundle)
    if bundle['kind']=='full':
        from iclr_algorithm1_experiments import regularized_a1_v2_full
    core.OUT=ROOT/args.output
    path=core.OUT/'bundles'/f'{args.bundle}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as handle:pickle.dump(bundle,handle)
    core.calibrate((args.bundle,args.method,args.seed))
    result=core.fit((args.bundle,args.method,args.seed,reference['strength'],reference['lr_index'],'full'))
    report=dict(bundle=args.bundle,method=args.method,seed=args.seed,iterations=result['iterations'],
                selected_state_matches=result['selected_state_sha256']==reference['selected_state_sha256'],
                validation_cost_error=result['best_validation_cost']-reference['best_validation_cost'],
                scope='one previously selected hyperparameter candidate; not a rerun of the full selection grid')
    (core.OUT/'replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['selected_state_matches']:
        raise RuntimeError('Training did not reproduce the checkpoint bit for bit; inspect software/device differences and validation discrepancy.')


if __name__=='__main__':main()
