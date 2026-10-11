"""Re-evaluate a saved A1 policy on the original held-out paths."""
import argparse
import json
import numpy as np
import torch
from . import ROOT
from .inputs import load_input
from iclr_algorithm1_experiments import regularized_a1_v2 as core
from iclr_algorithm1_experiments.projection_distance_run import configure, state_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='controlled_n1000')
    parser.add_argument('--method', default='projected_graph_neural_global')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', default='full')
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--output', default='outputs/checkpoint_evaluation.json')
    args = parser.parse_args()
    configure()
    index = json.loads((ROOT / 'reference_checkpoints/index.json').read_text())
    chosen = [e for e in index if all(e[k] == getattr(args, k) for k in ('bundle','method','seed','mode','zero'))]
    if len(chosen) != 1:
        parser.error('No unique matching checkpoint; consult reference_checkpoints/index.json.')
    entry = chosen[0]
    bundle = load_input(args.bundle)
    if bundle['kind'] == 'full':
        from iclr_algorithm1_experiments import regularized_a1_v2_full
    actor = core.build(bundle, args.method, args.seed)
    state = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    actor.load_state_dict(state)
    expected = json.loads((ROOT / entry['evaluation']).read_text())
    assert state_hash(state) == expected['selection']['selected_state_sha256']
    metrics, costs = core.evaluate(actor, bundle['test'], bundle, raw=True)
    with np.load((ROOT / entry['checkpoint']).with_name('path_costs.npz')) as data:
        frozen = data['test_costs']
    max_error = float(np.max(np.abs(costs - frozen)))
    agrees = bool(np.allclose(costs, frozen, rtol=1e-6, atol=1e-6))
    report = dict(bundle=args.bundle, method=args.method, seed=args.seed,
                  test_cost=metrics['mean_cost_float64'],
                  frozen_test_cost=float(frozen.mean()), max_path_cost_error=max_error,
                  matches_reference_within_tolerance=agrees)
    path = ROOT / args.output; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not agrees:
        raise RuntimeError('Checkpoint replay differs from the recorded path costs.')


if __name__ == '__main__':
    main()
