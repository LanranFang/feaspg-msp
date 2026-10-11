"""Regenerate the model-generated hydrology paths from fitted public-data models."""
import argparse
from pathlib import Path
import numpy as np
from . import ROOT
from real_hydro.stochastic import PeriodicARModel


def arrays_for(case, role):
    """Use the original 2,500-path pool plus the separately seeded extension."""
    model = PeriodicARModel.load(
        ROOT / 'real_data_hydro_experiments/data/processed' / case / 'stochastic_model')
    if role == 'fit':
        batches = [model.sample('fit',2500,48,seed) for seed in (61101,61606)]
    else:
        n, seed = {'validation':(500,61202), 'test':(4096,61303)}[role]
        batches = [model.sample(role,n,48,seed)]
    import dataclasses
    arrays = {}
    for field in dataclasses.fields(batches[0]):
        values = [getattr(b,field.name) for b in batches]
        if isinstance(values[0],np.ndarray):
            arrays[field.name] = values[0] if field.name == 'calendar' else np.concatenate(values)
    return arrays


def materialize(case, role):
    path = ROOT / 'outputs/generated_inputs' / f'{case}_{role}.npz'
    if not path.exists():
        arrays = arrays_for(case,role)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**arrays)
        temporary.replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('ons_subsystems','camels_br_12'), required=True)
    parser.add_argument('--check-exported', action='store_true')
    args = parser.parse_args()
    for role in ('fit','validation','test'):
        path = materialize(args.case,role)
        if args.check_exported:
            with np.load(ROOT/'inputs/real'/f'{args.case}_{role}.npz') as original, np.load(path) as generated:
                for name in original.files:
                    np.testing.assert_array_equal(original[name],generated[name],err_msg=f'{args.case}/{role}/{name}')
        print(f'Generated {args.case}/{role}'+(' and verified every array' if args.check_exported else ''),flush=True)
