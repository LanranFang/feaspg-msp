"""Restore frozen inputs without depending on the original research directory."""
import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import pickle

import numpy as np
from . import ROOT


def decode(x):
    if isinstance(x, list):
        return [decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if '__class__' in x:
        module, name = x['__class__'].split(':')
        cls = getattr(importlib.import_module(module), name)
        return cls(**{k: decode(v) for k, v in x['fields'].items()})
    if '__tuple__' in x:
        return tuple(decode(v) for v in x['__tuple__'])
    if '__array__' in x:
        return np.asarray(x['__array__'], dtype=x['dtype']).reshape(x['shape'])
    if '__path__' in x:
        return ROOT / x['__path__']
    return {k: decode(v) for k, v in x.items()}


def array_hashes(data):
    result = {}
    for field in dataclasses.fields(data):
        value = getattr(data, field.name)
        if isinstance(value, np.ndarray):
            result[field.name] = hashlib.sha256(
                str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
            ).hexdigest()
    return result


def recipes():
    return json.loads((ROOT / 'inputs/bundles.json').read_text())


def load_input(name, verify=True):
    from efficiency_switching_hydro.data import generate_dataset
    from real_hydro.dataset import TrajectoryBatch
    recipe = recipes()[name]
    bundle = {k: decode(v) for k, v in recipe.items()
              if not isinstance(v, dict) or '__dataset__' not in v}
    for role, spec in recipe.items():
        if not isinstance(spec, dict) or '__dataset__' not in spec:
            continue
        if spec['__dataset__'] == 'synthetic':
            pool = generate_dataset(spec['name'], spec['n_generate'], spec['seed'], bundle['system'])
            data = pool.take(np.arange(spec['n']), name=spec['name'])
        else:
            if 'generator' in spec:
                from .real_inputs import materialize
                input_path = materialize(spec['case'], spec['role'])
            else:
                input_path = ROOT / spec['file']
            with np.load(input_path, allow_pickle=False) as arrays:
                data = TrajectoryBatch(name=spec['name'], seed=spec['seed'],
                                       **{k: arrays[k] for k in arrays.files})
            data = data.take(np.arange(spec['n']), name=spec['name'])
        if verify and array_hashes(data) != spec['hashes']:
            raise RuntimeError(f'Input fingerprint mismatch: {name}/{role}')
        bundle[role] = data
    return bundle


def prepare_bundles(output=None):
    output = Path(output) if output else ROOT / 'iclr_algorithm1_experiments/outputs/regularized_a1_v2'
    dest = output / 'bundles'
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in recipes():
        path = dest / f'{name}.pkl'
        spec = recipes()[name]
        if not path.exists():
            bundle = load_input(name)
            # Locally generated pickle files preserve the original training API;
            # the distributed artifact contains JSON/NPZ inputs, not pickles.
            with path.open('wb') as handle:
                pickle.dump(bundle, handle)
            print(f'Prepared and verified {name}', flush=True)
        entries.append(dict(name=name, kind=spec['kind'], methods=list(spec['settings']),
                            n_fit=spec['fit']['n'], horizon=spec['horizon'],
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output / 'manifest.json').write_text(json.dumps({'bundles': entries}, indent=2) + '\n')
    (output / 'protocol.json').write_text((ROOT / 'inputs/regularized_protocol.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', choices=list(recipes()))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only or args.bundle:
        for name in ([args.bundle] if args.bundle else recipes()):
            load_input(name)
            print(f'All arrays verified: {name}', flush=True)
    else:
        prepare_bundles()
