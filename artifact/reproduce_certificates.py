"""Fresh evaluation of the frozen three-reservoir policies and continuations."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from . import ROOT
from synthetic_hydro_certificate import tightening_run as run
from synthetic_hydro_certificate.scaling_run import draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--first-repetition', type=int, default=0)
    parser.add_argument('--output', default='outputs/certificates')
    parser.add_argument('--save-path-statistics', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1 or args.first_repetition < 0:
        parser.error('Repetitions must be positive and the starting index nonnegative.')
    dest = ROOT / args.output; dest.mkdir(parents=True, exist_ok=True)
    h, engines, parameters, reference = run.load(3,18)
    rows = []; start = time.perf_counter()
    for rep in range(args.first_repetition, args.first_repetition + args.repetitions):
        calibration = draw(h, max(run.BUDGETS), np.random.SeedSequence([95731,3,18,0,rep]))
        calibrated = {(name,n): e.calibrate(calibration[:n].sum(0),n)
                      for name,e in engines.items() for n in run.BUDGETS}
        evaluation = draw(h, run.N_EV, np.random.SeedSequence([95731,3,18,1,rep]))
        block, payload = run.evaluate_block(h,engines,parameters,reference,calibrated,evaluation,rep)
        rows.extend(block)
        (dest / f'rep_{rep:04d}.json').write_text(json.dumps(block,indent=2)+'\n')
        if args.save_path_statistics:
            np.savez_compressed(dest / f'rep_{rep:04d}.npz',
                                calibration_paths=calibration,evaluation_paths=evaluation,**payload)
        print(f'Repetition {rep} finished; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest / 'results.csv',index=False)
    source = ROOT / 'reference_results/synthetic_hydro_certificate/outputs/tightened_v3/formal/results.csv'
    frozen = pd.read_csv(source)
    keys = ['repetition','checkpoint','generator','n_calibration','variant']
    left = frame.set_index(keys).sort_index()
    available = frozen.set_index(keys).sort_index()
    matched = left.index.intersection(available.index)
    checked = ['certificate','raw_gap_mean','calibration_correction','evaluation_correction']
    checked = [x for x in checked if x in left and x in available]
    errors = {c:float(np.max(np.abs(left.loc[matched,c]-available.loc[matched,c]))) for c in checked} if len(matched) else {}
    report = dict(rows=len(frame), matched_reference_rows=len(matched),
                  max_absolute_errors=errors, elapsed_seconds=time.perf_counter()-start,
                  source='unchanged frozen fitting objects; regenerated calibration and evaluation paths')
    (dest/'replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if errors and max(errors.values()) > 1e-8:
        raise RuntimeError('Certificate replay differs from the original record.')


if __name__ == '__main__':
    main()
