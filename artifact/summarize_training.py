"""Summarize newly selected A1 runs; do not mix them with archived results."""
import argparse
import json
import pandas as pd
from . import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='iclr_algorithm1_experiments/outputs/regularized_a1_v2/selected')
    parser.add_argument('--output',default='outputs/new_training_summary')
    args=parser.parse_args();rows=[]
    for path in sorted((ROOT/args.input).rglob('evaluation.json')):
        record=json.loads(path.read_text());choice=record['selection']
        row={k:record[k] for k in ('bundle','method','seed','mode','zero')}
        row.update(n_fit=choice['n_fit'],strength=choice['strength'],lambda_ref=choice['lambda_ref'],
                   learning_rate=choice['learning_rate'],selected_iteration=choice['selected_iteration'],
                   validation_cost=choice['best_validation_cost'],
                   test_cost=record['test']['mean_cost_float64'],
                   fitting_seconds=record['total_fit_seconds'])
        if 'historical' in record:row['historical_cost']=record['historical']['mean_cost_float64']
        rows.append(row)
    if not rows:parser.error('No new selected evaluations found. Run training and evaluation first.')
    dest=ROOT/args.output;dest.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'per_seed.csv',index=False)
    summary=frame.groupby(['bundle','method','mode','zero','n_fit'],dropna=False).agg(
        seeds=('seed','nunique'),test_cost_mean=('test_cost','mean'),test_cost_sd=('test_cost','std'),
        mean_fitting_seconds=('fitting_seconds','mean')).reset_index()
    summary.to_csv(dest/'summary.csv',index=False)
    print(summary.to_string(index=False))
    if (summary.seeds!=5).any():print('Some groups have fewer than five seeds; do not label them complete paper replications.')


if __name__=='__main__':main()
