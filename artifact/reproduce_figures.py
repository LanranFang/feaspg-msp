"""Rebuild preprint Figures 2--7 and Table 1 from frozen experiment records."""
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from . import ROOT
from .plot_controlled_budget_only import draw
from .plot_certificate_narrow import make_figure as make_narrow_certificate_figure
from .plot_gradient_mechanism import draw as draw_gradient
from .plot_plateau import draw as draw_plateau
from .plot_preprint_budget import draw as draw_preprint_budget
from .plot_preprint_hydrology import draw as draw_preprint_hydrology
from .plot_latency import draw_latency
from synthetic_hydro_certificate.tightening_report import read_frame, summarize, make_figure, check_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/paper')
    args = parser.parse_args()
    dest = ROOT / args.output; dest.mkdir(parents=True, exist_ok=True)
    references = ROOT / 'reference_results'
    # Each plotting function applies its own original figure style.
    for plot in (draw_gradient, draw_plateau, draw_preprint_budget, draw_preprint_hydrology):
        plt.rcdefaults()
        plot(dest)
    plt.rcdefaults()
    draw_latency(dest, pd.read_csv(references/'iclr_algorithm1_experiments/outputs/regularized_a1_v2/report/controlled_all_methods.csv'))
    plt.rcdefaults()
    draw(references/'paper/controlled_budget_only_data.csv',dest,beside_table=True)
    raw = read_frame(references/'synthetic_hydro_certificate/outputs/tightened_v3/formal/results.csv')
    check_grid(raw,200)
    summary = summarize(raw)
    summary.to_csv(dest/'certificate_summary.csv',index=False)
    make_figure(summary,dest)
    make_narrow_certificate_figure(summary,dest)
    table = pd.read_csv(references/'iclr_algorithm1_experiments/outputs/regularized_a1_v2/report/real_main_tables.csv')
    table.to_csv(dest/'hydrology_main_table_data.csv',index=False)
    labels = {'seasonal_reference':'Seasonal', 'algorithm1_mlp':'FeasPG+MLP',
              'algorithm1_deepsets':'FeasPG+DeepSets','projected_shac':'Projected SHAC',
              'rl_ppo':'PPO','rl_sac':'SAC','hydrology_state_sddp':'Finite-state SDDP'}
    lines=['| Method | ONS (10^9) | CAMELS-BR |','|---|---:|---:|']
    tex=[r'\begin{tabular}{lrr}',r'\hline',r'Method & ONS ($10^9$) & CAMELS-BR \\',r'\hline']
    for method,label in labels.items():
        values=[]
        for case,scale in (('ons',1e9),('camels_br_12',1)):
            rows=table[(table.case==case)&(table.method==method)]
            if rows.empty:
                values.append('--'); continue
            row=rows.iloc[0]
            values.append(f'{row["mean"]/scale:.3f}' if method=='seasonal_reference' else
                          f'{row["mean"]/scale:.3f} ± {row["sd"]/scale:.3f}')
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
        tex.append(label+' & '+' & '.join('$'+v.replace('±',r'\pm')+'$' for v in values)+r' \\')
    tex.extend([r'\hline',r'\end{tabular}'])
    (dest/'table1.md').write_text('\n'.join(lines)+'\n')
    (dest/'table1.tex').write_text('\n'.join(tex)+'\n')
    focus=summary[(summary.checkpoint==400)&(summary.generator=='pwl')&(summary.n_calibration==32768)]
    report={'certificate_repetitions_per_setting':200,
            'original_certificate_mean':float(focus[focus.variant=='original'].certificate_mean.iloc[0]),
            'tightened_certificate_mean':float(focus[focus.variant=='tightened'].certificate_mean.iloc[0]),
            'source':'frozen experiment records; no retraining in this command',
            'paper_url': 'https://optimization-online.org/?p=37002',
            'figure_map': {'2':'gradient_mechanism.pdf', '3':'proposal_distance_mechanism.pdf',
                           '4':'controlled_budget_preprint.pdf', '5':'real_hydrology_preprint.pdf',
                           '6':'certificate_hydro.pdf', '7':'controlled_budget_latency.pdf'},
            'tables': 'Table 1 rebuilt from numerical records; other tables are documented in docs/PAPER_MAP.md'}
    (dest/'rebuild_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
