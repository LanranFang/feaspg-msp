#!/usr/bin/env python3
r"""Narrow Figure 2 by 18% while retaining its original typeset font sizes.

The original asset is 7 by 3 inches at 0.90\linewidth. Exporting a 5.74 by
3 inch asset at 0.738\linewidth keeps the exact original PDF-to-paper scale.
Only plotting geometry changes; the frozen certificate summaries are unchanged.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / "extend_version" / "figures"
VARIANT_NAMES = {
    "original": "Confidence band + Hoeffding",
    "calibration_only": "Paired + Hoeffding",
    "evaluation_only": "Confidence band + Bernstein",
    "tightened": "Paired + Bernstein",
    "centered_hoeffding_eb": "Centered confidence band + Bernstein",
}
COLOR_PWL = "#087f8c"
COLOR_ORIGINAL = "#bc6840"
COLOR_ZERO = "#5b6573"
COLOR_REFERENCE = "#a94652"

def outward_decimal(value, places, upper=False):
    """Format a binary64 endpoint outward using exact integer arithmetic.

    Multiplying a float by 10**places before floor/ceil can itself round onto
    the wrong side of a decimal boundary. Fraction preserves the exact saved
    float, including such boundary cases and negative interval endpoints.
    """
    if not isinstance(places, int) or places < 0:
        raise ValueError('Expected a nonnegative decimal precision')
    factor = 10**places
    scaled = Fraction(float(value))*factor
    units = (-(-scaled.numerator//scaled.denominator) if upper else
             scaled.numerator//scaled.denominator)
    sign = '-' if units < 0 else ''
    integer, fractional = divmod(abs(units), factor)
    return (f'{sign}{integer}.{fractional:0{places}d}' if places else
            f'{sign}{integer}')


def interval_text(lower, upper, places):
    return (outward_decimal(lower, places),
            outward_decimal(upper, places, upper=True))


def make_figure(summary, dest, preview=False):
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.2,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    focus = summary[(summary.reservoirs == 3) & (summary.stages == 18)
                    & (summary.checkpoint == 400) & (summary.n_evaluation == 8192)]
    reference = focus.iloc[0]
    loss_low, loss_high = reference.loss_lower, reference.loss_upper
    loss_low_text, loss_high_text = interval_text(loss_low, loss_high, 2)
    # Keep the original physical left/right margins and inter-panel gap.
    # The plotting panels narrow, while font sizes and canvas height stay fixed.
    width_factor = .82
    original_panel_width = 7 * (.99 - .073) / 2.30
    original_gap = .30 * original_panel_width
    new_width = 7 * width_factor
    panel_width = (new_width - 7*.073 - 7*.01 - original_gap) / 2
    fig, axes = plt.subplots(1, 2, figsize=(new_width, 3))
    fig.subplots_adjust(left=7*.073/new_width, right=1-7*.01/new_width,
                        top=.855, bottom=.265, wspace=original_gap/panel_width)
    plots = []
    curves = [
        ('pwl', 'original', VARIANT_NAMES['original']+' (PWL)', COLOR_ORIGINAL, 's'),
        ('pwl', 'tightened', VARIANT_NAMES['tightened']+' (PWL)', COLOR_PWL, 'o'),
        ('zero', 'tightened', VARIANT_NAMES['tightened']+' (Zero)', COLOR_ZERO, '^'),
    ]
    for generator, variant, label, color, marker in curves:
        data = focus[(focus.generator == generator) & (focus.variant == variant)].sort_values('n_calibration')
        x = data.n_calibration.to_numpy()
        axes[0].plot(x, data.certificate_mean, label=label, color=color,
                     marker=marker, markersize=3.4, lw=1.35, zorder=4)
        axes[0].fill_between(x, data.certificate_q025, data.certificate_q975,
                             color=color, alpha=.18, lw=0, zorder=3)
        plots.append(data.assign(panel='a', display_label=label))
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks([2048, 8192, 32768], labels=['2,048', '8,192', '32,768'])
    axes[0].set_xlim(1600, 42000)
    axes[0].set_xlabel(r'Calibration trajectories $N_{\rm cal}$')
    axes[0].set_ylabel('Policy-loss upper bound')
    axes[0].set_title('(a) Calibration sweep', loc='left', pad=6)
    axes[0].set_ylim(top=1.28*max(data.certificate_q975.max() for data in plots))
    axes[0].legend(loc='upper right', frameon=True, framealpha=.9, edgecolor='none',
                   labelspacing=.30, borderpad=.30, handlelength=1.6)

    ablation_order = ('original', 'calibration_only', 'evaluation_only', 'tightened')
    ablation = focus[(focus.generator == 'pwl') & (focus.n_calibration == 32768)].set_index('variant').loc[list(ablation_order)]
    x = np.arange(4)
    bottom = np.zeros(4)
    components = [('raw_gap_mean', 'Raw gap', COLOR_PWL),
                  ('calibration_correction_mean', 'Calibration', '#dfac55'),
                  ('evaluation_correction_mean', 'Evaluation', '#8795b5')]
    for key, label, color in components:
        values = ablation[key].to_numpy()
        axes[1].bar(x, values, bottom=bottom, color=color, width=.57,
                    label=label, linewidth=0, zorder=3)
        bottom += values
    for index, value in enumerate(ablation.certificate_mean):
        axes[1].text(index, value+.65, f'{value:.1f}', ha='center', va='bottom', fontsize=7)
    axes[1].set_xticks(x, labels=[VARIANT_NAMES[name].replace('Confidence band', 'Confidence\nband').replace(' + ', '\n+ ')
                                for name in ablation_order])
    axes[1].tick_params(axis='x', length=0, labelsize=6.7)
    axes[1].set_xlabel(r'PWL; $N_{\rm cal}=32{,}768$')
    axes[1].set_ylabel('Certificate components')
    axes[1].set_title('(b) Correction ablation', loc='left', pad=6)
    axes[1].set_ylim(0, float(ablation.certificate_mean.max())*1.29)
    axes[1].legend(loc='upper right', ncol=3, frameon=False, fontsize=7.1,
                   columnspacing=.65, handlelength=.9, handletextpad=.35, borderaxespad=.1)
    plots.append(ablation.reset_index().assign(panel='b', display_label=lambda z: z.variant.map(VARIANT_NAMES)))
    for axis in axes:
        axis.axhspan(loss_low, loss_high, color=COLOR_REFERENCE, alpha=.17, lw=0, zorder=1)
        axis.axhline(loss_low, color=COLOR_REFERENCE, ls=':', lw=.65, zorder=5)
        axis.axhline(loss_high, color=COLOR_REFERENCE, ls=':', lw=.65, zorder=5)
        axis.grid(axis='y', alpha=.16, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    title = ('EXPLORATORY PREVIEW: 3 reused pilot repetitions' if preview else
             '3 coupled reservoirs, 18 stages; 200 fresh repetitions per setting')
    fig.suptitle(title, x=.53, y=.975, fontsize=8.5)
    fig.legend([Patch(facecolor=COLOR_REFERENCE, alpha=.25)],
               [f'Loss reference interval [{loss_low_text}, {loss_high_text}]; '
                r'$N_{\rm ev}=8{,}192$'],
               loc='lower center', bbox_to_anchor=(.53, .002), frameon=False, fontsize=7.2)
    for extension in ('pdf', 'png'):
        # Preserve the native figure size; tight cropping would change its LaTeX scale.
        fig.savefig(dest/f'certificate_hydro_narrow.{extension}', dpi=250)
    plt.close(fig)
    pd.concat(plots, ignore_index=True).to_csv(dest/'certificate_hydro_narrow_data.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=FIGURES / "certificate_hydro_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    summary = pd.read_csv(args.data)
    if not summary.repetitions.eq(200).all():
        raise ValueError("Expected the frozen 200-repetition certificate summary.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(summary, args.output_dir)
