"""Report completed fresh validation; keep old-pilot previews visibly separate.

Default execution requires the frozen 200-repetition run to be complete.
``--preview-pilot`` renders only a labeled preview under preview_pilot/ and
does not create formal results or overwrite manuscript assets.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/tightened_v3'
OLD_OUT = ROOT / 'outputs/scaling_v2'
VARIANTS = ('original', 'calibration_only', 'evaluation_only', 'tightened',
            'centered_hoeffding_eb')
VARIANT_NAMES = {
    'original': 'Confidence band + Hoeffding',
    'calibration_only': 'Paired + Hoeffding',
    'evaluation_only': 'Confidence band + Bernstein',
    'tightened': 'Paired + Bernstein',
    'centered_hoeffding_eb': 'Centered confidence band + Bernstein',
}
VARIANT_ZH = {
    'original': '置信带校准 + Hoeffding',
    'calibration_only': '配对校准 + Hoeffding',
    'evaluation_only': '置信带校准 + Bernstein',
    'tightened': '配对校准 + Bernstein',
    'centered_hoeffding_eb': '中心化置信带校准 + Bernstein',
}
KEYS = ['reservoirs', 'stages', 'checkpoint', 'generator',
        'n_calibration', 'n_evaluation', 'variant']
METRICS = [
    'raw_gap', 'calibration_correction', 'evaluation_correction', 'certificate',
    'certificate_fraction_cost', 'excess_lower', 'excess_upper',
    'raw_envelope', 'joint_corrected_range', 'evaluation_range_used',
    'joint_radius', 'calibration_cap', 'corrected_sample_variance',
    'raw_sample_variance', 'inner_seconds',
]
COLOR_PWL = '#087f8c'
COLOR_ORIGINAL = '#bc6840'
COLOR_ZERO = '#5b6573'
COLOR_REFERENCE = '#a94652'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n')


def records(frame):
    """Preserve binary64 endpoints and represent missing values as JSON null."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient='records')


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


def read_frame(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    required = set(KEYS + ['repetition', 'raw_gap', 'calibration_correction',
                          'evaluation_correction', 'certificate', 'policy_cost',
                          'loss_lower', 'loss_upper', 'max_arithmetic_interval',
                          'covers_reference_upper', 'below_reference_lower'])
    if missing := required.difference(frame.columns):
        raise RuntimeError(f'Missing result columns: {sorted(missing)}')
    if frame.duplicated(KEYS + ['repetition']).any():
        raise RuntimeError('Duplicate setting/repetition result.')
    # Hoeffding rows intentionally leave the unused variance field empty.
    numeric = frame.drop(columns=['variance_used'], errors='ignore').select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError('Nonfinite result value.')
    if (frame.loss_lower > frame.loss_upper).any() or (frame.policy_cost <= 0).any():
        raise RuntimeError('Invalid reference interval or policy cost.')
    for flag in ('covers_reference_upper', 'below_reference_lower'):
        if frame[flag].dtype != bool:
            converted = frame[flag].astype(str).str.lower().map({'true': True, 'false': False})
            if converted.isna().any():
                raise RuntimeError(f'Invalid Boolean result field: {flag}')
            frame[flag] = converted.astype(bool)
    cover = frame.certificate >= frame.loss_upper
    fail = frame.certificate < frame.loss_lower
    if not (cover == frame.covers_reference_upper).all() or not (fail == frame.below_reference_lower).all():
        raise RuntimeError('Stored reference coverage flags disagree with certificates.')
    # Separate outward rounding of displayed components can differ slightly
    # from the upward-rounded total; this is not a statistical discrepancy.
    component_sum = frame[['raw_gap', 'calibration_correction', 'evaluation_correction']].sum(axis=1)
    if not np.allclose(frame.certificate, component_sum, rtol=1e-11, atol=1e-8):
        raise RuntimeError('Certificate decomposition is inconsistent.')
    frame['certificate_fraction_cost'] = frame.certificate / frame.policy_cost
    frame['excess_lower'] = frame.certificate - frame.loss_upper
    frame['excess_upper'] = frame.certificate - frame.loss_lower
    return frame


def check_grid(frame, repetitions, reservoirs=(3,), stages=(18,)):
    expected = pd.MultiIndex.from_product(
        [reservoirs, stages, (40, 200, 400), ('zero', 'pwl'),
         (2048, 8192, 32768), (8192,), VARIANTS], names=KEYS)
    observed = pd.MultiIndex.from_frame(frame[KEYS].drop_duplicates())
    if len(observed) != len(expected) or len(expected.difference(observed)):
        raise RuntimeError('The predeclared configuration grid is incomplete or changed.')
    wanted = set(range(repetitions))
    for key, group in frame.groupby(KEYS, sort=False):
        if len(group) != repetitions or set(group.repetition) != wanted:
            raise RuntimeError(f'Incomplete independent repetition set for {key}.')
        for column in ('policy_cost', 'loss_lower', 'loss_upper'):
            if group[column].nunique() != 1:
                raise RuntimeError(f'Reference changed across repetitions: {column}.')
    return len(expected)


def summarize(frame, fresh=True):
    rows = []
    for key, group in frame.groupby(KEYS, sort=True):
        n = len(group)
        sufficient = int(group.covers_reference_upper.sum())
        failure = int(group.below_reference_lower.sum())
        row = dict(zip(KEYS, key))
        row.update(
            repetitions=n, reference_upper_covers=sufficient,
            definite_failures=failure, reference_unresolved=n-sufficient-failure,
            empirical_coverage_lower=sufficient/n,
            empirical_coverage_upper=1-failure/n,
            # These intervals describe the sufficient reference-upper event.
            # Their lower endpoint lower-bounds true-loss coverage probability.
            reference_upper_probability_ci_low=(
                0.0 if sufficient == 0 else
                float(beta_distribution.ppf(.025, sufficient, n-sufficient+1))) if fresh else None,
            reference_upper_probability_ci_high=(
                1.0 if sufficient == n else
                float(beta_distribution.ppf(.975, sufficient+1, n-sufficient))) if fresh else None,
            policy_cost=float(group.policy_cost.iloc[0]),
            loss_lower=float(group.loss_lower.iloc[0]),
            loss_upper=float(group.loss_upper.iloc[0]),
            max_arithmetic_interval=float(group.max_arithmetic_interval.max()),
        )
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric]
            row.update({f'{metric}_mean': float(values.mean()),
                        f'{metric}_sd': float(values.std(ddof=1)),
                        f'{metric}_q025': float(values.quantile(.025)),
                        f'{metric}_q975': float(values.quantile(.975))})
        rows.append(row)
    return pd.DataFrame(rows)


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
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    fig.subplots_adjust(left=.073, right=.99, top=.855, bottom=.225, wspace=.30)
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
    axes[1].set_xticks(x, labels=[VARIANT_NAMES[name].replace(' + ', '\n+ ')
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
        # Preserve the requested seven-by-three-inch physical figure size.
        fig.savefig(dest/f'certificate_hydro.{extension}', dpi=250)
    plt.close(fig)
    pd.concat(plots, ignore_index=True).to_csv(dest/'certificate_hydro_data.csv', index=False)


def tex_table(caption, label, columns, header, rows):
    return '\n'.join([
        r'\begin{table}[H]', r'\caption{'+caption+'}',
        r'\label{'+label+'}', r'\centering\small',
        r'\begin{tabular}{@{}'+columns+r'@{}}', r'\toprule',
        header+r'\\', r'\midrule', *[row+r'\\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
    ])


def make_tables(summary, pilot_summary, dest):
    if not (summary.repetitions == 200).all():
        raise RuntimeError('Formal table captions require 200 fresh repetitions.')
    focus = summary[(summary.generator == 'pwl') & (summary.checkpoint == 400)
                    & (summary.n_calibration == 32768)].set_index('variant').loc[list(VARIANTS)]
    reference = focus.iloc[0]
    loss_low_text, loss_high_text = interval_text(reference.loss_lower, reference.loss_upper, 5)
    rows = []
    for variant, row in focus.iterrows():
        rows.append(VARIANT_NAMES[variant]+' & '+' & '.join(
            f'{row[key]:.3f}' for key in ('raw_gap_mean', 'calibration_correction_mean',
                                         'evaluation_correction_mean', 'certificate_mean'))
                    +f' & {100*row.certificate_fraction_cost_mean:.2f}'+r'\%')
    caption = (
        r'Correction ablation for three coupled reservoirs and 18 stages, with a '
        r'frozen PWL continuation and update-400 policy. Means over 200 fresh '
        r'calibration/evaluation repetitions; $N_{\rm cal}=32{,}768$ and '
        r'$N_{\rm ev}=8{,}192$. The continuous policy loss is enclosed by '
        f'$[{loss_low_text},{loss_high_text}]$; '
        r'this is a reference interval, not an exact loss value. Confidence band '
        r'denotes conditional-mean confidence-band calibration; Paired denotes '
        r'independent-binary paired calibration; Bernstein denotes empirical '
        r'Bernstein evaluation. The last row '
        r'uses paired-contrast oscillation with coordinatewise Hoeffding calibration '
        r'and empirical Bernstein evaluation. No minimum over rows is selected.')
    (dest/'ablation_table.tex').write_text(tex_table(
        caption, 'tab:coupled-ablation', 'lrrrrr',
        r'Method & Raw gap & Calibration & Evaluation & $U$ & $U/J$', rows))

    selected = summary[(summary.variant == 'tightened') & (summary.n_calibration == 32768)]
    rows = []
    for checkpoint in (40, 200, 400):
        group = selected[selected.checkpoint == checkpoint].set_index('generator')
        pwl, zero = group.loc['pwl'], group.loc['zero']
        low_text, high_text = interval_text(pwl.loss_lower, pwl.loss_upper, 3)
        rows.append(
            f'{checkpoint} & {pwl.policy_cost:.3f} & '
            f'$[{low_text},{high_text}]$ & '
            f'${pwl.certificate_mean:.3f}\\pm{pwl.certificate_sd:.3f}$ & '
            f'${zero.certificate_mean:.3f}\\pm{zero.certificate_sd:.3f}$ & '
            f'{100*pwl.certificate_fraction_cost_mean:.2f}'+r'\%')
    (dest/'checkpoint_table.tex').write_text(tex_table(
        r'Predeclared training checkpoints for three coupled reservoirs and 18 '
        r'stages, using Paired + Bernstein. The fitted PWL continuation is held '
        r'fixed across checkpoints. Certificates show mean $\pm$ standard '
        r'deviation over 200 fresh repetitions at $N_{\rm cal}=32{,}768$ and '
        r'$N_{\rm ev}=8{,}192$. The loss column is a verified continuous-loss '
        r'interval. Coverage statements are marginal for each fixed setting.',
        'tab:coupled-checkpoints', 'rrrrrr',
        r'Update & $J$ & Loss reference & PWL $U$ & Zero $U$ & PWL $U/J$', rows))

    scale_rows = []
    scale_records = []
    for (reservoirs, stages), group in pilot_summary[
            (pilot_summary.generator == 'pwl') & (pilot_summary.checkpoint == 400)
            & (pilot_summary.n_calibration == 32768)].groupby(['reservoirs', 'stages']):
        methods = group.set_index('variant')
        old, new = methods.loc['original'], methods.loc['tightened']
        scale_rows.append(
            f'{int(reservoirs)} & {int(stages)} & '
            f'${100*old.certificate_fraction_cost_mean:.2f}\\pm'
            f'{100*old.certificate_fraction_cost_sd:.2f}$ & '
            f'${100*new.certificate_fraction_cost_mean:.2f}\\pm'
            f'{100*new.certificate_fraction_cost_sd:.2f}$')
        scale_records.append(dict(
            reservoirs=int(reservoirs), stages=int(stages), repetitions=3,
            classification='exploratory reuse of old pilot data',
            policy_cost=old.policy_cost, loss_lower=old.loss_lower, loss_upper=old.loss_upper,
            original_certificate_mean=old.certificate_mean,
            tightened_certificate_mean=new.certificate_mean,
            original_percent_mean=100*old.certificate_fraction_cost_mean,
            original_percent_sd=100*old.certificate_fraction_cost_sd,
            tightened_percent_mean=100*new.certificate_fraction_cost_mean,
            tightened_percent_sd=100*new.certificate_fraction_cost_sd))
    (dest/'scale_table.tex').write_text(tex_table(
        r'Exploratory scale comparison reusing three old pilot repetitions per '
        r'model, not fresh coverage validation. Entries are mean $\pm$ standard '
        r'deviation of PWL $U/J$ in percent, using update 400, '
        r'$N_{\rm cal}=32{,}768$ and $N_{\rm ev}=8{,}192$. The fitted policies, '
        r'continuations and trajectories are unchanged between methods.',
        'tab:coupled-scale', 'rrrr',
        r'Reservoirs & Stages & '
        r'\shortstack{Confidence band + Hoeffding\\$U/J$ (\%)} & '
        r'\shortstack{Paired + Bernstein\\$U/J$ (\%)}', scale_rows))
    scale_frame = pd.DataFrame(scale_records)
    scale_frame.to_csv(dest/'scale_comparison.csv', index=False)
    return scale_frame


def make_report(summary, scale, protocol, completion, frame, dest):
    focus = summary[(summary.checkpoint == 400) & (summary.n_calibration == 32768)]
    pwl = focus[focus.generator == 'pwl'].set_index('variant')
    old, new = pwl.loc['original'], pwl.loc['tightened']
    zero = focus[(focus.generator == 'zero') & (focus.variant == 'tightened')].iloc[0]
    improvement = 1-new.certificate_mean/old.certificate_mean
    cal_improvement = 1-new.calibration_correction_mean/old.calibration_correction_mean
    ev_improvement = 1-new.evaluation_correction_mean/old.evaluation_correction_mean
    loss_low_text, loss_high_text = interval_text(new.loss_lower, new.loss_upper, 5)
    excess_low_text, excess_high_text = interval_text(new.excess_lower_mean, new.excess_upper_mean, 4)
    if (summary.reference_upper_covers == 200).all():
        coverage_text = (
            f'全部 {len(summary)} 个固定设置各有 200/200 次证书超过参考损失区间上端。'
            f'该充分覆盖事件的点态 95% Clopper–Pearson 区间为 '
            f'[{100*summary.reference_upper_probability_ci_low.min():.2f}%, 100%]。')
    else:
        coverage_text = (
            f'共 {len(summary)} 个固定设置；超过参考区间上端的次数最少为 '
            f'{int(summary.reference_upper_covers.min())}/200。各设置的点态 '
            'Clopper–Pearson 区间完整保存在 summary.csv。')
    lines = [
        '# 三库十八阶段：校准与评估修正收紧结果', '',
        f'PWL 主设置的平均证书由 **{old.certificate_mean:.4f}** 降至 '
        f'**{new.certificate_mean:.4f}**，下降 **{100*improvement:.1f}%**。'
        f'连续策略损失的可验证参考区间为 **[{loss_low_text}, {loss_high_text}]**。', '',
        '这是已经完成的独立正式验证：3 个耦合水库、18 个阶段，'
        '每个固定设置 200 次独立校准/评估重复；校准预算为 2,048、8,192、32,768，'
        '评估预算固定为 8,192。主方法事先固定为配对校准 + Bernstein，主检查点为第 400 次更新。'
        '图表中的 Bernstein 指 empirical Bernstein；Paired 指独立二元扰动的配对校准。', '',
        '## 保持不变的部分', '',
        '固定策略、三个训练检查点、PWL 延续值、模型和内层优化问题均沿用 scaling_v2；'
        '没有重新训练，也没有根据新证书或参考最优值挑选策略。正式验证使用新的 '
        '`SeedSequence([95731,3,18,block,rep])`；block 0 为校准，block 1 为评估。'
        '所有包络在打开评估数据前冻结。', '',
        '## 主设置的修正消融', '',
        '下表固定 PWL、更新 400，Ncal=32,768、Nev=8,192；数值为 200 次重复平均。', '',
        '| 方法 | 原始配对差值 | 校准修正 | 评估修正 | 证书 U | U/J |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        row = pwl.loc[variant]
        lines.append(f'| {VARIANT_ZH[variant]} | {row.raw_gap_mean:.4f} | '
                     f'{row.calibration_correction_mean:.4f} | {row.evaluation_correction_mean:.4f} | '
                     f'{row.certificate_mean:.4f} | {100*row.certificate_fraction_cost_mean:.2f}% |')
    lines += [
        '', f'主方法校准修正下降 {100*cal_improvement:.1f}%，评估修正下降 '
        f'{100*ev_improvement:.1f}%。原始配对差值保持一致。'
        f'同预算下，Paired + Bernstein (Zero) 证书为 {zero.certificate_mean:.4f}。', '',
        f'Paired + Bernstein (PWL) 证书占策略成本约 {100*new.certificate_fraction_cost_mean:.2f}%；'
        f'相对参考损失区间，其平均超额宽度位于 '
        f'[{excess_low_text}, {excess_high_text}]。'
        '参考区间没有变成精确最优值；不能将其端点或中点当成真实损失点估计。', '',
        '文中、图例及表格展示的验证区间端点均以精确分数运算向外舍入；'
        '图中的参考横带和覆盖判定使用未作显示舍入的端点。均值和标准差使用普通显示舍入。', '',
        '## 覆盖率的准确解释', '', coverage_text, '',
        '这里统计的是 U ≥ 参考损失区间上端，它是覆盖真实损失的充分条件。'
        '因此该事件的概率及其置信下端提供真实覆盖概率的保守下界；'
        '所列 Clopper–Pearson 区间不是未知精确真实损失覆盖率的等式估计。'
        '位于参考区间内部的证书归为无法判定，低于区间下端的证书才是确定漏覆盖。', '',
        f'确定漏覆盖共 {int(summary.definite_failures.sum())} 行；'
        f'参考区间内无法判定共 {int(summary.reference_unresolved.sum())} 行。'
        f'总计 {len(frame):,} 行来自每设置 200 次重复；同一次重复的预算、检查点、'
        '延续值和消融共享数据，不能将全部行数当作独立覆盖试验次数。'
        '置信区间和名义 95% 保证均为固定设置的点态／边际陈述，'
        '没有声称所有 90 个设置同时覆盖。', '',
        '## 图与表', '',
        '- `certificate_hydro.pdf` / `.png`：左图为校准样本扫描，右图为 PWL 修正分解；'
        '曲线阴影为 200 次重复的中央 95% 经验分位区间，并非均值置信区间。'
        '红色横带为参考损失区间，不是误差条。',
        '- `certificate_hydro_data.csv`：作图所用分组统计。',
        '- `ablation_table.tex`：五种修正方案；不根据结果选择最小值。',
        '- `checkpoint_table.tex`：三个预定训练检查点，Paired + Bernstein 的 PWL 与 Zero 证书。',
        '- `scale_table.tex` / `scale_comparison.csv`：九组规模的旧数据诊断，各仅三次重复，'
        '只作探索性对照，不参与正式覆盖验证。',
        '- `summary.csv`：每个设置的均值、标准差、2.5%/97.5% 分位数、充分覆盖次数及区间。', '',
        '## 适用范围与验证', '',
        '联合校准球使用当前合成模型“阶段之间独立的二元冲击”结构；'
        '多个水库在同一阶段共享冲击是允许的。该收紧不能直接外推到时间相关来水、'
        '连续条件特征或 Figure 3 的非凸调度模型。阶段独立性是新增证明的实质条件。', '',
        '评估项对“原始配对差值 + 随路径变化的校准修正”整体应用经验 Bernstein 界，'
        '包括两者的方差和协方差。均值和样本方差用浮点数的精确二进制分数累计；'
        '连续内层问题、向外舍入和预先冻结的结构包络保持有效。', '',
        f'正式运行最大数值区间宽度为 {frame.max_arithmetic_interval.max():.3g}，'
        '低于预设的 1e-6 上限。没有丢弃失败样本；正式完成记录与源码、拟合文件哈希已核对。', '',
        '复现报告：`python -m synthetic_hydro_certificate.tightening_report`。'
        '报告只写入当前输出目录，不自动覆盖论文图或主 TeX 文件。',
    ]
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    result = dict(
        classification=protocol['classification'], protocol=protocol, completion=completion,
        settings=len(summary), total_rows=len(frame), repetitions_per_setting=200,
        primary_reference=dict(policy_cost=new.policy_cost, loss_lower=new.loss_lower,
                               loss_upper=new.loss_upper),
        primary_improvement_fraction=float(improvement),
        calibration_improvement_fraction=float(cal_improvement),
        evaluation_improvement_fraction=float(ev_improvement),
        reference_upper_coverage=dict(
            interpretation='Sufficient coverage event; its probability lower-bounds true-loss coverage.',
            interval='pointwise two-sided 95% Clopper-Pearson for the sufficient event',
            minimum_count=int(summary.reference_upper_covers.min()),
            minimum_probability_lower_bound=float(summary.reference_upper_probability_ci_low.min()),
            definite_failures=int(summary.definite_failures.sum()),
            unresolved=int(summary.reference_unresolved.sum())),
        focus=records(focus), summary=records(summary), exploratory_scale=records(scale),
        provenance={
            'formal_results_sha256': hashlib.sha256((dest/'formal/results.csv').read_bytes()).hexdigest(),
            'pilot_results_sha256': hashlib.sha256((dest/'pilot/results.csv').read_bytes()).hexdigest(),
            'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'new_seed_namespace': 95731, 'fitting_unchanged': True,
        })
    write_json(dest/'results.json', result)
    print(f'Formal settings: {len(summary)}; rows: {len(frame)}; '
          f'PWL U: {old.certificate_mean:.6f} -> {new.certificate_mean:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-pilot', action='store_true',
                        help='Render a clearly labeled old-data preview in preview_pilot/.')
    args = parser.parse_args()
    pilot = read_frame(OUT/'pilot/results.csv')
    check_grid(pilot, 3, reservoirs=(1, 2, 3), stages=(6, 12, 18))
    pilot_summary = summarize(pilot, fresh=False)
    if args.preview_pilot:
        dest = OUT/'preview_pilot'
        dest.mkdir(parents=True, exist_ok=True)
        focus = pilot_summary[(pilot_summary.reservoirs == 3) & (pilot_summary.stages == 18)]
        make_figure(focus, dest, preview=True)
        focus.to_csv(dest/'summary.csv', index=False)
        (dest/'PREVIEW.md').write_text(
            '这是旧预实验三次重复的排版预览，不是新的正式结果。\n'
            '正式报告必须等待 formal/COMPLETED.json 和完整 results.csv。\n')
        print(f'Exploratory preview only: {dest}')
        return
    formal = OUT/'formal'
    if not (formal/'COMPLETED.json').exists() or not (formal/'results.csv').exists():
        raise RuntimeError('Formal run is incomplete; no formal report or paper figure was written.')
    protocol = json.loads((formal/'protocol.json').read_text())
    completion = json.loads((formal/'COMPLETED.json').read_text())
    if protocol['repetitions'] != 200 or completion['repetitions'] != 200 or completion['failures'] != 0:
        raise RuntimeError('Formal repetition/failure record differs from the declared design.')
    for name, expected in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    if hashlib.sha256((OLD_OUT/'r3_t18/fit.npz').read_bytes()).hexdigest() != protocol['fitting_sha256']:
        raise RuntimeError('Frozen fitting artifact changed.')
    frame = read_frame(formal/'results.csv')
    settings = check_grid(frame, 200)
    if settings != 90 or len(frame) != 18000 or len(frame) != completion['rows']:
        raise RuntimeError('Formal result count is inconsistent.')
    summary = summarize(frame)
    summary.to_csv(OUT/'summary.csv', index=False)
    make_figure(summary, OUT)
    scale = make_tables(summary, pilot_summary, OUT)
    make_report(summary, scale, protocol, completion, frame, OUT)


if __name__ == '__main__':
    main()
