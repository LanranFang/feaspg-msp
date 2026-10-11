"""Transparent comparison of all nine coupled-hydro pilot cells."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .scaling_run import OUT,BUDGETS,N_EV,REPETITIONS


def main():
    frames=[];audit=[]
    for r in (1,2,3):
        for t in (6,12,18):
            d=OUT/f'r{r}_t{t}'
            done=json.loads((d/'COMPLETED.json').read_text())
            f=pd.read_csv(d/'results.csv')
            if len(f)!=54 or len(f)!=done['rows']:raise RuntimeError('Incomplete pilot cell')
            frames.append(f);audit.extend(json.loads((d/'continuous_audits.json').read_text()))
    f=pd.concat(frames,ignore_index=True)
    if f.duplicated(['reservoirs','stages','repetition','checkpoint','generator','n_calibration']).any():raise RuntimeError('Duplicate result cell')
    f.to_csv(OUT/'all_results.csv',index=False)
    group=['reservoirs','stages','checkpoint','generator','n_calibration','n_evaluation']
    summary=f.groupby(group).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    selected=summary[(summary.checkpoint==400)&(summary.n_calibration==max(BUDGETS))]
    table=[]
    for (r,t),rows in selected.groupby(['reservoirs','stages']):
        g=rows.set_index('generator');a=g.loc['pwl'];z=g.loc['zero']
        table.append(dict(reservoirs=int(r),stages=int(t),policy_cost=a.policy_cost,
            loss_lower=a.loss_lower,loss_upper=a.loss_upper,pwl=a.certificate,zero=z.certificate,
            pwl_percent=100*a.certificate_fraction_cost,zero_percent=100*z.certificate_fraction_cost,
            raw_gap=a.raw_gap,calibration=a.calibration_correction,evaluation=a.evaluation_correction,
            inner_seconds=a.inner_seconds))
    table=pd.DataFrame(table);table.to_csv(OUT/'scale_comparison.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors={1:'#4878a8',2:'#d68136',3:'#249083'}
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
    for ax,name in zip(axes,['pwl','zero']):
        for r,color in colors.items():
            sub=selected[(selected.reservoirs==r)&(selected.generator==name)].sort_values('stages')
            ax.plot(sub.stages,sub.certificate_fraction_cost*100,'o-',color=color,label=f'{r} reservoir'+('s' if r>1 else ''),lw=1.7)
            ax.fill_between(sub.stages,100*sub.loss_lower/sub.policy_cost,100*sub.loss_upper/sub.policy_cost,color=color,alpha=.18)
        ax.set_xticks([6,12,18]);ax.set_xlabel('Stages');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        ax.set_title('Fitted PWL continuation' if name=='pwl' else 'Zero continuation')
    axes[0].set_ylabel('Certificate / expected policy cost (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles+[Patch(facecolor='gray',alpha=.2)],labels+['Reference loss intervals'],
               loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Scale pilot: final checkpoint; 32,768 calibration + 8,192 evaluation paths',fontsize=10)
    fig.tight_layout(rect=(0,.10,1,.95))
    for suffix in ('pdf','png'):fig.savefig(OUT/f'scale_comparison.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)

    target=summary[(summary.reservoirs==3)&(summary.stages==18)]
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for name,color in [('zero','#626f7c'),('pwl','#249083')]:
        sub=target[(target.generator==name)&(target.n_calibration==max(BUDGETS))].sort_values('checkpoint')
        axes[0].plot(sub.checkpoint,sub.certificate,'o-',color=color,label=name.upper())
    ref=sub
    axes[0].fill_between(ref.checkpoint,ref.loss_lower,ref.loss_upper,color='#bb554c',alpha=.3,label='Reference loss interval')
    axes[0].set_xticks([40,200,400]);axes[0].set_xlabel('Training update');axes[0].set_ylabel('Policy-loss upper bound')
    axes[0].set_title('Same continuation; different fixed policies');axes[0].legend(frameon=False,fontsize=8)
    sub=target[(target.generator=='pwl')&(target.checkpoint==400)].sort_values('n_calibration')
    x=np.arange(len(sub));bottom=np.zeros(len(sub))
    for key,label,color in [('raw_gap','Paired gap','#249083'),('calibration_correction','Calibration','#e2aa51'),('evaluation_correction','Evaluation','#7987ac')]:
        axes[1].bar(x,sub[key],bottom=bottom,color=color,label=label,width=.6);bottom+=sub[key].to_numpy()
    axes[1].axhspan(float(sub.loss_lower.iloc[0]),float(sub.loss_upper.iloc[0]),color='#bb554c',alpha=.25)
    axes[1].set_xticks(x,labels=['2,048','8,192','32,768']);axes[1].set_xlabel('Calibration paths; evaluation fixed at 8,192')
    axes[1].set_ylabel('PWL certificate components');axes[1].set_title('Where the conservatism comes from');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Three coupled reservoirs, 18 stages — engineering pilot',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('pdf','png'):fig.savefig(OUT/f'three_reservoirs_18_stages.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)

    targetrow=table[(table.reservoirs==3)&(table.stages==18)].iloc[0]
    lines=['# 水库数量与阶段数量：规模预实验','',
        '这是完整的 3×3 规模扫描。各设置使用同一模型族和数据预算；每个设置只有 3 次校准/评估重复，不能作为正式覆盖率实验。',
        '新模型包含容量上限、异质水库和共享供电需求。全部决策保持连续，内层采用可证明精确的整数顶点 DP。', '',
        '下表固定最后一个训练检查点（400 次更新），Ncal=32,768、Nev=8,192，数值为三次重复的平均。', '',
        '| 水库 | 阶段 | 策略成本 J | 真实损失的参考区间 | PWL 证书 U | Zero 证书 U | PWL U/J | Zero U/J |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.reservoirs} | {row.stages} | {row.policy_cost:.3f} | [{row.loss_lower:.3f}, {row.loss_upper:.3f}] | {row.pwl:.3f} | {row.zero:.3f} | {row.pwl_percent:.2f}% | {row.zero_percent:.2f}% |')
    lines+=['','## 3 库、18 阶段','',
        f'PWL 的 U={targetrow.pwl:.4f}，分解为原始差值 {targetrow.raw_gap:.4f} + 校准修正 {targetrow.calibration:.4f} + 评估修正 {targetrow.evaluation:.4f}。',
        f'策略期望成本约 {targetrow.policy_cost:.4f}，真实损失位于 [{targetrow.loss_lower:.4f}, {targetrow.loss_upper:.4f}]。该参考是区间，不能把端点或中点当成精确真实损失。',
        f'精确 DP 对 8,192 条评估路径的一次批量内层计算平均约 {targetrow.inner_seconds:.4f} 秒（8 CPU 线程，共享路径后缀）。这不包含训练、预处理、独立 MIP 审计或参考值计算。',
        '', '## 如何解释','',
        '- 3 库、18 阶段在这类有界、共享二元冲击、可分 PWL 延续值的模型上可以计算；不能外推到 Figure 3 的非凸水文模型。',
        '- 使用相同校准/评估预算比较规模。绝对上界随问题总成本变化，必须同时看 U/J 和超额宽度。',
        '- PWL 改善原始配对差值不代表有限样本证书必然优于 Zero；校准范围和一般包络可能抵消改进。',
        '- 仅有三个重复，表中及图中的平均用于工程判断，不能称为 95% 经验覆盖率验证。',
        '- 同一延续值用于三个检查点，完整结果保留。后期训练不保证每个检查点都严格优于前一个。',
        '- 现有主论文和 Overleaf 包尚未替换为这组预实验。', '',
        '## 核对','',
        f'- 完整结果：{len(f)} 个证书，所有 9 个规模设置齐全。',
        f'- {len(audit)} 次规模相关的独立连续求解核对，最大差异 {max(a["absolute_error"] for a in audit):.3g}。',
        f'- 所有证书覆盖参考损失区间上端：{bool(f.covered_reference_upper.all())}。这是本轮预实验事实，不是总体覆盖概率的估计。',
        f'- 最大策略/数值区间宽度：{f.max_arithmetic_interval.max():.3g}，预设上限 1e-6。',
        '- 原始轨迹、内层整数目标、固定策略、PWL 系数、种子、源码哈希、独立审计、参考最优值区间均保留。', '',
        '复现方式和证明见 ../../SCALING_PROTOCOL.md。图：scale_comparison.png、three_reservoirs_18_stages.png。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(table.round(5).to_string(index=False))
    print('Rows',len(f),'audits',len(audit),'max error',max(a['absolute_error'] for a in audit))


if __name__=='__main__':main()
