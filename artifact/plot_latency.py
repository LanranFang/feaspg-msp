"""Rebuild preprint Figure 7 from the regularized comparison records."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def draw_latency(output, data):
    # Exact two-panel drawing block from regularized_a1_v2_report.update_figures.
    from . import plot_controlled_budget_latency as plot
    plot.configure_matplotlib()
    methods = plot.METHODS
    lo = (data.test_cost_mean_over_train_seeds-data.test_cost_train_seed_sd).min()/1000
    hi = (data.test_cost_mean_over_train_seeds+data.test_cost_train_seed_sd).max()/1000
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), gridspec_kw={'width_ratios': (1.18, 1)})
    plot.draw_budget_axis(axes[0], data[data.policy.isin(methods)], title='(a) Sample-path budget')
    axes[0].set_ylim(min(47, lo-3), max(148, hi+3))
    primary = data[data.n_fit.eq(2500) & data.policy.isin(methods)]
    for method in methods:
        group = primary[primary.policy.eq(method)]
        style = plot.STYLES[method]
        axes[1].errorbar(group.batch1_action_latency_ms_median_across_seeds,
                        group.test_cost_mean_over_train_seeds/1000,
                        yerr=group.test_cost_train_seed_sd/1000, marker=style['marker'],
                        color=style['color'], linestyle='none', capsize=2)
    axes[1].set_title('(b) Cost and recorded CPU latency')
    axes[1].set_xlabel('Batch-1 proposal + projection latency (ms)')
    axes[1].set_ylabel(r'Test cost at $N_{\rm fit}=2{,}500$ ($\times10^3$)')
    axes[1].grid(alpha=.2)
    fig.legend(handles=plot.legend_handles(), loc='lower center', ncol=5, frameon=False)
    fig.subplots_adjust(left=.085, right=.99, top=.88, bottom=.31, wspace=.33)
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'controlled_budget_latency.{ext}', dpi=400, bbox_inches='tight')
    plt.close(fig)

