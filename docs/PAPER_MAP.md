# Map to the 104-page preprint

The authority for this package is [`paper/main_preprint.pdf`](../paper/main_preprint.pdf),
*Differentiating Through Moving Recourse: Feasible Policy Optimization and
Finite-Sample Certification for Multistage Stochastic Programs*, by Lanran
Fang, Hyuk Park, and Grani A. Hanasusanto. Its SHA-256 and the corresponding
TeX source SHA-256 are recorded in `paper/provenance.json`.

The internal names `algorithm1`, `A1`, and `regularized_a1_v2` refer to the
method called **FeasPG** in the preprint. They are retained for compatibility
with saved parameters and experiment records.

## Methods

| Paper item | Implementation |
|---|---|
| Algorithm 1: projected policy, complete pathwise gradient | `efficiency_switching_hydro/projection.py`, `simulator.py`, `policies.py`; `real_data_hydro_experiments/real_hydro/projection.py`, `simulator.py`; `iclr_algorithm1_experiments/regularized_a1_v2.py` |
| Proposal-distance objective and decreasing weight | `iclr_algorithm1_experiments/projection_distance.py`, `projection_distance_run.py`, `regularized_a1_v2.py` |
| Proposal-state / moving-recourse derivative omissions | `algorithm1_gradient_study/`; `iclr_algorithm1_experiments/mechanisms.py`, `regularized_a1_v2.py` (`channel`) |
| Detach projected decision only inside the regularizer | `iclr_algorithm1_experiments/auxiliary_gradient_v1.py`, `auxiliary_gradient_v1_diagnostics.py` |
| Algorithm 2: finite-sample calibration and certification | `synthetic_hydro_certificate/certificate.py`, `scaling.py`, `tightening.py`; exact benchmark inner DP in `scaling_inner.cpp` |
| Projected SHAC | `iclr_algorithm1_experiments/shac.py`; `real_hydro/shac.py` |
| PPO, SAC, linearized SDDP | `efficiency_switching_hydro/`; `real_hydro/`; `artifact/controlled_sddp.py` is a reconstructed launch driver |

The proofs and general abstract assumptions in Appendices A--D are manuscript
results. The supplied certificate solvers implement the specific small hydro
benchmarks. Appendix E is a worked mathematical inventory example, with no
reported numerical experiment; the unrelated legacy inventory experiments
from the research directory are excluded.

## Figures

Run `python -m artifact.reproduce_figures` from the repository root. It rebuilds
Figures 2--7 from archived measurements and recomputes the certificate summary
over all 200 repetitions. Figure 1 is a conceptual TikZ diagram; its source is
copied into the output directory, rather than treated as experiment data.

| Figure | Topic | Portable drawing module / data |
|---|---|---|
| 1 | Two-stage policy and moving recourse | `paper/figures/figure01.tex`, extracted from the exact preprint TeX |
| 2 | Cost FD, regularized FD, horizon amplification | `artifact/plot_gradient_mechanism.py`; `reference_results/iclr_algorithm1_experiments/outputs/formal_protocol_v1/01_mechanisms/` and `auxiliary_gradient_v1/diagnostics/finite_difference.csv` |
| 3 | Constructed zero-gradient initialization, first 300 of 500 updates | `artifact/plot_plateau.py`; `projection_plateau_v1/report/all_validation_histories.csv` |
| 4 | Controlled sample-path budget | `artifact/plot_preprint_budget.py`; `reference_results/paper/controlled_budget_only_data.csv` |
| 5 | ONS/CAMELS-BR budgets, including 5,000 paths | `artifact/plot_preprint_hydrology.py`; `reference_results/paper/real_hydrology_preprint_data.csv` |
| 6 | Coupled-reservoir certificates | `synthetic_hydro_certificate/tightening_report.py`; `reference_results/synthetic_hydro_certificate/outputs/tightened_v3/formal/results.csv` |
| 7 | Controlled budget and recorded CPU latency | `artifact/plot_latency.py`; `regularized_a1_v2/report/controlled_all_methods.csv` |

Paths in the last column without a leading `reference_results/` are relative
to `reference_results/iclr_algorithm1_experiments/outputs/`.
The PDFs actually embedded in the supplied preprint are kept unchanged in
`paper/figures/`. Rebuilt plots have the same numerical data; PDF byte identity
is not expected because metadata, fonts, and backend versions can differ.

The 5,000-path SDDP points in Figure 5 are retained from the published CSV and
the archived budget report. The preprint describes them as a subsequent
extension. They are not newly generated baseline runs during packaging.

## Tables and appendices

Verbatim TeX snapshots of all 14 tables are in `paper/tables/table01.tex` through
`table14.tex`. These are the manuscript's printed values, not independently
recomputed estimates. Table 1 also has a numerical CSV-to-Markdown/TeX rebuild
in `artifact/reproduce_figures.py`. The following map identifies the records
and training drivers for the other tables.

| Table / appendix | Experiment and source |
|---|---|
| 1, F/K | ONS/CAMELS-BR, N_fit=2,500; `regularized_a1_v2` (`real`), `real_hydro.cli`; `regularized_a1_v2/report/real_main_tables.csv` |
| 2, H | Complete controlled comparison, N_fit=1,000; `algorithm1_full_comparison.run_comparison`, `regularized_a1_v2_full`; `reference_results/algorithm1_full_comparison/` |
| 3, H | Post-primary RL learning-rate sensitivity; `run_post_primary_lr_sensitivity`; `formal_protocol_v1/02_data_budget/` |
| 4, I | Startup stress test; `projection_plateau`; `projection_plateau_v1/report/summary.csv` and `per_seed.csv` |
| 5--7, I | Nominal regularization controls and diagnostics; `projection_distance_run`, `projection_distance_extend`, `regularized_a1_v2`; `projection_distance_v1/` and `regularized_a1_v2/report/` |
| 8, J | Complete controlled budget grid; `regularized_a1_v2` (`controlled`), `run_suite` (`data_budget`); `regularized_a1_v2/report/controlled_all_methods.csv` |
| 9, J | Seven size/horizon settings; `regularized_a1_v2` (`scaling`); `regularized_a1_v2/report/`, `formal_protocol_v1/03_scaling/` |
| 10, K | Hydrology CVaR95; `regularized_a1_v2/report/real_main_tables.csv` and underlying real baseline summaries |
| 11, K | Paired bootstrap and seed intervals; `regularized_a1_v2/report/real_paired_effects.csv` |
| 12, K | Locked historical evaluation; `regularized_a1_v2/report/` and `reference_results/real_data_hydro_experiments/outputs/formal_core_v1/` |
| 13, L | Single-reservoir certificate; `synthetic_hydro_certificate.run`; `reference_results/synthetic_hydro_certificate/outputs/certificate_v1/` |
| 14, M | Coupled certificate components; `artifact.reproduce_certificates`, `tightening_run`; frozen `tightened_v3/formal/results.csv` |

Appendix G defines the MLP, DeepSets, Graph, and SHAC policy classes. They are
implemented in the policy modules and supplementary comparison package.
See [`REPRODUCING.md`](../REPRODUCING.md) for actual training commands and costs.

## Scientific scope

- Experiment training uses Adam on fixed fitting pools, with validation
  selection. The theoretical fresh-minibatch SGD theorem is a separate result.
- Main FeasPG rows use positive-regularization `regularized_a1_v2` runs.
  Earlier `run_suite` A1 rows supply zero-weight controls and retained baselines.
- Test and historical paths are not used for hyperparameter/checkpoint selection.
- The coupled certificate optimum is bracketed, not claimed to be known
  exactly. Its small exact DP does not scale to the 12-reservoir nonlinear model.
- Full grids, baseline extensions, and raw-data preprocessing were not rerun
  while preparing this package. See `VALIDATION.md` for checks performed now.
