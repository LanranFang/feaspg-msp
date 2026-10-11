# Experiment map and full commands

All commands below assume the installation described in `README.md`. Run them
from the package root. Full grids are substantially more expensive than the
smoke checks. Use `--workers` to bound A1 parallelism; the scientific protocol
uses one CPU thread per worker. Internal A1 names refer to FeasPG.

## Main FeasPG comparisons and matched gradient ablation

```bash
python -m artifact.inputs
python -m iclr_algorithm1_experiments.regularized_a1_v2 train --group controlled --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 evaluate --group controlled --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 train --group real --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 evaluate --group real --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 train --group channel --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 evaluate --group channel --workers 4
```

Preparation materializes approximately 0.75 GB of local training bundles;
these are regenerated files and are not all included in the archive.
The controlled grid has fitting budgets 100, 250, 500, 1,000 and 2,500.
The real-case A1 grid has 250, 500, 1,000, 2,500 and 5,000.
Every displayed mean uses five training seeds. `channel` evaluates full,
omitted proposal-state, and omitted moving-recourse derivatives. Omission
arms share the strength chosen for the complete-gradient arm and select their
learning rate by validation. This is the matched-MLP experiment behind the
80.872k to 78.423k statement.

The `train` commands write candidate histories and parameters. The `evaluate`
commands freeze validation choices before the held-out evaluations. Results
are written under `iclr_algorithm1_experiments/outputs/regularized_a1_v2/`.

To summarize newly completed selected runs (without reading any reference
checkpoint weights):

```bash
python -m artifact.summarize_training
```

## Controlled baselines and implementation diagnostics

```bash
python -m iclr_algorithm1_experiments.run_suite --experiments data_budget --device cpu --rl-device cpu
python -m iclr_algorithm1_experiments.run_suite --experiments mechanisms --device cpu --rl-device cpu
python -m artifact.controlled_sddp
```

The first command reproduces the earlier common-protocol runs, including
Projected SHAC, PPO, SAC, and the zero-weight A1 controls. Retain its baseline
rows when comparing against current regularized A1. For the original GPU
baseline execution, use `--rl-device cuda` on suitable hardware.
The mechanism suite includes finite differences, gradient channels and
horizon sensitivity. For a small interface check, add `--smoke` and a separate
`--output-dir outputs/controlled_baseline_smoke`.

`artifact.controlled_sddp` is a reconstructed launch driver around the original
SDDP implementation. It uses two joint outcomes per stage, 50 iterations and
validation at initialization and every ten iterations. Planning uses a
first-order surrogate; evaluation uses the nonlinear simulator. Its complete
25-run curve has not been rerun during artifact assembly, so the archived
curve and the reconstructed launch driver are distinguished in
`VALIDATION.md`.

Post-primary learning-rate sensitivity, after the baseline grid completes:

```bash
python -m iclr_algorithm1_experiments.run_post_primary_lr_sensitivity --device cpu
```

## ONS and CAMELS-BR baselines

These commands consume the included prepared models. They do not download
or audit the original raw-data archives.

```bash
python -m real_hydro.cli run --case ons_subsystems --case camels_br_12 --device cpu --methods seasonal_reference projected_shac rl_ppo rl_sac --output-root outputs/real_baselines
python -m real_hydro.cli run --case ons_subsystems --case camels_br_12 --device cpu --methods hydrology_state_sddp --fit-budgets 100 250 500 1000 2500 --max-validation-checks 4 --output-root outputs/real_sddp
```

The SDDP protocol uses 80 iterations and four periodic validation checks.
Use `--device cuda` to match the original GPU baseline training where
available. Do not use `--all`: unrelated development case IDs remain in some
shared library registries, but their data and results are not part of this
submission. A quick baseline check is:

```bash
python -m real_hydro.cli run --case ons_subsystems --case camels_br_12 --smoke --device cpu --methods seasonal_reference --output-root outputs/real_baseline_smoke
```

## Regularization, scaling, and policy classes

Constructed plateau experiment:

```bash
python -m iclr_algorithm1_experiments.projection_plateau prepare
python -m iclr_algorithm1_experiments.projection_plateau train --workers 4
python -m iclr_algorithm1_experiments.projection_plateau_audit
python -m iclr_algorithm1_experiments.projection_plateau evaluate --workers 4
python -m iclr_algorithm1_experiments.projection_plateau_report
```

Earlier nominal regularization ablations require the completed zero-weight
controlled baseline grid above; that grid supplies its original manifest and
control checkpoints:

```bash
python -m iclr_algorithm1_experiments.projection_distance_run --n-fit 1000 --workers 4
python -m iclr_algorithm1_experiments.projection_distance_extend --workers 4
```

Detaching only the projected action in the auxiliary regularization loss
requires completed `channel` training and selection above:

```bash
python -m iclr_algorithm1_experiments.auxiliary_gradient_v1 prepare
python -m iclr_algorithm1_experiments.auxiliary_gradient_v1 fit --workers 4
python -m iclr_algorithm1_experiments.auxiliary_gradient_v1 evaluate
python -m iclr_algorithm1_experiments.auxiliary_gradient_v1_diagnostics
```

Scale and topology transfer:

```bash
python -m iclr_algorithm1_experiments.run_suite --experiments scaling graph_transfer --device cpu --rl-device cpu
python -m iclr_algorithm1_experiments.regularized_a1_v2 train --group scaling --workers 4
python -m iclr_algorithm1_experiments.regularized_a1_v2 evaluate --group scaling --workers 4
```

The regularized runs replace the main scaling A1 rows; graph-transfer records
retain their separate original protocol. Supplementary 11-policy comparison:

```bash
python -m algorithm1_full_comparison.run_comparison --device cpu --rl-device cpu
python -m iclr_algorithm1_experiments.regularized_a1_v2_full train
python -m iclr_algorithm1_experiments.regularized_a1_v2_full evaluate
```

The full-comparison rerun uses eight training workers and six evaluation
workers as in its original driver. Its original baseline rows and positive
A1 replacements are stored separately in `reference_results/`.

## Certification studies

The main coupled-hydro replay is described in `README.md`. The exact inner DP
exploits this small benchmark's piecewise-linear/integral structure; it is not
a scalable global solver for the 12-reservoir nonlinear training benchmark.
The three-reservoir case has 210 storage states and 5,400 state/action vertices.
The oracle is a verified interval for the continuous stochastic optimum, not
a claim that its exact value was computed.

The original scaling driver can rerun the engineering pilot and continuous
solver audits using the included fitting objects:

```bash
python -m synthetic_hydro_certificate.scaling_run --reservoirs 1 2 3 --stages 6 12 18
python -m synthetic_hydro_certificate.tightening_run pilot
```

For independent refitting and recomputing references into a new output
directory, without replacing the frozen fitting objects:

```bash
python -m artifact.refit_certificate --reservoirs 3 --stages 18
```

This driver invokes the original `run_cell` in a separate directory. It can
take substantially longer than replay because it computes reference values
and uses Gurobi for independent continuous optimization audits.

The single-reservoir study has a separately frozen training protocol:

```bash
python -m synthetic_hydro_certificate.run prepare
python -m synthetic_hydro_certificate.run pilot
python -m synthetic_hydro_certificate.run formal
python -m synthetic_hydro_certificate.report
```

Its formal driver uses 200 repetitions, six calibration/evaluation budgets,
and the predeclared budget pairs. Included fitting objects are checked by
their original checksums. Raw formal paths can be regenerated from seeds;
the archive retains per-repetition numerical reports instead of hundreds of
megabytes of path arrays.

## Paper-to-code map

Use [docs/PAPER_MAP.md](docs/PAPER_MAP.md), which follows the supplied
104-page preprint's Figures 1--7 and Tables 1--14. Earlier submission figure
numbers do not apply to this package.

Rebuild the preprint figures without training:

```bash
python -m artifact.reproduce_figures --output outputs/paper
```

The hydrology figure includes archived 5,000-path SDDP extension points. The
baseline commands above invoke the retained research runners; full numerical
recovery of the archived multi-seed curves has not been established during
packaging. The preprint itself notes that its original standalone SDDP runner
was absent from the original artifact. The code supplied here includes solver
modules and launch interfaces, with that validation limitation preserved.
