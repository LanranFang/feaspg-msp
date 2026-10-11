# FeasPG: Multistage Stochastic Programming

Code and numerical artifacts for **Differentiating Through Moving Recourse:
Feasible Policy Optimization and Finite-Sample Certification for Multistage
Stochastic Programs**, by **Lanran Fang, Hyuk Park, and Grani A. Hanasusanto**.

This repository is aligned with the supplied **104-page preprint** in
[`paper/main_preprint.pdf`](paper/main_preprint.pdf). It includes projected
policy optimization, moving-recourse gradients, proposal-distance
regularization, synthetic hydro benchmarks, ONS/CAMELS-BR case studies,
and finite-sample policy certificates.

## Quick start

Validated with Python 3.13 on Linux. Run commands from this repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e real_data_hydro_experiments
python -m artifact.reproduce_figures
```

The last command rebuilds **Figures 2--7 and Table 1** from saved numerical
records in `outputs/paper/`. Figure 1 is a conceptual diagram: its TikZ source
is copied to the output directory. Original preprint figure PDFs and verbatim
TeX snapshots of all **14 tables** are retained under `paper/`.
See [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) for the exact paper-to-code map.

Plotting does not train models. To check short training runs and evaluate
saved policies on the original test paths:

```bash
python -m artifact.smoke
python -m artifact.evaluate_checkpoint
python -m artifact.evaluate_checkpoint --bundle ons_subsystems_n2500 --method algorithm1_mlp
python -m artifact.evaluate_checkpoint --bundle camels_br_12_n2500 --method algorithm1_deepsets
```

The smoke command performs two Adam updates for each benchmark, using small
batches. It checks numerical execution; it is not a paper experiment.
Input pools are generated locally and checked against frozen SHA-256
fingerprints. All 255 saved FeasPG policies are indexed in
`reference_checkpoints/index.json`; the evaluator accepts `--seed`, `--mode`,
and `--zero` to select other entries.

## Certificate replay

The coupled-reservoir engine requires `g++` with OpenMP support. It compiles
on first use. No commercial solver is needed to replay the frozen certificate:

```bash
OMP_NUM_THREADS=2 python -m artifact.reproduce_certificates --repetitions 1
```

This generates fresh calibration/evaluation paths for one original repetition,
recomputes all 90 configurations, and compares with the archived reports.
Use `--repetitions 200` for the complete main certificate study.
Refitting and continuous optimization audits have additional solver requirements.

## Repository layout

| Directory | Contents |
|---|---|
| `artifact/` | Portable input preparation, smoke runs, checkpoint/certificate replay, and preprint plotting |
| `efficiency_switching_hydro/` | Controlled nonlinear hydro model, projection, simulator, policies, PPO/SAC, linearized SDDP |
| `iclr_algorithm1_experiments/` | FeasPG training, gradient ablations, regularization, SHAC, budgets, scaling, transfer |
| `algorithm1_gradient_study/` | Finite differences and derivative omission |
| `algorithm1_full_comparison/` | Supplementary policy-class comparison |
| `real_data_hydro_experiments/` | ONS/CAMELS-BR prepared models and training/evaluation modules |
| `synthetic_hydro_certificate/` | Calibration, certificate arithmetic, paired correction, and exact small-benchmark inner DP |
| `inputs/` | Configurations, seeds, input recipes, and fingerprints |
| `reference_results/` | Archived numerical summaries and certificate reports |
| `reference_checkpoints/` | Selected FeasPG parameters and per-path costs |
| `paper/` | Exact supplied PDF, embedded figure PDFs, table/diagram source snapshots, provenance |
| `docs/` | Paper map and historical validation record |

The internal names `A1`, `algorithm1`, and `regularized_a1_v2` refer to FeasPG.
They are kept compatible with the original checkpoint and experiment formats.
Frozen certificate fitting objects remain under
`synthetic_hydro_certificate/outputs/` because the original loading API uses
those paths. Generated outputs and caches are excluded from the upload bundle.

## Full experiments and validation

[`REPRODUCING.md`](REPRODUCING.md) contains the training commands.
Main FeasPG results use the **positive-regularization reruns**, rather than
older zero-weight A1 comparisons. Experiments use Adam on fixed fitting pools
and validation selection; the theoretical fresh-minibatch SGD result is separate.
Figure 5 retains the subsequent 5,000-path hydrology extension, including its
archived SDDP points.

For RL and solver baselines, install the optional dependencies:

```bash
python -m pip install -r requirements-baselines.txt
python -m pytest -q
```

The full test suite includes Gurobi checks and requires a suitable license.
Large solver runs may need a license that supports their problem sizes.
Baseline curves and complete multi-seed training grids were not rerun while
assembling this repository. Saved-record plotting, policy replay, and new
training are documented separately in [`VALIDATION.md`](VALIDATION.md).

[`DATA.md`](DATA.md) describes data preparation and upstream attribution.
[`CHANGES.md`](CHANGES.md) records packaging changes.
Run `python -m artifact.verify_manifest` to check distributed file integrity.
Citation metadata is in [`CITATION.cff`](CITATION.cff).
