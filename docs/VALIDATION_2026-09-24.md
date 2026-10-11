# Artifact validation

Validation date: 2026-09-24. Python 3.13 on Linux, CPU execution unless a test
explicitly invokes Gurobi. Dependency versions are recorded in
`requirements.txt` and `requirements-baselines.txt`.

## Checks completed

| Check | Outcome |
|---|---|
| Original mathematical/model tests | 113 passed; repeated after extracting the ZIP into a separate temporary directory (12.46 seconds for the relocated test run) |
| Input export | All array fields of all 24 benchmark bundles match their original frozen fingerprints |
| Real hydrology regeneration | ONS and CAMELS-BR fitting, validation and test arrays regenerated from the fitted models and original seeds; every array matches exactly |
| Small training checks | Two complete-gradient Adam updates on controlled Graph, ONS MLP and CAMELS-BR DeepSets; finite losses and gradients |
| Complete training-candidate replay | Controlled Graph, N_fit=100, seed 0, 50 Adam updates: selected parameter state matches exactly; validation-cost discrepancy zero |
| Saved-policy replay | Controlled Graph N_fit=1000, ONS MLP N_fit=2500, CAMELS-BR DeepSets N_fit=2500, all seed 0: every test-path cost matches exactly |
| Additional relocated replay | Controlled matched MLP N_fit=100, seed 0: every test-path cost matches exactly |
| Coupled-hydro certificates | Repetitions 0 and 99, all 90 settings each; regenerated calibration/evaluation paths and recomputed inner DP/statistics. Maximum certificate discrepancy about 7.1e-15 |
| Main figures and hydrology table | Rebuilt from archived numerical records; certificate means reproduce 39.5354 and 13.2535 |
| Prepared-data CLI | Both ONS and CAMELS-BR Seasonal smoke grids completed after relocation |
| Relocation | Research module imports resolve inside the extracted archive, with the inherited PYTHONPATH removed |
| Dependency closure | No missing absolute or relative internal Python imports |
| Anonymization | No personal home-directory paths, author names, Overleaf project credentials, Git metadata, or raw execution logs included |

The unit tests cover projection and simulator behavior, policy-gradient
mechanisms, training/baseline interfaces, finite-sample arithmetic, small
continuous-optimization comparisons, and paired-correction calculations.
Passing these checks is an implementation check, not a replacement for the
paper's mathematical assumptions or proofs.

The first generation of hydrology pools takes longer than cached smoke runs.
The first certificate run also compiles the C++ engine. Runtime varies with
hardware; the timings above describe the validation environment only.

## What was not rerun during packaging

The entire five-seed, multi-budget, multi-hyperparameter training matrix was
not rerun. Its existing selected checkpoints, path costs, settings and
summaries are included, and representative policies/training computations
were replayed as described above.

The complete 25-run linearized-SDDP controlled curve was not rerun. Its
numerical curve is archived; `artifact/controlled_sddp.py` reconstructs the
launch procedure from the stated protocol around the original solver and
simulator. Independent small solver tests passed, but those do not establish
bit-for-bit recovery of that full curve.

The raw upstream ONS/CAMELS-BR archives are not included, and their download
and full preprocessing were not repeated. Prepared-model fingerprints and
generated paths were checked instead; see `DATA.md`.
