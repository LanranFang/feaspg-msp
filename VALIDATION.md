# Preprint package validation

Validation date: **2026-10-11**. Python 3.13.0 on Linux, PyTorch
2.11.0+cu130; CPU runs. The historical September 24 artifact audit is retained
separately in `docs/VALIDATION_2026-09-24.md`. It is not a record of checks
performed today.

## Checks performed for this package

| Check | Outcome |
|---|---|
| Mathematical/model test suite | **113 passed** in 12.33 seconds, including licensed small Gurobi checks |
| Training smoke | Two Adam updates each: controlled Graph, ONS MLP, CAMELS-BR DeepSets; finite objectives and gradients |
| Saved-policy replay | Controlled Graph N_fit=1,000, ONS MLP N_fit=2,500, CAMELS-BR DeepSets N_fit=2,500; seed 0; maximum per-path cost difference **0** for all three |
| Independent certificate replay | Repetition 0, all **90** settings; fresh calibration/evaluation paths; maximum certificate difference **7.11e-15** |
| Paper figures and main table | Figures 2--7 and Table 1 rebuilt; certificate means **39.5354024** and **13.2534751** |
| Preprint alignment | FD errors match printed values; all five hydrology budget series include 5,000 paths and match archived reports exactly; 500-update startup histories have five seeds per arm |
| Source fidelity | Training/model/certificate Python files are byte-identical to the inherited portable artifact |
| Relocation | Clean distributed files copied to a temporary directory; inherited PYTHONPATH removed; all six research package imports resolve locally; figure build succeeds; controlled matched MLP N_fit=100 replay has zero per-path discrepancy |
| Documentation | English Markdown only; current internal documentation links resolve |
| Release integrity | Distributed-file SHA-256 manifest and ZIP CRC verified after assembly |

Machine-readable outcomes are in `artifact/validation_summary.json` and
`artifact/preprint_alignment.json`. Check final distributed-file integrity with
`python -m artifact.verify_manifest`; regenerate the release archive with
`python -m artifact.package`.

## Limits of this validation

The complete multi-seed, multi-budget, multi-hyperparameter training grids
were not retrained. Baseline extensions and historical GPU runtimes were not
recomputed. Existing results, selected parameters, input fingerprints, and
representative replays are provided.

The supplied data are prepared ONS/CAMELS-BR models and historical evaluation
arrays. Full upstream downloads and raw-data preprocessing were not repeated;
see `DATA.md`.

Table 1 is rebuilt from saved numerical records. The linked paper contains
all 14 tables and the conceptual Figure 1; these manuscript assets are
excluded from the code distribution.

The inner DP and verified optimum interval apply to the specific small
certificate benchmarks. They do not establish global optimality of policies
in the larger nonlinear control experiments.

## Code-only distribution check

After excluding the manuscript and exported images, the plotting entry point
is checked without the local `paper/` directory. Source checksums and both
release ZIPs are refreshed. Figures generated during verification remain local.
