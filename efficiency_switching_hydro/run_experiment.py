"""Run one context-free efficiency-switching hydro profile."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    DataConfig,
    ExperimentConfig,
    PROFILES,
    RL_ALGORITHMS,
    RLConfig,
    SDDPConfig,
    TrainingConfig,
)
from .data import DataSplits, generate_splits
from .evaluation import evaluate_with_costs, flatten
from .plotting import policy_comparison, training_curves
from .policies import count_parameters, make_policy
from .train import fit_checkpoint


DEFAULT_POLICIES = (
    "myopic",
    "projected_affine",
    "projected_polynomial",
    "projected_graph_neural",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-reservoirs", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--shock-factors", type=int, default=4)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument(
        "--rl-algorithms", nargs="*", choices=RL_ALGORITHMS, default=[]
    )
    parser.add_argument("--skip-sddp", action="store_true")
    parser.add_argument(
        "--reuse-direct-checkpoints",
        action="store_true",
        help="Load compatible direct-policy checkpoints instead of fitting again.",
    )
    parser.add_argument(
        "--direct-checkpoint-dir",
        help="Load direct-policy checkpoints from another experiment directory.",
    )
    parser.add_argument("--reuse-rl-checkpoints", action="store_true")
    parser.add_argument("--n-fit", type=int, default=1000)
    parser.add_argument(
        "--fit-pool-size",
        type=int,
        help="Generate this many fit paths, then train on the first n-fit paths.",
    )
    parser.add_argument("--n-validation", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--n-evaluation", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--lr-affine", type=float, default=5e-4)
    parser.add_argument("--lr-polynomial", type=float, default=1.5e-4)
    parser.add_argument("--lr-neural", type=float, default=3e-4)
    parser.add_argument("--lr-mlp", type=float, default=3e-4)
    parser.add_argument("--rl-reward-scale", type=float, default=1000.0)
    parser.add_argument("--rl-gamma", type=float, default=1.0)
    parser.add_argument("--rl-validation-every", type=int, default=50)
    parser.add_argument("--rl-network-width", type=int, default=256)
    parser.add_argument("--rl-network-depth", type=int, default=2)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-lr-candidates", nargs="+", type=float)
    parser.add_argument("--ppo-rollout-episodes", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--sac-lr", type=float, default=3e-4)
    parser.add_argument("--sac-lr-candidates", nargs="+", type=float)
    parser.add_argument("--td3-lr", type=float, default=1e-3)
    parser.add_argument("--td3-lr-candidates", nargs="+", type=float)
    parser.add_argument("--rl-replay-buffer-size", type=int, default=100000)
    parser.add_argument("--rl-learning-starts", type=int, default=1000)
    parser.add_argument("--rl-batch-size", type=int, default=256)
    parser.add_argument("--rl-gradient-steps", type=int, default=1)
    parser.add_argument("--rl-n-steps", type=int, default=3)
    parser.add_argument("--td3-action-noise", type=float, default=0.10)
    parser.add_argument("--rl-device", default="auto")
    parser.add_argument(
        "--ppo-device",
        help="Override rl-device for PPO (CPU is usually faster for an MLP actor).",
    )
    parser.add_argument(
        "--off-policy-device",
        help="Override rl-device for SAC and TD3.",
    )
    parser.add_argument("--sddp-iterations", type=int, default=50)
    parser.add_argument("--sddp-validation-every", type=int, default=5)
    parser.add_argument("--sddp-noise-scenarios", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate_blocks(
    policy,
    splits: DataSplits,
    config: ExperimentConfig,
    device: str,
    run_id: str,
    test_costs: dict[str, np.ndarray],
) -> dict:
    row: dict = {}
    for name, dataset in (
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        start = time.perf_counter()
        metrics, costs = evaluate_with_costs(policy, dataset, config, device)
        elapsed = time.perf_counter() - start
        row.update(flatten(name, metrics))
        row[f"{name}_wall_time_seconds"] = elapsed
        row[f"{name}_seconds_per_path_stage"] = elapsed / max(
            dataset.n * config.T, 1
        )
        if name == "test":
            test_costs[run_id] = costs
    return row


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    def first_numeric(group: pd.DataFrame, column: str) -> float:
        if column not in group:
            return float("nan")
        values = group[column].dropna()
        return float(values.iloc[0]) if not values.empty else float("nan")

    rows = []
    for policy, group in per_run.groupby("policy", sort=False):
        rows.append(
            {
                "profile": group["profile"].iloc[0],
                "policy": policy,
                "role": group["role"].iloc[0],
                "n_runs": len(group),
                "num_parameters": int(group["num_parameters"].iloc[0]),
                "training_num_parameters": float(
                    group["training_num_parameters"].iloc[0]
                ),
                "number_of_cuts_mean": float(group["number_of_cuts"].mean()),
                "selected_iteration_mean": float(group["selected_iteration"].mean()),
                "fit_trajectories_mean": float(group["fit_trajectories"].mean()),
                "fit_environment_steps_mean": float(
                    group["fit_environment_steps"].mean()
                ),
                "gradient_updates_mean": float(group["gradient_updates"].mean()),
                "selected_learning_rate": first_numeric(
                    group, "selected_learning_rate"
                ),
                "n_hyperparameter_candidates": first_numeric(
                    group, "n_hyperparameter_candidates"
                ),
                "tuning_environment_steps_mean": (
                    float(group["tuning_environment_steps"].mean())
                    if "tuning_environment_steps" in group
                    else float("nan")
                ),
                "tuning_gradient_updates_mean": (
                    float(group["tuning_gradient_updates"].mean())
                    if "tuning_gradient_updates" in group
                    else float("nan")
                ),
                "fit_wall_time_seconds_mean": float(group["fit_wall_time_seconds"].mean()),
                "test_seconds_per_path_stage_mean": float(
                    group["test_seconds_per_path_stage"].mean()
                ),
                "validation_mean_cost": float(group["validation_mean_cost"].mean()),
                "test_mean_cost": float(group["test_mean_cost"].mean()),
                "test_std_across_runs": (
                    float(group["test_mean_cost"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "test_mean_thermal_generation": float(
                    group["test_mean_thermal_generation"].mean()
                ),
                "test_mean_peaker_generation": float(
                    group["test_mean_peaker_generation"].mean()
                ),
                "test_mean_terminal_cost": float(group["test_mean_terminal_cost"].mean()),
                "test_mean_evaporation": float(group["test_mean_evaporation"].mean()),
                "test_mean_ramp_cost": float(group["test_mean_ramp_cost"].mean()),
                "test_mean_storage_penalty": float(
                    group["test_mean_storage_penalty"].mean()
                ),
                "max_action_violation": float(
                    group[["validation_max_action_violation", "test_max_action_violation"]]
                    .max()
                    .max()
                ),
                "max_storage_violation": float(
                    group[["validation_max_storage_violation", "test_max_storage_violation"]]
                    .max()
                    .max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("test_mean_cost")


def paired_comparisons(test_costs: dict[str, np.ndarray]) -> pd.DataFrame:
    candidates = sorted(
        key
        for key in test_costs
        if key.startswith("projected_graph_neural_global")
        or key.startswith("projected_mlp_pathwise")
    )
    rows = []
    for candidate in candidates:
        for baseline, baseline_cost in sorted(test_costs.items()):
            if baseline == candidate:
                continue
            difference = test_costs[candidate] - baseline_cost
            se = float(difference.std(ddof=1) / max(len(difference), 1) ** 0.5)
            mean = float(difference.mean())
            rows.append(
                {
                    "candidate_run": candidate,
                    "baseline_run": baseline,
                    "n_test": len(difference),
                    "mean_candidate_minus_baseline": mean,
                    "paired_se": se,
                    "ci95_lower": mean - 1.96 * se,
                    "ci95_upper": mean + 1.96 * se,
                    "candidate_win": int(mean + 1.96 * se < 0.0),
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    args = parse_args()
    config = ExperimentConfig.make(
        args.profile, args.n_reservoirs, args.horizon, args.shock_factors
    )
    fit_pool_size = args.fit_pool_size or args.n_fit
    if fit_pool_size < args.n_fit:
        raise ValueError("fit-pool-size must be at least n-fit.")
    data_config = DataConfig(
        n_fit=args.n_fit,
        n_validation=args.n_validation,
        n_test=args.n_test,
        n_calibration=args.n_calibration,
        n_evaluation=args.n_evaluation,
    )
    training = TrainingConfig(
        iterations=args.iterations,
        batch_size=args.batch_size,
        validation_every=args.validation_every,
        train_seeds=tuple(args.train_seeds),
        lr_affine=args.lr_affine,
        lr_polynomial=args.lr_polynomial,
        lr_neural=args.lr_neural,
        lr_mlp=args.lr_mlp,
        device=args.device,
    )
    rl_config = RLConfig(
        reward_scale=args.rl_reward_scale,
        gamma=args.rl_gamma,
        validation_every_episodes=args.rl_validation_every,
        network_width=args.rl_network_width,
        network_depth=args.rl_network_depth,
        ppo_learning_rate=args.ppo_lr,
        ppo_rollout_episodes=args.ppo_rollout_episodes,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        sac_learning_rate=args.sac_lr,
        td3_learning_rate=args.td3_lr,
        replay_buffer_size=args.rl_replay_buffer_size,
        learning_starts=args.rl_learning_starts,
        off_policy_batch_size=args.rl_batch_size,
        gradient_steps=args.rl_gradient_steps,
        off_policy_n_steps=args.rl_n_steps,
        td3_action_noise=args.td3_action_noise,
        device=args.rl_device,
    )
    sddp_config = SDDPConfig(
        iterations=args.sddp_iterations,
        validation_every=args.sddp_validation_every,
        noise_scenarios=args.sddp_noise_scenarios,
    )
    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    checkpoints.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": asdict(config),
                "data": asdict(data_config),
                "fit_pool_size": fit_pool_size,
                "training": asdict(training),
                "rl": asdict(rl_config),
                "sddp": asdict(sddp_config),
                "policies": args.policies,
                "rl_algorithms": args.rl_algorithms,
                "rl_learning_rate_candidates": {
                    "ppo": args.ppo_lr_candidates or [args.ppo_lr],
                    "sac": args.sac_lr_candidates or [args.sac_lr],
                    "td3": args.td3_lr_candidates or [args.td3_lr],
                },
                "rl_algorithm_devices": {
                    "ppo": args.ppo_device or args.rl_device,
                    "sac": args.off_policy_device or args.rl_device,
                    "td3": args.off_policy_device or args.rl_device,
                },
                "run_sddp": not args.skip_sddp,
                "reuse_direct_checkpoints": args.reuse_direct_checkpoints,
                "direct_checkpoint_dir": args.direct_checkpoint_dir,
                "reuse_rl_checkpoints": args.reuse_rl_checkpoints,
                "information_pattern": (
                    "No context. Storage, previous release, transit queue, and stage are observed; "
                    "current inflow and demand shocks are observed after release."
                ),
                "reserved_data_note": (
                    "Calibration and evaluation use independent seeds and are generated but never "
                    "read during fitting, selection, or this stage-one report."
                ),
            },
            handle,
            indent=2,
        )
    generation_data_config = replace(data_config, n_fit=fit_pool_size)
    splits = generate_splits(generation_data_config, config)
    if fit_pool_size != args.n_fit:
        splits = DataSplits(
            fit=splits.fit.take(np.arange(args.n_fit), name="fit"),
            validation=splits.validation,
            test=splits.test,
            calibration=splits.calibration,
            evaluation=splits.evaluation,
        )
    recovered_fit_times: dict[tuple[str, str], float] = {}
    partial_results_path = output / "partial_per_run_results.csv"
    if args.reuse_direct_checkpoints and partial_results_path.exists():
        previous_rows = pd.read_csv(partial_results_path)
        for _, previous_row in previous_rows.iterrows():
            recovered_fit_times[
                (str(previous_row["policy"]), str(previous_row["train_seed"]))
            ] = float(previous_row["fit_wall_time_seconds"])
    rows: list[dict] = []
    histories: list[dict] = []
    test_costs: dict[str, np.ndarray] = {}
    rl_hyperparameters: list[dict] = []
    direct_checkpoint_dir = (
        Path(args.direct_checkpoint_dir)
        if args.direct_checkpoint_dir is not None
        else None
    )
    if (
        direct_checkpoint_dir is not None
        and (direct_checkpoint_dir / "checkpoints").is_dir()
    ):
        direct_checkpoint_dir = direct_checkpoint_dir / "checkpoints"
    direct_learning_rates = {
        "projected_affine": training.lr_affine,
        "projected_polynomial": training.lr_polynomial,
        "projected_graph_neural": training.lr_neural,
        "projected_graph_neural_global": training.lr_neural,
        "projected_mlp_pathwise": training.lr_mlp,
    }

    def persist_progress() -> None:
        pd.DataFrame(rows).to_csv(
            output / "partial_per_run_results.csv", index=False
        )
        pd.DataFrame(histories).to_csv(
            output / "partial_training_curves.csv", index=False
        )
        if test_costs:
            pd.DataFrame(
                {"trajectory": np.arange(splits.test.n), **test_costs}
            ).to_csv(output / "partial_test_path_costs.csv", index=False)
        if rl_hyperparameters:
            pd.DataFrame(rl_hyperparameters).to_csv(
                output / "partial_rl_hyperparameter_selection.csv", index=False
            )

    for requested in args.policies:
        canonical = make_policy(requested, config).policy_name
        if canonical == "myopic":
            policy = make_policy(canonical, config).to(training.device)
            row = {
                "profile": config.profile,
                "policy": canonical,
                "role": "deployable_policy",
                "train_seed": "fixed",
                "selected_iteration": 0,
                "best_validation_cost": float("nan"),
                "checkpoint_path": "",
                "num_parameters": count_parameters(policy),
                "training_num_parameters": count_parameters(policy),
                "number_of_cuts": 0,
                "completed_iterations": 0,
                "fit_function_evaluations": 0,
                "fit_trajectories": 0,
                "fit_environment_steps": 0,
                "gradient_updates": 0,
                "training_method": "fixed_rule",
                "fit_wall_time_seconds": 0.0,
            }
            row.update(
                evaluate_blocks(
                    policy,
                    splits,
                    config,
                    training.device,
                    "myopic_fixed",
                    test_costs,
                )
            )
            row["best_validation_cost"] = row["validation_mean_cost"]
            row["reused_checkpoint"] = 0
            rows.append(row)
            persist_progress()
            continue
        for seed in training.train_seeds:
            torch.manual_seed(seed)
            policy = make_policy(canonical, config).to(training.device)
            checkpoint_path = checkpoints / f"{canonical}_seed{seed}.pt"
            load_path = (
                direct_checkpoint_dir / f"{canonical}_seed{seed}.pt"
                if direct_checkpoint_dir is not None
                else checkpoint_path
            )
            if direct_checkpoint_dir is not None and not load_path.exists():
                raise FileNotFoundError(
                    f"Missing requested direct-policy checkpoint: {load_path}"
                )
            reused = (
                args.reuse_direct_checkpoints or direct_checkpoint_dir is not None
            ) and load_path.exists()
            if reused:
                checkpoint_stat = load_path.stat()
                saved = torch.load(
                    load_path,
                    map_location=training.device,
                    weights_only=False,
                )
                if saved["policy_name"] != canonical or saved["profile"] != config.profile:
                    raise ValueError(
                        f"Checkpoint {load_path} is incompatible with this run."
                    )
                policy.load_state_dict(saved["state_dict"])
                policy.eval()
                selected_iteration = int(saved["selected_iteration"])
                best_validation_cost = float(saved["best_validation_cost"])
                completed_iterations = int(saved["completed_iterations"])
                elapsed = recovered_fit_times.get(
                    (canonical, str(seed)),
                    max(0.0, checkpoint_stat.st_mtime - checkpoint_stat.st_atime),
                )
                reported_checkpoint_path = load_path
                histories.append(
                    {
                        "policy": canonical,
                        "train_seed": seed,
                        "iteration": selected_iteration,
                        "fit_trajectories_seen": selected_iteration
                        * training.batch_size,
                        "environment_steps": selected_iteration
                        * training.batch_size
                        * config.T,
                        "train_cost": float("nan"),
                        "validation_cost": best_validation_cost,
                        "is_best": 1,
                        "history_recovered_from_checkpoint": 1,
                    }
                )
            else:
                start = time.perf_counter()
                result = fit_checkpoint(
                    policy,
                    splits.fit,
                    splits.validation,
                    config,
                    training,
                    seed,
                    checkpoint_path,
                )
                elapsed = time.perf_counter() - start
                histories.extend(result.history)
                selected_iteration = result.selected_iteration
                best_validation_cost = result.best_validation_cost
                completed_iterations = result.completed_iterations
                reported_checkpoint_path = checkpoint_path
            row = {
                "profile": config.profile,
                "policy": canonical,
                "role": "deployable_policy",
                "train_seed": seed,
                "selected_iteration": selected_iteration,
                "best_validation_cost": best_validation_cost,
                "checkpoint_path": str(reported_checkpoint_path),
                "num_parameters": count_parameters(policy),
                "training_num_parameters": count_parameters(policy),
                "number_of_cuts": 0,
                "completed_iterations": completed_iterations,
                "fit_function_evaluations": completed_iterations * training.batch_size,
                "fit_trajectories": completed_iterations * training.batch_size,
                "fit_environment_steps": (
                    completed_iterations * training.batch_size * config.T
                ),
                "gradient_updates": completed_iterations,
                "selected_learning_rate": direct_learning_rates[canonical],
                "n_hyperparameter_candidates": 1,
                "tuning_fit_trajectory_uses": completed_iterations
                * training.batch_size,
                "tuning_environment_steps": (
                    completed_iterations * training.batch_size * config.T
                ),
                "tuning_gradient_updates": completed_iterations,
                "training_method": "pathwise_adam",
                "fit_wall_time_seconds": elapsed,
                "reused_checkpoint": int(reused),
            }
            row.update(
                evaluate_blocks(
                    policy,
                    splits,
                    config,
                    training.device,
                    f"{canonical}_seed{seed}",
                    test_costs,
                )
            )
            rows.append(row)
            persist_progress()

    if args.rl_algorithms:
        from .rl import canonical_algorithm
        from .rl import fit_checkpoint as fit_rl
        from .rl import load_checkpoint as load_rl

        learning_rate_candidates = {
            "ppo": args.ppo_lr_candidates or [rl_config.ppo_learning_rate],
            "sac": args.sac_lr_candidates or [rl_config.sac_learning_rate],
            "td3": args.td3_lr_candidates or [rl_config.td3_learning_rate],
        }
        learning_rate_fields = {
            "ppo": "ppo_learning_rate",
            "sac": "sac_learning_rate",
            "td3": "td3_learning_rate",
        }
        algorithm_devices = {
            "ppo": args.ppo_device or rl_config.device,
            "sac": args.off_policy_device or rl_config.device,
            "td3": args.off_policy_device or rl_config.device,
        }
        for requested in args.rl_algorithms:
            algorithm = canonical_algorithm(requested)
            canonical = f"rl_{algorithm}"
            candidate_rates = learning_rate_candidates[algorithm]
            for seed in training.train_seeds:
                candidates: list[dict] = []
                for candidate_index, learning_rate in enumerate(candidate_rates):
                    rate_tag = f"{learning_rate:.8g}".replace(".", "p")
                    checkpoint_name = (
                        f"{canonical}_seed{seed}.zip"
                        if len(candidate_rates) == 1
                        else f"{canonical}_lr{rate_tag}_seed{seed}.zip"
                    )
                    checkpoint_path = checkpoints / checkpoint_name
                    reused_candidate = (
                        args.reuse_rl_checkpoints and checkpoint_path.exists()
                    )
                    candidate_config = replace(
                        rl_config,
                        device=algorithm_devices[algorithm],
                        **{learning_rate_fields[algorithm]: learning_rate},
                    )
                    start = time.perf_counter()
                    if reused_candidate:
                        candidate_policy, metadata = load_rl(
                            algorithm,
                            checkpoint_path,
                            config,
                            candidate_config.device,
                        )
                        candidate_history = [
                            {
                                "policy": canonical,
                                "train_seed": seed,
                                "iteration": int(metadata["selected_iteration"]),
                                "fit_trajectories_seen": int(
                                    metadata["selected_iteration"]
                                ),
                                "environment_steps": int(
                                    metadata["selected_environment_step"]
                                ),
                                "train_cost": float("nan"),
                                "validation_cost": float(
                                    metadata["best_validation_cost"]
                                ),
                                "is_best": 1,
                                "history_recovered_from_checkpoint": 1,
                            }
                        ]
                        candidate = {
                            "policy": candidate_policy,
                            "selected_iteration": int(
                                metadata["selected_iteration"]
                            ),
                            "best_validation_cost": float(
                                metadata["best_validation_cost"]
                            ),
                            "completed_episodes": int(
                                metadata["completed_episodes"]
                            ),
                            "environment_steps": int(metadata["environment_steps"]),
                            "gradient_updates": int(metadata["gradient_updates"]),
                            "deployment_parameters": int(
                                metadata["deployment_parameters"]
                            ),
                            "training_parameters": int(
                                metadata["training_parameters"]
                            ),
                            "history": candidate_history,
                        }
                    else:
                        result = fit_rl(
                            algorithm,
                            splits.fit,
                            splits.validation,
                            config,
                            candidate_config,
                            seed,
                            checkpoint_path,
                            evaluation_device=training.device,
                        )
                        candidate = {
                            "policy": result.policy,
                            "selected_iteration": result.selected_iteration,
                            "best_validation_cost": result.best_validation_cost,
                            "completed_episodes": result.completed_episodes,
                            "environment_steps": result.environment_steps,
                            "gradient_updates": result.gradient_updates,
                            "deployment_parameters": result.deployment_parameters,
                            "training_parameters": result.training_parameters,
                            "history": result.history,
                        }
                    candidate["fit_wall_time_seconds"] = time.perf_counter() - start
                    candidate["checkpoint_path"] = checkpoint_path
                    candidate["learning_rate"] = float(learning_rate)
                    candidate["candidate_index"] = candidate_index
                    candidate["reused"] = reused_candidate
                    candidates.append(candidate)

                selected_index = min(
                    range(len(candidates)),
                    key=lambda index: candidates[index]["best_validation_cost"],
                )
                selected = candidates[selected_index]
                for candidate_index, candidate in enumerate(candidates):
                    candidate_label = (
                        canonical
                        if len(candidates) == 1
                        else f"{canonical}_lr{candidate['learning_rate']:.8g}"
                    )
                    for entry in candidate["history"]:
                        history_entry = dict(entry)
                        history_entry["policy"] = candidate_label
                        history_entry["base_policy"] = canonical
                        history_entry["candidate_learning_rate"] = candidate[
                            "learning_rate"
                        ]
                        history_entry["candidate_selected"] = int(
                            candidate_index == selected_index
                        )
                        histories.append(history_entry)
                    rl_hyperparameters.append(
                        {
                            "profile": config.profile,
                            "algorithm": algorithm,
                            "policy": canonical,
                            "train_seed": seed,
                            "candidate_index": candidate_index,
                            "learning_rate": candidate["learning_rate"],
                            "best_validation_cost": candidate[
                                "best_validation_cost"
                            ],
                            "selected": int(candidate_index == selected_index),
                            "selected_iteration": candidate["selected_iteration"],
                            "environment_steps": candidate["environment_steps"],
                            "gradient_updates": candidate["gradient_updates"],
                            "fit_wall_time_seconds": candidate[
                                "fit_wall_time_seconds"
                            ],
                            "checkpoint_path": str(candidate["checkpoint_path"]),
                        }
                    )
                    if candidate_index != selected_index:
                        candidate["policy"].close()

                policy = selected["policy"]
                selected_iteration = selected["selected_iteration"]
                best_validation_cost = selected["best_validation_cost"]
                completed_episodes = selected["completed_episodes"]
                environment_steps = selected["environment_steps"]
                gradient_updates = selected["gradient_updates"]
                deployment_parameters = selected["deployment_parameters"]
                training_parameters = selected["training_parameters"]
                elapsed = sum(
                    candidate["fit_wall_time_seconds"] for candidate in candidates
                )
                row = {
                    "profile": config.profile,
                    "policy": canonical,
                    "role": "deployable_policy",
                    "train_seed": seed,
                    "selected_iteration": selected_iteration,
                    "best_validation_cost": best_validation_cost,
                    "checkpoint_path": str(selected["checkpoint_path"]),
                    "num_parameters": deployment_parameters,
                    "training_num_parameters": training_parameters,
                    "number_of_cuts": 0,
                    "completed_iterations": completed_episodes,
                    "fit_function_evaluations": completed_episodes,
                    "fit_trajectories": completed_episodes,
                    "fit_environment_steps": environment_steps,
                    "gradient_updates": gradient_updates,
                    "selected_learning_rate": selected["learning_rate"],
                    "n_hyperparameter_candidates": len(candidates),
                    "tuning_fit_trajectory_uses": completed_episodes
                    * len(candidates),
                    "tuning_environment_steps": sum(
                        candidate["environment_steps"] for candidate in candidates
                    ),
                    "tuning_gradient_updates": sum(
                        candidate["gradient_updates"] for candidate in candidates
                    ),
                    "training_method": "model_free_actor_critic",
                    "fit_wall_time_seconds": elapsed,
                    "reused_checkpoint": int(selected["reused"]),
                }
                row.update(
                    evaluate_blocks(
                        policy,
                        splits,
                        config,
                        training.device,
                        f"{canonical}_seed{seed}",
                        test_costs,
                    )
                )
                rows.append(row)
                persist_progress()
                policy.close()

    if not args.skip_sddp:
        from .sddp import fit_checkpoint as fit_sddp
        from .sddp import policy_name as sddp_policy_name

        canonical = sddp_policy_name(config)
        for seed in training.train_seeds:
            start = time.perf_counter()
            result = fit_sddp(
                splits.fit,
                splits.validation,
                config,
                sddp_config,
                seed,
                checkpoints / f"{canonical}_seed{seed}.pt",
            )
            elapsed = time.perf_counter() - start
            histories.extend(result.history)
            row = {
                "profile": config.profile,
                "policy": canonical,
                "role": "deployable_policy",
                "train_seed": seed,
                "selected_iteration": result.selected_iteration,
                "best_validation_cost": result.best_validation_cost,
                "checkpoint_path": result.checkpoint_path,
                "num_parameters": 0,
                "training_num_parameters": 0,
                "number_of_cuts": result.number_of_cuts,
                "completed_iterations": sddp_config.iterations,
                "fit_function_evaluations": float("nan"),
                "fit_trajectories": float("nan"),
                "fit_environment_steps": float("nan"),
                "gradient_updates": float("nan"),
                "training_method": "linearized_empirical_sddp",
                "fit_wall_time_seconds": elapsed,
                "reused_checkpoint": 0,
            }
            row.update(
                evaluate_blocks(
                    result.policy,
                    splits,
                    config,
                    training.device,
                    f"{canonical}_seed{seed}",
                    test_costs,
                )
            )
            rows.append(row)
            persist_progress()
            result.policy.close()

    per_run = pd.DataFrame(rows)
    history = pd.DataFrame(histories)
    summary = summarize(per_run)
    paired = paired_comparisons(test_costs)
    per_run.to_csv(output / "per_run_results.csv", index=False)
    history.to_csv(output / "training_curves.csv", index=False)
    summary.to_csv(output / "main_results.csv", index=False)
    pd.DataFrame({"trajectory": np.arange(splits.test.n), **test_costs}).to_csv(
        output / "test_path_costs.csv", index=False
    )
    paired.to_csv(output / "paired_test_comparisons.csv", index=False)
    pd.DataFrame(rl_hyperparameters).to_csv(
        output / "rl_hyperparameter_selection.csv", index=False
    )
    training_curves(history, figures / "training_curves.png")
    policy_comparison(summary, figures / "policy_comparison.png")
    print(f"Wrote {config.profile} outputs to {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
