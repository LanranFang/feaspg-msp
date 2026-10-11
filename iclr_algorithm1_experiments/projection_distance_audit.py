"""Independent integrity audit of the auxiliary-loss experiment artifacts.

This verifier does not import the runner or auxiliary-loss implementation.
With --replay it also recomputes selected-policy evaluation and a prespecified
sample of initialization gradient norms directly from the unchanged simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate
from .config import ICLRProtocol, SeedBlock
from .policies import build_policy

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "iclr_algorithm1_experiments/outputs/formal_protocol_v1/02_data_budget"


class AuditError(RuntimeError):
    """A frozen-protocol or numerical consistency check failed."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def data_hash(dataset):
    digest = hashlib.sha256()
    for field in ("inflow", "demand", "price"):
        array = np.ascontiguousarray(getattr(dataset, field))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def close(actual, expected, message, rtol=1e-12, atol=1e-12):
    require(bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)), message)


def coefficient(arm, reference, iteration):
    """Iteration is one-based, independently expressing the frozen schedule."""
    return {"zero": 0.0, "tenth": 0.1 * reference, "fixed": reference,
            "decay": reference / math.sqrt(iteration)}[arm]


def verify_norms(initialization, n_fit):
    cost = np.asarray(initialization["cost_gradient_norms"], dtype=np.float64)
    distance = np.asarray(initialization["distance_gradient_norms"], dtype=np.float64)
    require(cost.shape == distance.shape == (n_fit,), "Initialization must use every fitting path separately")
    require(np.isfinite(cost).all() and np.isfinite(distance).all(), "Nonfinite initialization gradient norms")
    require((cost >= 0).all() and (distance >= 0).all(), "Negative initialization gradient norm")
    close(initialization["median_cost_gradient_norm"], np.median(cost), "Incorrect median cost-gradient norm")
    close(initialization["median_distance_gradient_norm"], np.median(distance), "Incorrect median distance-gradient norm")
    close(initialization["lambda_ref"], np.median(cost) / (np.median(distance) + 1e-12),
          "lambda_ref must be the ratio of medians")
    require(initialization["n_paths"] == n_fit, "Incorrect calibration path count")
    require(initialization["gradient_mode"] == "full", "Calibration omitted complete recourse derivative")


def verify_history(result, training, validation, iterations, batch_size, validation_every):
    require(len(training) == iterations, "Wrong number of optimizer updates")
    require(training.iteration.tolist() == list(range(1, iterations + 1)), "Training update indices differ")
    require(training.fit_trajectories_seen.tolist() == [batch_size * k for k in range(1, iterations + 1)],
            "Wrong fitting-trajectory accounting")
    expected_lambda = [coefficient(result["arm"], result["lambda_ref"], k)
                       for k in range(1, iterations + 1)]
    close(training.lambda_value, expected_lambda, "Incorrect or shifted lambda schedule")
    finite_columns = ["train_cost", "train_objective", "gradient_norm_preclip", "mean_cost",
                      "proposal_distance", "active_projection_fraction", "max_action_violation",
                      "max_storage_violation"]
    require(np.isfinite(training[finite_columns].to_numpy()).all(), "Nonfinite training diagnostic")
    close(training.train_cost, training.mean_cost, "Operating cost changed in diagnostics")
    # Values were evaluated in float32. Reconstructing from CSV in float64 can
    # differ by one float32 rounding unit, so account for that explicitly.
    close(training.train_objective,
          training.train_cost + training.lambda_value * training.proposal_distance,
          "Training objective is not C + lambda * half-summed proposal distance",
          rtol=3e-7, atol=1e-5)
    require((training.gradient_norm_preclip >= 0).all(), "Negative gradient norm")
    require(training.gradient_clipped.tolist() == (training.gradient_norm_preclip > result["gradient_clip"]).astype(int).tolist(),
            "Gradient clipping diagnostic disagrees with the original clip threshold")
    require(training.active_projection_fraction.between(0, 1).all(), "Active projection fraction outside [0,1]")
    require((training.proposal_distance >= 0).all(), "Negative proposal distance")
    for key in ("max_action_violation", "max_storage_violation"):
        require(training[key].between(0, 1e-5).all(), f"Training feasibility failure: {key}")
    checks = [0] + [k for k in range(1, iterations + 1) if k % validation_every == 0 or k == iterations]
    require(validation.iteration.tolist() == checks, "Validation checkpoint schedule changed")
    require(np.isfinite(validation.mean_cost).all(), "Nonfinite validation cost")
    best_index = int(np.argmin(validation.mean_cost.to_numpy()))
    require(result["selected_iteration"] == int(validation.iloc[best_index].iteration),
            "Checkpoint selection did not minimize original validation C with first-tie priority")
    close(result["best_validation_cost"], float(validation.iloc[best_index].mean_cost),
          "Selected validation cost does not match history")
    observed = training.loc[training.validation_cost.notna(), ["iteration", "validation_cost"]]
    require(observed.iteration.tolist() == checks[1:], "Training history has missing or extra validation checks")
    close(observed.validation_cost, validation.mean_cost.iloc[1:], "Two validation histories disagree")


def reconstructed_protocol(payload):
    values = dict(payload)
    for item in fields(ICLRProtocol):
        value = values[item.name]
        if item.name.endswith("_seeds") and isinstance(value, dict):
            values[item.name] = SeedBlock(**value)
        elif isinstance(value, list):
            values[item.name] = tuple(value)
    return ICLRProtocol(**values)


def build_initial(protocol, hydro, seed, name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_policy(name, hydro, protocol).to(protocol.device)


def direct_metrics(trace):
    with torch.no_grad():
        delta = trace.proposals - trace.actions
        distance = 0.5 * delta.square().sum(dim=-1).sum(dim=-1)
        return {
            "mean_cost": float(trace.mean_cost),
            "proposal_distance": float(distance.mean()),
            "active_projection_fraction": float((delta.square().sum(dim=-1).sqrt() > 1e-6).double().mean()),
            "max_action_violation": float(trace.max_action_violation),
            "max_storage_violation": float(trace.max_storage_violation),
        }


def audit(output: Path, replay=False):
    torch.set_num_threads(1)
    output = Path(output).resolve()
    frozen = json.loads((output / "PROTOCOL.json").read_text())
    is_extension = frozen["experiment"] == "conditional_projection_distance_budget_extension"
    protocol = reconstructed_protocol(frozen["protocol"])
    n_fit, name, arms = frozen["n_fit"], frozen["policy"], frozen["arms"]
    rates = protocol.graph_learning_rates if name == "projected_graph_neural_global" else protocol.mlp_learning_rates
    require(len(set(arms)) == len(arms) and set(arms) <= {"zero", "tenth", "fixed", "decay"}, "Invalid/duplicate experiment arms")
    for source, expected in frozen["source_sha256"].items():
        require(file_hash(ROOT / source) == expected, f"Frozen source changed: {source}")
    if is_extension:
        parent = Path(frozen["parent"])
        for key, relative in (("parent_protocol_sha256", "PROTOCOL.json"),
                              ("parent_selection_sha256", "SELECTION.json"),
                              ("parent_decision_sha256", "report/DECISION.json"),
                              ("parent_independent_audit_sha256", "INDEPENDENT_AUDIT.json")):
            require(file_hash(parent / relative) == frozen[key], f"Frozen parent artifact changed: {relative}")
        parent_protocol = json.loads((parent / "PROTOCOL.json").read_text())
        parent_selection = json.loads((parent / "SELECTION.json").read_text())
        parent_decision = json.loads((parent / "report/DECISION.json").read_text())
        parent_audit = json.loads((parent / "INDEPENDENT_AUDIT.json").read_text())
        require(parent_audit["status"] == "passed" and parent_audit["n_candidates"] == 60 and parent_audit["n_selected"] == 20,
                "Parent full-grid independent audit did not pass")
        require(parent_audit["original_baselines_bitwise_verified"] == 5 and parent_audit["independently_replayed_selected_evaluations"] == 20,
                "Parent baseline/evaluation replay audit is incomplete")
        require(parent_audit["protocol_sha256"] == frozen["parent_protocol_sha256"], "Parent audit refers to another protocol")
        require(parent_protocol["n_fit"] == 1000 and parent_protocol["protocol"] == frozen["protocol"],
                "Extension changed original parent experiment settings")
        inherited = parent_selection["selected_positive_arm"]
        require(arms == [inherited] and parent_decision["selected_positive_arm"] == inherited,
                "Extension changed the validation-selected lambda version")
        require(parent_decision["extend_displayed_mlp_budgets"] and not parent_decision["smoke"],
                "Parent formal decision did not permit extension")
        require(parent_selection["mean_validation"][inherited] < parent_selection["mean_validation"]["zero"],
                "Parent validation improvement is absent")
        parent_evaluations = json.loads((parent / "evaluations.json").read_text())
        test_means = {arm: np.mean([item["test"]["mean_path_cost_float64"] for item in parent_evaluations if item["arm"] == arm])
                      for arm in ("zero", inherited)}
        require(test_means[inherited] < test_means["zero"], "Parent paired mean test improvement is absent")
        require(n_fit in [n for n in protocol.fit_budgets if n != 1000], "Unexpected extension budget")
        for relative, digest in frozen["original_selected_checkpoint_sha256"].items():
            require(file_hash(ROOT / relative) == digest, "Original baseline checkpoint changed")
        require(file_hash(ORIGINAL / "data_manifest.json") == frozen["original_data_manifest_sha256"], "Original data manifest changed")
        if frozen["original_test_path_costs_sha256"] is not None:
            require(file_hash(ORIGINAL / "test_path_costs.csv") == frozen["original_test_path_costs_sha256"], "Original test path-cost archive changed")
    else:
        require(file_hash(ORIGINAL / "protocol.json") == frozen["original_protocol_sha256"], "Original protocol changed")
    require(file_hash(ORIGINAL / "per_run_results.csv") == frozen["original_results_sha256"], "Original results changed")
    hydro = protocol.hydro_config()
    pool = generate_dataset("fit_pool", max(protocol.fit_budgets), protocol.budget_seeds.fit, hydro)
    fit = pool.take(np.arange(n_fit))
    validation = generate_dataset("validation", protocol.n_validation, protocol.budget_seeds.validation, hydro)
    for key, dataset in (("fit_pool_sha256", pool), ("fit_sha256", fit), ("validation_sha256", validation)):
        require(data_hash(dataset) == frozen[key], f"Frozen data fingerprint mismatch: {key}")
    iterations, batch_size = protocol.iterations(n_fit), protocol.batch_size
    initializations = {}
    replayed_initialization_paths = 0
    for seed in protocol.train_seeds:
        path = output / "initialization" / f"seed{seed}.json"
        info = json.loads(path.read_text())
        verify_norms(info, n_fit)
        initializations[seed] = info
        require(info["fit_sha256"] == frozen["fit_sha256"], "Calibration fitting pool changed")
        require(info["fitting_data_seed"] == protocol.budget_seeds.fit, "Calibration used wrong data seed")
        require(info["seed"] == seed and info["n_fit"] == n_fit and info["policy"] == name, "Initialization identity mismatch")
        expected_batches = np.random.default_rng(seed).permutation(n_fit).reshape(iterations, batch_size)
        saved_batches = np.load(path.with_suffix(".batches.npy"))
        require(np.array_equal(saved_batches, expected_batches), "Batch permutation differs from original single pass")
        require(hashlib.sha256(saved_batches.tobytes()).hexdigest() == info["batches_sha256"], "Batch hash mismatch")
        policy = build_initial(protocol, hydro, seed, name)
        initial_state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
        require(tensor_hash(initial_state) == info["initial_state_sha256"] == tensor_hash(policy.state_dict()),
                "Saved initialization differs from seeded original builder")
        if replay:
            for index in sorted({0, n_fit // 2, n_fit - 1}):
                trace = simulate(policy, fit.take(np.asarray([index])), hydro, device=protocol.device)
                distance_objective = 0.5 * (trace.proposals - trace.actions).square().sum()
                params = tuple(policy.parameters())
                cg = torch.autograd.grad(trace.mean_cost, params, retain_graph=True, allow_unused=True)
                dg = torch.autograd.grad(distance_objective, params, allow_unused=True)
                for column, gradients in (("cost_gradient_norms", cg), ("distance_gradient_norms", dg)):
                    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))
                    close(info[column][index], norm, f"Initialization path-gradient replay mismatch: seed{seed}/{index}/{column}", rtol=1e-10)
                replayed_initialization_paths += 1
    expected_keys = {(seed, arm, rate) for seed in protocol.train_seeds for arm in arms for rate in range(len(rates))}
    candidates = {}
    for path in sorted((output / "candidates").glob("seed*/*/lr*/result.json")):
        result = json.loads(path.read_text())
        key = (result["seed"], result["arm"], result["lr_index"])
        require(key not in candidates and key in expected_keys, "Unexpected or duplicate training candidate")
        candidates[key] = result
        seed, arm, rate = key
        require(path.parent == output / result["path"], "Candidate directory metadata mismatch")
        require(result["n_fit"] == n_fit and result["policy"] == name and result["iterations"] == iterations,
                "Candidate model/budget identity differs from protocol")
        close(result["learning_rate"], rates[rate], "Candidate learning rate differs from protocol")
        close(result["lambda_ref"], initializations[seed]["lambda_ref"], "Candidate changed initial lambda_ref")
        for key_hash in ("initial_state_sha256", "fit_sha256", "batches_sha256"):
            require(result[key_hash] == initializations[seed][key_hash], f"Candidate changed {key_hash}")
        require(result["gradient_clip"] == protocol.gradient_clip, "Gradient clip differs from original")
        training = pd.read_csv(path.parent / "training.csv")
        validations = pd.read_csv(path.parent / "validation.csv")
        verify_history(result, training, validations, iterations, batch_size, protocol.validation_every(n_fit))
        checkpoint = torch.load(path.parent / "selected.pt", map_location="cpu", weights_only=False)
        require(tensor_hash(checkpoint["state_dict"]) == result["selected_state_sha256"], "Selected tensor hash mismatch")
        for field in result:
            require(checkpoint[field] == result[field], f"Selected checkpoint metadata differs: {field}")
        require(math.isfinite(result["fit_time_seconds"]) and 0 <= result["monitoring_seconds"] <= result["fit_time_seconds"],
                "Invalid runtime accounting")
    require(set(candidates) == expected_keys, "Incomplete candidate grid")
    selected = [min((candidates[seed, arm, rate] for rate in range(len(rates))),
                    key=lambda item: (item["best_validation_cost"], item["lr_index"]))
                for seed in protocol.train_seeds for arm in arms]
    selection = json.loads((output / "SELECTION.json").read_text())
    require(selection["selected"] == selected, "Frozen selection did not minimize per-arm validation cost")
    means = {arm: float(np.mean([item["best_validation_cost"] for item in selected if item["arm"] == arm])) for arm in arms}
    require(selection["mean_validation"] == means, "Across-seed validation aggregation differs")
    positive = [arm for arm in arms if arm != "zero"]
    chosen = min(positive, key=lambda arm: means[arm]) if positive else None
    require(selection["selected_positive_arm"] == chosen, "Positive arm selected using non-validation criterion")
    if not is_extension:
        require(selection["validation_improves"] == (chosen is not None and means[chosen] < means.get("zero", float("inf"))),
                "Validation improvement flag is incorrect")
    else:
        require(selection["parent_decision_sha256"] == frozen["parent_decision_sha256"], "Extension selection uses a different parent decision")
    baseline_checks = 0
    if not protocol.smoke:
        for result in selected:
            if result["arm"] != "zero":
                continue
            original_path = ORIGINAL / "checkpoints" / f"n{n_fit}" / f"{name}_selected_seed{result['seed']}.pt"
            old = torch.load(original_path, map_location="cpu", weights_only=False)
            new = torch.load(output / result["path"] / "selected.pt", map_location="cpu", weights_only=False)
            require(old["state_dict"].keys() == new["state_dict"].keys(), "Baseline parameter keys changed")
            require(all(torch.equal(old["state_dict"][key], value) for key, value in new["state_dict"].items()),
                    "Lambda-zero parameters do not reproduce original Figure3 exactly")
            for field in ("learning_rate", "selected_iteration", "best_validation_cost"):
                require(old[field] == new[field], f"Lambda-zero selection differs from original: {field}")
            baseline_checks += 1
    evaluations = json.loads((output / "evaluations.json").read_text())
    require(len(evaluations) == len(selected), "Incomplete selected-model evaluations")
    evaluations_by_path = {item["path"]: item for item in evaluations}
    require(len(evaluations_by_path) == len(evaluations) and set(evaluations_by_path) == {item["path"] for item in selected},
            "Evaluations duplicate or omit selected candidates")
    test = generate_dataset("test", protocol.n_test, protocol.budget_seeds.test, hydro)
    test_fingerprint = data_hash(test)
    if not protocol.smoke:
        require(test_fingerprint == frozen["expected_test_sha256"], "Test data differs from original Figure3")
    for result in selected:
        destination = output / result["path"]
        evaluation = json.loads((destination / "evaluation.json").read_text())
        require(evaluation == evaluations_by_path[result["path"]], "Aggregate and individual evaluation disagree")
        for field in result:
            require(evaluation[field] == result[field], f"Evaluation changed selected model metadata: {field}")
        require(evaluation["test_sha256"] == test_fingerprint, "Evaluation used different test paths")
        costs = np.load(destination / "test_costs.npy")
        require(costs.shape == (protocol.n_test,) and np.isfinite(costs).all(), "Invalid saved test path costs")
        close(evaluation["test"]["mean_path_cost_float64"], np.mean(costs), "Test path mean mismatch")
        close(evaluation["test"]["path_cost_sd"], np.std(costs, ddof=1), "Test path SD mismatch")
        close(evaluation["validation"]["mean_cost"], result["best_validation_cost"], "Selected validation cost replay mismatch", rtol=0, atol=0)
        for split, n_paths in (("validation", protocol.n_validation), ("test", protocol.n_test)):
            metrics = evaluation[split]
            require(metrics["n_paths"] == n_paths, "Wrong evaluated split size")
            require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite evaluation metric")
            require(0 <= metrics["active_projection_fraction"] <= 1, "Invalid active projection fraction")
            require(metrics["proposal_distance"] >= 0, "Negative evaluation proposal distance")
            require(0 <= metrics["max_action_violation"] <= 1e-5 and 0 <= metrics["max_storage_violation"] <= 1e-5,
                    "Selected policy violates feasibility tolerance")
        diagnostic = evaluation["fit_gradient_diagnostic"]
        require(diagnostic["n_paths"] == min(32, n_fit), "Fitting gradient diagnostic used wrong path count")
        for key in ("cost_gradient_norm", "distance_gradient_norm", "total_gradient_norm"):
            require(math.isfinite(diagnostic[key]) and diagnostic[key] >= 0, "Nonfinite/negative selected-policy gradient norm")
        close(diagnostic["lambda_value"], coefficient(result["arm"], result["lambda_ref"], max(1, result["selected_iteration"])),
              "Selected-policy gradient uses wrong lambda schedule position")
        if replay:
            policy = build_initial(protocol, hydro, result["seed"], name)
            policy.load_state_dict(torch.load(destination / "selected.pt", map_location="cpu", weights_only=False)["state_dict"])
            policy.eval()
            with torch.no_grad():
                for split, dataset in (("validation", validation), ("test", test)):
                    trace = simulate(policy, dataset, hydro, device=protocol.device)
                    for key, value in direct_metrics(trace).items():
                        close(evaluation[split][key], value, f"Independent evaluation replay differs: {result['path']}/{split}/{key}", rtol=3e-7, atol=1e-8)
                    if split == "test":
                        require(np.array_equal(trace.total_cost.cpu().numpy().astype(np.float64), costs),
                                "Independent test cost replay is not bitwise equal")
    completed = json.loads((output / "COMPLETED.json").read_text())
    require(completed["n_candidates"] == len(candidates) and completed["n_selected"] == len(selected), "Completion counts incorrect")
    if is_extension:
        require(completed["n_baseline_training_runs"] == 0, "Extension unexpectedly retrained baselines")
    return {"status": "passed", "experiment": frozen["experiment"], "n_fit": n_fit,
            "n_candidates": len(candidates), "n_optimizer_updates": len(candidates) * iterations,
            "n_selected": len(selected), "n_test_costs": len(selected) * protocol.n_test,
            "original_baselines_bitwise_verified": baseline_checks,
            "replayed_initialization_paths": replayed_initialization_paths,
            "independently_replayed_selected_evaluations": len(selected) if replay else 0,
            "auditor_sha256": file_hash(__file__), "protocol_sha256": file_hash(output / "PROTOCOL.json"),
            "limitations": "Integrity checks cannot reconstruct unsaved intermediate parameter states; selected-policy and sampled initialization numerical replays are separately counted."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(args.output, args.replay)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        (args.output / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (args.output / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
