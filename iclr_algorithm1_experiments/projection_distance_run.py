"""Matched-data projection-distance ablation; existing experiments stay immutable.

Fit/validation selection finishes before test data are generated. Each run uses
the original nested pool, initialization, optimizer, batches, and LR candidates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algorithm1_gradient_study.training import validation_cost
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate
from efficiency_switching_hydro.train import unused_batches
from .config import ICLRProtocol
from .io import dataset_fingerprint, write_frame
from .policies import build_policy
from .projection_distance import (
    estimate_lambda_ref, lambda_at_step, projection_distance,
    regularized_objective, trace_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "iclr_algorithm1_experiments/outputs/formal_protocol_v1/02_data_budget"
ARMS = ("zero", "tenth", "fixed", "decay")


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def configure() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def data(protocol, n_fit):
    hydro = protocol.hydro_config()
    # The RNG fills all factors before idiosyncratic noise: generating n_fit
    # directly is NOT equivalent to taking a prefix of the original 2500 pool.
    pool = generate_dataset("fit_pool", max(protocol.fit_budgets), protocol.budget_seeds.fit, hydro)
    fit = pool.take(np.arange(n_fit), name=f"fit_n{n_fit}")
    validation = generate_dataset("validation", protocol.n_validation, protocol.budget_seeds.validation, hydro)
    return hydro, pool, fit, validation


def initial_policy(protocol, hydro, seed, policy_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_policy(policy_name, hydro, protocol).to(protocol.device)


def prepare_seed(args):
    protocol, n_fit, seed, policy_name, output = args
    configure()
    output = Path(output)
    destination = output / "initialization" / f"seed{seed}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    hydro, _, fit, _ = data(protocol, n_fit)
    policy = initial_policy(protocol, hydro, seed, policy_name)
    training = protocol.direct_training(n_fit)
    batches = unused_batches(n_fit, training.iterations, training.batch_size, np.random.default_rng(seed))
    statistics = estimate_lambda_ref(policy, fit, hydro, device=protocol.device)
    statistics.update({
        "seed": seed, "n_fit": n_fit, "policy": policy_name,
        "initial_state_sha256": state_hash(policy.state_dict()),
        "fit_sha256": dataset_fingerprint(fit),
        "batches_sha256": hashlib.sha256(batches.tobytes()).hexdigest(),
        "calibration_role": "fitting data only; frozen before all optimizer updates",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination.with_suffix(".pt"))
    np.save(destination.with_suffix(".batches.npy"), batches)
    save_json(destination, statistics)
    return statistics


def candidate_path(output, seed, arm, lr_index):
    return Path(output) / "candidates" / f"seed{seed}" / arm / f"lr{lr_index}"


@torch.no_grad()
def evaluate_trace(policy, dataset, hydro, device):
    policy.eval()
    trace = simulate(policy, dataset, hydro, device=device)
    metrics = trace_metrics(trace)
    costs = trace.total_cost.detach().cpu().numpy().astype(np.float64)
    metrics["mean_path_cost_float64"] = float(costs.mean())
    metrics["path_cost_sd"] = float(costs.std(ddof=1)) if len(costs) > 1 else 0.0
    metrics["n_paths"] = len(costs)
    return metrics, costs


def gradient_diagnostic(policy, dataset, hydro, device, lambda_value):
    policy.eval()
    trace = simulate(policy, dataset, hydro, device=device)
    parameters = tuple(policy.parameters())
    gc = torch.autograd.grad(trace.mean_cost, parameters, retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(projection_distance(trace).mean(), parameters, allow_unused=True)
    def vector(grads):
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                          for p, g in zip(parameters, grads)]).double()
    c, d = vector(gc), vector(gd)
    return {
        "cost_gradient_norm": float(torch.linalg.vector_norm(c)),
        "distance_gradient_norm": float(torch.linalg.vector_norm(d)),
        "total_gradient_norm": float(torch.linalg.vector_norm(c + lambda_value * d)),
        "lambda_value": float(lambda_value), "n_paths": dataset.n,
        "role": "fixed first 32 fitting paths (or entire smaller smoke pool); no updates",
    }


def fit_one(args):
    protocol, n_fit, seed, policy_name, arm, lr_index, output = args
    configure()
    output = Path(output)
    destination = candidate_path(output, seed, arm, lr_index)
    result_path = destination / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    hydro, _, fit, validation = data(protocol, n_fit)
    policy = initial_policy(protocol, hydro, seed, policy_name)
    initialization = json.loads((output / "initialization" / f"seed{seed}.json").read_text())
    assert state_hash(policy.state_dict()) == initialization["initial_state_sha256"]
    training = protocol.direct_training(n_fit)
    rates = protocol.graph_learning_rates if policy_name == "projected_graph_neural_global" else protocol.mlp_learning_rates
    learning_rate = rates[lr_index]
    lambda_ref = initialization["lambda_ref"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=training.weight_decay)
    batches = unused_batches(n_fit, training.iterations, training.batch_size, np.random.default_rng(seed))
    assert hashlib.sha256(batches.tobytes()).hexdigest() == initialization["batches_sha256"]
    best_state = copy.deepcopy(policy.state_dict())
    best_iteration = 0
    start = time.perf_counter()
    best_validation = validation_cost(policy, validation, hydro, training.device)
    history = []
    validation_history = []
    monitoring_seconds = 0.0
    mt = time.perf_counter()
    initial_metrics, _ = evaluate_trace(policy, validation, hydro, training.device)
    validation_history.append({"iteration": 0, **initial_metrics})
    monitoring_seconds += time.perf_counter() - mt
    for iteration in range(1, training.iterations + 1):
        policy.train()
        batch = fit.take(batches[iteration - 1], name="fit_batch")
        optimizer.zero_grad(set_to_none=True)
        trace = simulate(policy, batch, hydro, device=training.device, gradient_mode="full")
        lam = lambda_at_step(arm, lambda_ref, iteration - 1)
        loss = regularized_objective(trace, lam)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), training.gradient_clip)
        optimizer.step()
        mt = time.perf_counter()
        with torch.no_grad():
            metrics = trace_metrics(trace)
        record = {
            "iteration": iteration, "lambda_value": lam,
            "fit_trajectories_seen": iteration * training.batch_size,
            "train_cost": float(trace.mean_cost.detach()),
            "train_objective": float(loss.detach()),
            "gradient_norm_preclip": float(grad_norm),
            "gradient_clipped": int(float(grad_norm) > training.gradient_clip),
            **metrics,
        }
        monitoring_seconds += time.perf_counter() - mt
        if iteration % training.validation_every == 0 or iteration == training.iterations:
            # Same original float32 mean and strict checkpoint comparison.
            policy.eval()
            with torch.no_grad():
                vtrace = simulate(policy, validation, hydro, device=training.device)
                current_validation = float(vtrace.mean_cost.cpu())
            if current_validation < best_validation:
                best_validation = current_validation
                best_iteration = iteration
                best_state = copy.deepcopy(policy.state_dict())
            mt = time.perf_counter()
            with torch.no_grad():
                validation_history.append({"iteration": iteration, **trace_metrics(vtrace)})
            monitoring_seconds += time.perf_counter() - mt
            record["validation_cost"] = current_validation
        history.append(record)
    fit_time = time.perf_counter() - start
    final_state_sha256 = state_hash(policy.state_dict())
    policy.load_state_dict(best_state)
    result = {
        "n_fit": n_fit, "policy": policy_name, "seed": seed, "arm": arm,
        "lr_index": lr_index, "learning_rate": learning_rate,
        "lambda_ref": lambda_ref, "best_validation_cost": best_validation,
        "selected_iteration": best_iteration, "iterations": training.iterations,
        "fit_time_seconds": fit_time, "monitoring_seconds": monitoring_seconds,
        "initialization_seconds_shared_per_seed": initialization["elapsed_seconds"],
        "initial_state_sha256": initialization["initial_state_sha256"],
        "selected_state_sha256": state_hash(best_state),
        "final_state_sha256": final_state_sha256,
        "fit_sha256": initialization["fit_sha256"],
        "batches_sha256": initialization["batches_sha256"],
        "gradient_clip": training.gradient_clip,
        "path": str(destination.relative_to(output)),
    }
    torch.save({**result, "state_dict": best_state}, destination / "selected.pt")
    write_frame(pd.DataFrame(history), destination / "training.csv")
    write_frame(pd.DataFrame(validation_history), destination / "validation.csv")
    save_json(result_path, result)
    return result


def evaluate_one(args):
    protocol, output, result = args
    configure()
    output = Path(output)
    destination = output / result["path"]
    if (destination / "evaluation.json").exists():
        return json.loads((destination / "evaluation.json").read_text())
    if not (output / "SELECTION.json").exists():
        raise RuntimeError("Test access requires frozen selection.")
    hydro, _, fit, validation = data(protocol, result["n_fit"])
    test = generate_dataset("test", protocol.n_test, protocol.budget_seeds.test, hydro)
    if not protocol.smoke:
        expected = json.loads((ORIGINAL / "data_manifest.json").read_text())["test"]["sha256"]
        assert dataset_fingerprint(test) == expected
    policy = initial_policy(protocol, hydro, result["seed"], result["policy"])
    payload = torch.load(destination / "selected.pt", weights_only=False, map_location="cpu")
    policy.load_state_dict(payload["state_dict"])
    start = time.perf_counter()
    val_metrics, _ = evaluate_trace(policy, validation, hydro, protocol.device)
    test_metrics, costs = evaluate_trace(policy, test, hydro, protocol.device)
    evaluation_seconds = time.perf_counter() - start
    last_k = max(0, result["selected_iteration"] - 1)
    diagnostic = gradient_diagnostic(
        policy, fit.take(np.arange(min(32, fit.n)), "fit_diagnostic"), hydro,
        protocol.device, lambda_at_step(result["arm"], result["lambda_ref"], last_k),
    )
    evaluation = {
        **result, "validation": val_metrics, "test": test_metrics,
        "fit_gradient_diagnostic": diagnostic,
        "evaluation_seconds": evaluation_seconds, "test_sha256": dataset_fingerprint(test),
    }
    np.save(destination / "test_costs.npy", costs)
    save_json(destination / "evaluation.json", evaluation)
    return evaluation


def baseline_reproduction(output, selected, smoke):
    checks = []
    if smoke:
        return {"status": "smoke: original formal comparison not applicable", "checks": []}
    for result in selected:
        if result["arm"] != "zero":
            continue
        name = result["policy"]
        original_path = ORIGINAL / "checkpoints" / f"n{result['n_fit']}" / f"{name}_selected_seed{result['seed']}.pt"
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        current = torch.load(Path(output) / result["path"] / "selected.pt", map_location="cpu", weights_only=False)
        same = (original["state_dict"].keys() == current["state_dict"].keys()
                and all(torch.equal(original["state_dict"][key], value)
                        for key, value in current["state_dict"].items()))
        checks.append({
            "seed": result["seed"], "bitwise_equal_parameters": same,
            "original_learning_rate": original["learning_rate"],
            "new_learning_rate": result["learning_rate"],
            "original_selected_iteration": original["selected_iteration"],
            "new_selected_iteration": result["selected_iteration"],
            "validation_difference": result["best_validation_cost"] - original["best_validation_cost"],
            "max_parameter_abs_difference": max(float((original["state_dict"][key] - value).abs().max())
                                                  for key, value in current["state_dict"].items()),
        })
    return {"all_bitwise_equal": bool(checks) and all(
        row["bitwise_equal_parameters"]
        and row["original_learning_rate"] == row["new_learning_rate"]
        and row["original_selected_iteration"] == row["new_selected_iteration"]
        and row["validation_difference"] == 0.0 for row in checks), "checks": checks}


def parallel(function, tasks, workers, phase):
    values = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(function, item) for item in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            value = future.result()
            values.append(value)
            print(f"{phase} {i}/{len(tasks)} seed={value.get('seed')} arm={value.get('arm', '')}", flush=True)
    return values


def freeze(output, protocol, n_fit, policy_name, arms, workers):
    hydro, pool, fit, validation = data(protocol, n_fit)
    expected = json.loads((ORIGINAL / "data_manifest.json").read_text())
    if not protocol.smoke:
        assert dataset_fingerprint(pool) == expected["fit_pool"]["sha256"]
        assert dataset_fingerprint(validation) == expected["validation"]["sha256"]
    sources = [
        "iclr_algorithm1_experiments/projection_distance.py",
        "iclr_algorithm1_experiments/projection_distance_run.py",
        "iclr_algorithm1_experiments/config.py", "iclr_algorithm1_experiments/policies.py",
        "algorithm1_gradient_study/training.py", "efficiency_switching_hydro/simulator.py",
        "efficiency_switching_hydro/projection.py", "efficiency_switching_hydro/policies.py",
        "efficiency_switching_hydro/physics.py", "efficiency_switching_hydro/data.py",
        "efficiency_switching_hydro/config.py", "efficiency_switching_hydro/train.py",
    ]
    payload = {
        "experiment": "projection_distance_auxiliary_loss", "protocol": asdict(protocol),
        "n_fit": n_fit, "policy": policy_name, "arms": list(arms),
        "lambda_reference": "per-path full parameter-gradient norms at initialization over ALL fitting paths; median(C norms)/(median(d_pd norms)+1e-12); no clipping; frozen constant",
        "distance": "0.5 sum over stages and reservoir action coordinates of (proposal - projected action)^2; no detach or normalization",
        "active_projection": "fraction of path-stage vectors with Euclidean ||u-x|| > 1e-6; distinct from fraction of binding constraints",
        "lambda_schedule": "k=0 on first optimizer update; zero, 0.1*ref, ref, ref/sqrt(k+1)",
        "selection": "checkpoint and LR minimize original float32 validation mean C for each arm/seed; positive arm with lowest mean validation C across 5 seeds chosen BEFORE test access",
        "extension_rule": "extend validation-selected positive arm to remaining displayed MLP budgets only if its mean validation C and paired mean test C both improve over zero; report seed uncertainty regardless of sign; no new lambda search based on test",
        "test_role": "all 4 arms reported after selection frozen; never used for fitting/LR/arm selection",
        "fit_pool_sha256": dataset_fingerprint(pool), "fit_sha256": dataset_fingerprint(fit),
        "validation_sha256": dataset_fingerprint(validation),
        "expected_test_sha256": expected["test"]["sha256"] if not protocol.smoke else None,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "original_protocol_sha256": sha256(ORIGINAL / "protocol.json"),
        "original_results_sha256": sha256(ORIGINAL / "per_run_results.csv"),
        "workers": workers, "torch_threads_per_worker": 1,
        "timing_note": "wall times include monitoring and concurrent workers; not isolated latency benchmarks; shared initial lambda calibration charged separately per seed",
    }
    path = output / "PROTOCOL.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise RuntimeError("Frozen protocol/source mismatch. Use a new output directory.")
    else:
        save_json(path, payload)
        save_json(output / "environment.json", {
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
        })


def run(output: Path, protocol, n_fit=1000, policy_name="projected_mlp_matched", arms=ARMS, workers=8):
    configure()
    output.mkdir(parents=True, exist_ok=True)
    freeze(output, protocol, n_fit, policy_name, arms, workers)
    start = time.perf_counter()
    parallel(prepare_seed, [(protocol, n_fit, seed, policy_name, str(output)) for seed in protocol.train_seeds],
             min(workers, len(protocol.train_seeds)), "initialization")
    rates = protocol.graph_learning_rates if policy_name == "projected_graph_neural_global" else protocol.mlp_learning_rates
    tasks = [(protocol, n_fit, seed, policy_name, arm, lr_index, str(output))
             for lr_index in range(len(rates)) for seed in protocol.train_seeds for arm in arms]
    # Prespecified shuffle interleaves arms to reduce ordering bias in wall times.
    np.random.default_rng(92751).shuffle(tasks)
    results = parallel(fit_one, tasks, workers, "fit")
    results.sort(key=lambda row: (row["seed"], row["arm"], row["lr_index"]))
    selected = [min((row for row in results if row["seed"] == seed and row["arm"] == arm),
                    key=lambda row: (row["best_validation_cost"], row["lr_index"]))
                for seed in protocol.train_seeds for arm in arms]
    mean_validation = {arm: float(np.mean([row["best_validation_cost"] for row in selected if row["arm"] == arm]))
                       for arm in arms}
    positive = [arm for arm in arms if arm != "zero"]
    chosen = min(positive, key=lambda arm: mean_validation[arm]) if positive else None
    selection = {"mean_validation": mean_validation, "selected_positive_arm": chosen,
                 "validation_improves": chosen is not None and mean_validation[chosen] < mean_validation.get("zero", float("inf")),
                 "selected": selected}
    if (output / "SELECTION.json").exists():
        if json.loads((output / "SELECTION.json").read_text()) != selection:
            raise RuntimeError("Selection changed on resume.")
    else:
        save_json(output / "SELECTION.json", selection)
    reproduction = baseline_reproduction(output, selected, protocol.smoke)
    save_json(output / "BASELINE_REPRODUCTION.json", reproduction)
    if not protocol.smoke and not reproduction["all_bitwise_equal"]:
        raise RuntimeError("Zero-lambda baseline differs from the original; resolve before accessing test data.")
    evaluations = parallel(evaluate_one, [(protocol, str(output), row) for row in selected], workers, "evaluate")
    evaluations.sort(key=lambda row: (row["seed"], row["arm"]))
    save_json(output / "evaluations.json", evaluations)
    save_json(output / "COMPLETED.json", {"n_candidates": len(results), "n_selected": len(selected),
                                         "wall_seconds_this_invocation": time.perf_counter() - start})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "iclr_algorithm1_experiments/outputs/projection_distance_v1/controlled_n1000")
    parser.add_argument("--n-fit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy", default="projected_mlp_matched", choices=("projected_mlp_matched", "projected_graph_neural_global"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = ICLRProtocol.smoke_protocol() if args.smoke else ICLRProtocol()
    n_fit = protocol.base_n_fit if args.smoke else args.n_fit
    if n_fit not in protocol.fit_budgets:
        parser.error("n-fit must match an original displayed budget")
    run(args.output.resolve(), protocol, n_fit, args.policy, tuple(args.arms), args.workers)


if __name__ == "__main__":
    main()
