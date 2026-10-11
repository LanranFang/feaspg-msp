"""Isolated positive-lambda reruns of all A1 fitting experiments in the draft.

All lambda/LR candidates use the SAME fitting prefix. Selection is per budget
on validation operating cost; no larger-budget fitting paths are used to tune
a smaller-budget policy. Original simulators and frozen results are immutable.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real_data_hydro_experiments"))
OUT = ROOT / "iclr_algorithm1_experiments/outputs/regularized_a1_v2"
from .config import ICLRProtocol
from .projection_distance import projection_distance, regularized_objective, trace_metrics
from .projection_distance_run import configure, save_json, sha256, state_hash

STRENGTHS = (.1, 1., 10.)


def load_bundle(name):
    with (OUT / "bundles" / f"{name}.pkl").open("rb") as f:
        return pickle.load(f)


def simulate(actor, data, b, raw=False, mode="full", dtype=torch.float32):
    if b["kind"] == "real":
        from real_hydro.simulator import simulate as sim
        return sim(actor, data, b["raw_system"] if raw else b["system"],
                   n_regimes=b["n_regimes"], dtype=dtype, device="cpu")
    from efficiency_switching_hydro.simulator import simulate as sim
    return sim(actor, data, b["system"], gradient_mode=mode, dtype=dtype)


def build(b, method, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if b["kind"] == "real":
        from real_hydro.policies import ProjectedMLPPolicy, ProjectedDeepSetsPolicy
        c = b["settings"][method]
        cls = ProjectedMLPPolicy if method == "algorithm1_mlp" else ProjectedDeepSetsPolicy
        kw = {"width": c["width"]}
        if method == "algorithm1_mlp":
            kw["depth"] = c["depth"]
        return cls(b["system"], b["n_regimes"], **kw)
    if b["kind"] == "channel":
        from algorithm1_gradient_study.models import build_policy, initialize_channel_policy
        actor = build_policy("mlp", b["system"], b["study"])
        initialize_channel_policy(actor, 40000 + seed)
        return actor
    from .policies import build_policy
    return build_policy(method, b["system"], ICLRProtocol())


def candidate_dir(name, method, seed, strength, lr_index, mode="full"):
    return OUT / "runs" / name / method / f"seed{seed}" / mode / f"a{strength:g}_lr{lr_index}"


def calibration_path(name, method, seed):
    return OUT / "initialization" / name / method / f"seed{seed}.json"


def calibrate(task):
    name, method, seed = task
    configure()
    path = calibration_path(name, method, seed)
    if path.exists():
        return json.loads(path.read_text())
    b = load_bundle(name)
    actor = build(b, method, seed)
    parameters = tuple(actor.parameters())
    rows = []
    start = time.perf_counter()
    for j in range(min(32, b["fit"].n)):
        tr = simulate(actor, b["fit"].take(np.array([j])), b)
        distance = projection_distance(tr).mean()
        gc = torch.autograd.grad(tr.mean_cost, parameters, retain_graph=True, allow_unused=True)
        gd = torch.autograd.grad(distance, parameters, allow_unused=True)
        norm = lambda gs: math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
        rows.append({"fit_path": j, "cost_grad": norm(gc), "distance_grad": norm(gd),
                     "cost": float(tr.mean_cost.detach()), "distance": float(distance.detach())})
    mc, md = np.median([r["cost_grad"] for r in rows]), np.median([r["distance_grad"] for r in rows])
    if mc > 0 and md > 0:
        ref = float(mc / (md + 1e-12))
        rule = "ratio_of_median_parameter_gradient_norms"
    else:
        cap = np.asarray(b["system"].release_capacity)
        ref = max(float(np.mean([abs(r["cost"]) for r in rows])), 1e-12) / max(.5 * b["horizon"] * float(cap @ cap), 1e-12)
        rule = "degenerate_gradient_fallback_cost_over_release_capacity_squared"
    payload = {"bundle": name, "method": method, "seed": seed, "lambda_ref": ref,
               "rule": rule, "median_cost_grad": float(mc), "median_distance_grad": float(md),
               "n_fit_paths_used": len(rows), "rows": rows,
               "initial_state_sha256": state_hash(actor.state_dict()),
               "elapsed_seconds": time.perf_counter()-start}
    assert np.isfinite(ref) and ref > 0
    save_json(path, payload)
    return payload


@torch.no_grad()
def evaluate(actor, data, b, raw=False):
    actor.eval()
    tr = simulate(actor, data, b, raw=raw)
    values = tr.total_cost.cpu().numpy().astype(np.float64)
    return {**trace_metrics(tr), "mean_cost_float64": float(values.mean())}, values


def fit(task):
    name, method, seed, strength, lr_index, mode = task
    configure()
    dest = candidate_dir(*task)
    if (dest / "result.json").exists():
        return json.loads((dest / "result.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    b = load_bundle(name)
    c = b["settings"][method]
    actor = build(b, method, seed)
    cal = json.loads(calibration_path(name, method, seed).read_text())
    assert state_hash(actor.state_dict()) == cal["initial_state_sha256"]
    lr = c["learning_rates"][lr_index]
    batches = np.random.default_rng(seed).permutation(b["fit"].n)[:c["iterations"]*2].reshape(c["iterations"], 2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-6)
    best_state = copy.deepcopy(actor.state_dict())
    metrics, _ = evaluate(actor, b["validation"], b)
    initial_validation = metrics["mean_cost"]
    best_cost = initial_validation
    best_iteration = 0
    histories = [{"iteration": 0, "validation_cost": best_cost, "lambda": strength*cal["lambda_ref"]}]
    maximum_gradient = 0.
    start = time.perf_counter()
    for iteration in range(1, c["iterations"]+1):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        tr = simulate(actor, b["fit"].take(batches[iteration-1]), b, mode=mode)
        lam = strength * cal["lambda_ref"] / math.sqrt(iteration)
        loss = regularized_objective(tr, lam)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 10))
        assert math.isfinite(norm) and math.isfinite(float(loss.detach()))
        maximum_gradient = max(norm, maximum_gradient)
        optimizer.step()
        row = {"iteration": iteration, "lambda": lam, "train_cost": float(tr.mean_cost.detach()),
               "train_distance": float(projection_distance(tr).mean().detach()), "gradient_norm": norm}
        if iteration % c["validation_every"] == 0 or iteration == c["iterations"]:
            metrics, _ = evaluate(actor, b["validation"], b)
            row.update({"validation_cost": metrics["mean_cost"], "validation_distance": metrics["proposal_distance"],
                        "active_projection_fraction": metrics["active_projection_fraction"]})
            if metrics["mean_cost"] < best_cost:
                best_cost = metrics["mean_cost"]
                best_state = copy.deepcopy(actor.state_dict())
                best_iteration = iteration
        histories.append(row)
    elapsed = time.perf_counter() - start
    actor.load_state_dict(best_state)
    metrics, _ = evaluate(actor, b["validation"], b)
    result = {"bundle": name, "method": method, "seed": seed, "strength": strength,
              "lr_index": lr_index, "learning_rate": lr, "mode": mode,
              "lambda_ref": cal["lambda_ref"], "initial_validation": initial_validation,
              "best_validation_cost": best_cost, "selected_iteration": best_iteration,
              "validation": metrics, "runtime_seconds": elapsed,
              "iterations": c["iterations"], "n_fit": b["fit"].n,
              "max_gradient_norm": maximum_gradient,
              "initial_state_sha256": cal["initial_state_sha256"],
              "selected_state_sha256": state_hash(best_state),
              "batches_sha256": hashlib.sha256(batches.tobytes()).hexdigest(),
              "calibration_seconds": cal["elapsed_seconds"], "path": str(dest)}
    torch.save(best_state, dest / "selected.pt")
    pd.DataFrame(histories).to_csv(dest / "history.csv", index=False)
    save_json(dest / "result.json", result)
    return result


def test_selected(task):
    name, method, seed, mode, zero = task
    configure()
    dest = OUT / "selected" / name / method / f"seed{seed}" / mode / ("zero" if zero else "positive")
    if (dest / "evaluation.json").exists():
        return json.loads((dest / "evaluation.json").read_text())
    selection = json.loads((dest / "selection.json").read_text())
    chosen = selection["chosen"]
    b = load_bundle(name)
    actor = build(b, method, seed)
    actor.load_state_dict(torch.load(Path(chosen["path"]) / "selected.pt", weights_only=True))
    assert state_hash(actor.state_dict()) == chosen["selected_state_sha256"]
    metrics, costs = evaluate(actor, b["test"], b, raw=True)
    result = {"bundle": name, "method": method, "seed": seed, "mode": mode,
              "zero": zero, "selection": chosen, "test": metrics,
              "total_fit_seconds": sum(c["runtime_seconds"] for c in selection["candidates"]),
              "calibration_seconds": chosen["calibration_seconds"],
              "candidate_count": len(selection["candidates"])}
    arrays = {"test_costs": costs}
    if b.get("historical") is not None:
        hm, hc = evaluate(actor, b["historical"], b, raw=True)
        result["historical"] = hm
        arrays["historical_costs"] = hc
    np.savez_compressed(dest / "path_costs.npz", **arrays)
    save_json(dest / "evaluation.json", result)
    return result


def prepare():
    from artifact.inputs import prepare_bundles
    prepare_bundles(OUT)


def parallel(fn, tasks, workers):
    if not tasks:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(fn, t) for t in tasks]):
            r = future.result()
            results.append(r)
            print(json.dumps({k: r[k] for k in ("bundle", "method", "seed", "strength", "mode", "best_validation_cost") if k in r}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--group", choices=("all", "controlled", "real", "scaling", "channel", "pilot"), default="all")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    prepare()
    entries = [e for e in json.loads((OUT / "manifest.json").read_text())["bundles"] if e["kind"] != "full"]
    def include(e):
        if args.group == "all": return True
        if args.group == "pilot": return e["n_fit"] == 1000 and (e["name"].startswith("controlled") or e["kind"] == "real")
        if args.group == "real": return e["kind"] == "real"
        return e["name"].startswith(args.group)
    entries = [e for e in entries if include(e)]
    if args.command == "prepare": return
    if args.command == "train":
        calibration_tasks = [(e["name"], m, seed) for e in entries for m in e["methods"] for seed in range(5)]
        parallel(calibrate, calibration_tasks, args.workers)
        tasks = []
        for e in entries:
            b = load_bundle(e["name"])
            modes = ("full", "no_proposal_state", "no_moving_recourse") if e["kind"] == "channel" else ("full",)
            for m in e["methods"]:
                for seed in range(5):
                    for mode in modes:
                        strengths = (0., *STRENGTHS) if e["n_fit"] == 1000 and (e["name"].startswith("controlled") or e["kind"] == "real") else STRENGTHS
                        for a in strengths:
                            for lr in range(len(b["settings"][m]["learning_rates"])):
                                tasks.append((e["name"], m, seed, a, lr, mode))
        # Shuffle prespecified tasks to mix costs and avoid an all-long tail.
        np.random.default_rng(9172026).shuffle(tasks)
        parallel(fit, tasks, args.workers)
    else:
        evaluation_tasks = []
        for e in entries:
            modes = ("full", "no_proposal_state", "no_moving_recourse") if e["kind"] == "channel" else ("full",)
            for m in e["methods"]:
                for seed in range(5):
                    for mode in modes:
                        for zero in (False, True):
                            parent = OUT / "runs" / e["name"] / m / f"seed{seed}" / mode
                            candidates = [json.loads(f.read_text()) for f in parent.glob("*/result.json")]
                            candidates = [c for c in candidates if (c["strength"] == 0) == zero]
                            if zero and not candidates: continue
                            expected = len(load_bundle(e["name"])["settings"][m]["learning_rates"])*(1 if zero else 3)
                            assert len(candidates) == expected, (e["name"], m, seed, mode, len(candidates), expected)
                            eligible = candidates
                            if e["kind"] == "channel" and mode != "full":
                                full_parent = OUT / "runs" / e["name"] / m / f"seed{seed}" / "full"
                                full_candidates = [json.loads(f.read_text()) for f in full_parent.glob("*/result.json")]
                                full_choice = min(full_candidates, key=lambda c: (c["best_validation_cost"], c["strength"], c["lr_index"]))
                                eligible = [c for c in candidates if c["strength"] == full_choice["strength"]]
                            chosen = min(eligible, key=lambda c: (c["best_validation_cost"], c["strength"], c["lr_index"]))
                            dest = OUT / "selected" / e["name"] / m / f"seed{seed}" / mode / ("zero" if zero else "positive")
                            save_json(dest / "selection.json", {"chosen": chosen, "candidates": candidates,
                                     "eligible_candidates": len(eligible),
                                     "frozen_before_test_unix_time": time.time()})
                            evaluation_tasks.append((e["name"], m, seed, mode, zero))
        parallel(test_selected, evaluation_tasks, min(args.workers, 10))


if __name__ == "__main__":
    main()
