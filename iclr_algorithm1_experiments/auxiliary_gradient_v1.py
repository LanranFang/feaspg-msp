"""Isolate detachment of projected actions in the proposal-distance loss.

The original simulator and all earlier experiment records remain immutable.
All operating-cost and state-recurrence derivatives remain complete.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import regularized_a1_v2 as core
from .projection_distance import projection_distance, regularized_objective
from .projection_distance_run import configure, save_json, sha256, state_hash

OUT = core.ROOT / "iclr_algorithm1_experiments/outputs/auxiliary_gradient_v1"
NAME, METHOD = "channel_n1000", "mlp"


def loss_for(trace, weight, detach_auxiliary_action=False):
    if not detach_auxiliary_action:
        return regularized_objective(trace, weight)
    distance = .5 * (trace.proposals - trace.actions.detach()).square().sum((1, 2)).mean()
    return trace.mean_cost + weight * distance


def read(path):
    return json.loads(path.read_text())


def prepare():
    configure()
    protocol = OUT / "protocol.json"
    if protocol.exists():
        frozen = read(protocol)
        assert frozen["runner_sha256"] == sha256(Path(__file__))
        return frozen
    b = core.load_bundle(NAME)
    controls = {}
    for seed in range(5):
        selection = core.OUT / "selected" / NAME / METHOD / f"seed{seed}/full/positive/selection.json"
        selected = read(selection)["chosen"]
        candidates = [read(core.candidate_dir(NAME, METHOD, seed, selected["strength"], i) / "result.json")
                      for i in range(3)]
        assert min(candidates, key=lambda x:(x["best_validation_cost"], x["lr_index"])) == selected
        controls[str(seed)] = {"selected": selected, "candidates": candidates,
                               "selection_source": str(selection), "selection_sha256": sha256(selection)}
    frozen = {"prepared_unix_time": time.time(), "runner_sha256": sha256(Path(__file__)),
              "bundle": NAME, "bundle_sha256": sha256(core.OUT / "bundles" / f"{NAME}.pkl"),
              "seeds": list(range(5)), "settings": b["settings"][METHOD],
              "n_fit": b["fit"].n, "n_validation": b["validation"].n, "n_test": b["test"].n,
              "positive_controls": controls,
              "arms": ["zero", "complete", "detach_auxiliary_action"],
              "scope": "Only the auxiliary residual uses actions.detach(); all rollout and cost derivatives remain full.",
              "selection": "Positive strength is frozen from the existing full-gradient validation selection, shared by both positive arms. Each arm selects LR and checkpoint by validation cost. Zero controls use the same three LRs. New selections freeze before new test evaluation.",
              "new_candidates": 30, "reused_complete_candidates": 15,
              "reproduction": "One complete 500-update candidate is independently replayed and checked against its frozen checkpoint and history."}
    save_json(protocol, frozen)
    return frozen


def fit(task):
    configure()
    seed, arm, lr_index = task
    dest = OUT / "runs" / f"seed{seed}" / arm / f"lr{lr_index}"
    if (dest / "result.json").exists():
        return read(dest / "result.json")
    dest.mkdir(parents=True, exist_ok=True)
    protocol = read(OUT / "protocol.json")
    b = core.load_bundle(NAME)
    c = b["settings"][METHOD]
    actor = core.build(b, METHOD, seed)
    cal = read(core.calibration_path(NAME, METHOD, seed))
    assert state_hash(actor.state_dict()) == cal["initial_state_sha256"]
    strength = 0. if arm == "zero" else protocol["positive_controls"][str(seed)]["selected"]["strength"]
    batches = np.random.default_rng(seed).permutation(b["fit"].n)[:c["iterations"]*2].reshape(c["iterations"], 2)
    batch_hash = hashlib.sha256(batches.tobytes()).hexdigest()
    assert batch_hash == protocol["positive_controls"][str(seed)]["selected"]["batches_sha256"]
    lr = c["learning_rates"][lr_index]
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-6)
    best_state = copy.deepcopy(actor.state_dict())
    metrics, _ = core.evaluate(actor, b["validation"], b)
    initial_validation = best_cost = metrics["mean_cost"]
    best_iteration = 0
    histories = [{"iteration": 0, "validation_cost": best_cost, "lambda": strength*cal["lambda_ref"]}]
    start = time.perf_counter()
    for iteration in range(1, c["iterations"]+1):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        tr = core.simulate(actor, b["fit"].take(batches[iteration-1]), b)
        weight = strength*cal["lambda_ref"]/math.sqrt(iteration)
        loss = loss_for(tr, weight, arm == "detach_auxiliary_action")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 10))
        assert math.isfinite(norm) and math.isfinite(float(loss.detach()))
        optimizer.step()
        row = {"iteration": iteration, "lambda": weight, "train_cost": float(tr.mean_cost.detach()),
               "train_distance": float(projection_distance(tr).mean().detach()), "gradient_norm": norm}
        if iteration % c["validation_every"] == 0 or iteration == c["iterations"]:
            metrics, _ = core.evaluate(actor, b["validation"], b)
            row.update({"validation_cost": metrics["mean_cost"], "validation_distance": metrics["proposal_distance"],
                        "active_projection_fraction": metrics["active_projection_fraction"]})
            if metrics["mean_cost"] < best_cost:
                best_cost, best_iteration = metrics["mean_cost"], iteration
                best_state = copy.deepcopy(actor.state_dict())
        histories.append(row)
    elapsed = time.perf_counter()-start
    actor.load_state_dict(best_state)
    metrics, _ = core.evaluate(actor, b["validation"], b)
    result = {"seed": seed, "arm": arm, "lr_index": lr_index, "learning_rate": lr,
              "strength": strength, "lambda_ref": cal["lambda_ref"], "initial_validation": initial_validation,
              "best_validation_cost": best_cost, "selected_iteration": best_iteration,
              "validation": metrics, "runtime_seconds": elapsed, "iterations": c["iterations"],
              "n_fit": b["fit"].n, "batches_sha256": batch_hash,
              "initial_state_sha256": cal["initial_state_sha256"],
              "selected_state_sha256": state_hash(best_state), "path": str(dest)}
    torch.save(best_state, dest / "selected.pt")
    pd.DataFrame(histories).to_csv(dest / "history.csv", index=False)
    if arm == "complete_replay":
        original = protocol["positive_controls"][str(seed)]["candidates"][lr_index]
        assert result["selected_state_sha256"] == original["selected_state_sha256"]
        assert result["best_validation_cost"] == original["best_validation_cost"]
        old = pd.read_csv(Path(original["path"])/"history.csv", float_precision="round_trip")
        new = pd.read_csv(dest/"history.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(old, new)
        result["reproduces_original_checkpoint_and_history_exactly"] = True
    save_json(dest / "result.json", result)
    return result


def run_fits(workers):
    p = prepare()
    tasks = [(s,a,i) for s in p["seeds"] for a in ("zero","detach_auxiliary_action") for i in range(3)]
    tasks.insert(0,(0,"complete_replay",p["positive_controls"]["0"]["selected"]["lr_index"]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        fs={ex.submit(fit,t):t for t in tasks}
        for f in as_completed(fs):
            r=f.result()
            print("DONE",fs[f],"validation",r["best_validation_cost"],"seconds",round(r["runtime_seconds"],1),flush=True)


def evaluate():
    configure()
    p=prepare()
    selections=[]
    for seed in p["seeds"]:
        for arm in p["arms"]:
            candidates = p["positive_controls"][str(seed)]["candidates"] if arm=="complete" else [read(OUT/"runs"/f"seed{seed}"/arm/f"lr{i}/result.json") for i in range(3)]
            chosen=min(candidates,key=lambda r:(r["best_validation_cost"],r["lr_index"]))
            dest=OUT/"selected"/f"seed{seed}"/arm
            f=dest/"selection.json"
            if not f.exists(): save_json(f,{"seed":seed,"arm":arm,"chosen":chosen,"candidates":candidates,"frozen_unix_time":time.time()})
            else: assert read(f)["chosen"]==chosen
            selections.append((seed,arm,dest,chosen))
    # Every selection in this experiment is frozen before any new test report.
    b=core.load_bundle(NAME)
    rows=[]
    for seed,arm,dest,r in selections:
        actor=core.build(b,METHOD,seed)
        actor.load_state_dict(torch.load(Path(r["path"])/"selected.pt",weights_only=True))
        assert state_hash(actor.state_dict())==r["selected_state_sha256"]
        metrics,costs=core.evaluate(actor,b["test"],b)
        tr=core.simulate(actor,b["fit"].take(np.arange(32)),b)
        params=tuple(actor.parameters())
        weight=r["strength"]*r["lambda_ref"]/math.sqrt(max(1,r["selected_iteration"]))
        def norm(y,retain):
            gs=torch.autograd.grad(y,params,retain_graph=retain,allow_unused=True)
            return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
        gc=norm(tr.mean_cost,True)
        total=norm(loss_for(tr,weight,arm=="detach_auxiliary_action"),True)
        complete=norm(regularized_objective(tr,weight),False)
        row={"seed":seed,"arm":arm,"validation_cost":r["best_validation_cost"],
             "test_cost":metrics["mean_cost_float64"],"proposal_distance":metrics["proposal_distance"],
             "active_projection_fraction":metrics["active_projection_fraction"],
             "max_action_violation":metrics["max_action_violation"],"max_storage_violation":metrics["max_storage_violation"],
             "cost_gradient_norm":gc,"training_rule_gradient_norm":total,"complete_gradient_norm":complete,
             "strength":r["strength"],"lambda_ref":r["lambda_ref"],"lambda_at_checkpoint":weight,
             "learning_rate":r["learning_rate"],"selected_iteration":r["selected_iteration"],
             "three_lr_fit_seconds":sum(c["runtime_seconds"] for c in read(dest/"selection.json")["candidates"])}
        np.savez_compressed(dest/"test_costs.npz",costs=costs)
        save_json(dest/"evaluation.json",row)
        rows.append(row)
    f=pd.DataFrame(rows);f.to_csv(OUT/"per_seed.csv",index=False)
    f.groupby("arm").agg(test_mean=("test_cost","mean"),test_sd=("test_cost","std"),
                         validation_mean=("validation_cost","mean"),validation_sd=("validation_cost","std"),
                         distance_mean=("proposal_distance","mean"),distance_sd=("proposal_distance","std"),
                         active_mean=("active_projection_fraction","mean"),cost_gradient_mean=("cost_gradient_norm","mean"),
                         training_gradient_mean=("training_rule_gradient_norm","mean"),
                         fit_seconds_mean=("three_lr_fit_seconds","mean")).to_csv(OUT/"summary.csv")
    print((OUT/"summary.csv").read_text(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","fit","evaluate"));parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if args.command=="prepare": prepare()
    elif args.command=="fit": run_fits(args.workers)
    else: evaluate()
