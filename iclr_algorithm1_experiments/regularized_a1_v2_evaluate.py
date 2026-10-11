"""Idempotently freeze validation choices before evaluating any requested block."""
import argparse
import json
from pathlib import Path
import time
from .regularized_a1_v2_full import core, test_worker, METHODS, NAME
from .projection_distance_run import save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--group",choices=("pilot","all","ready"),default="all")
    args=parser.parse_args()
    entries=json.loads((core.OUT/"manifest.json").read_text())["bundles"]
    if args.group=="pilot":
        entries=[e for e in entries if e["n_fit"]==1000 and (e["name"].startswith("controlled") or e["kind"]=="real")]
    else:
        entries=[e for e in entries if e["name"]!=NAME]
        entries.append({"name":NAME,"methods":METHODS,"kind":"full","n_fit":1000})
    if args.group=="ready":
        ready=[]
        for e in entries:
            b=core.load_bundle(e["name"])
            modes=("full","no_proposal_state","no_moving_recourse") if e["kind"]=="channel" else ("full",)
            controls=e["n_fit"]==1000 and (e["name"].startswith("controlled") or e["kind"]=="real")
            expected=sum(5*len(modes)*(4 if controls else 3)*len(b["settings"][m]["learning_rates"]) for m in e["methods"])
            actual=len(list((core.OUT/"runs"/e["name"]).glob("*/seed*/*/*/result.json")))
            if actual==expected:ready.append(e)
            else:assert actual<expected,(e["name"],actual,expected)
        entries=ready
        print("Complete bundles:",[e["name"] for e in entries],flush=True)
    frozen=[]
    for e in entries:
        b=core.load_bundle(e["name"])
        modes=("full","no_proposal_state","no_moving_recourse") if e["kind"]=="channel" else ("full",)
        zero_expected=e["n_fit"]==1000 and (e["name"].startswith("controlled") or e["kind"]=="real")
        for m in e["methods"]:
            for seed in range(5):
                for mode in modes:
                    for zero in ((False,True) if zero_expected else (False,)):
                        parent=core.OUT/"runs"/e["name"]/m/f"seed{seed}"/mode
                        candidates=[json.loads(f.read_text()) for f in parent.glob("*/result.json")]
                        candidates=[c for c in candidates if (c["strength"]==0)==zero]
                        expected=len(b["settings"][m]["learning_rates"])*(1 if zero else 3)
                        assert len(candidates)==expected,(e["name"],m,seed,mode,zero,len(candidates),expected)
                        eligible=candidates
                        if e["kind"]=="channel" and mode!="full":
                            full_parent=parent.parent/"full"
                            choices=[json.loads(f.read_text()) for f in full_parent.glob("*/result.json")]
                            choice=min(choices,key=lambda c:(c["best_validation_cost"],c["strength"],c["lr_index"]))
                            eligible=[c for c in candidates if c["strength"]==choice["strength"]]
                        chosen=min(eligible,key=lambda c:(c["best_validation_cost"],c["strength"],c["lr_index"]))
                        dest=core.OUT/"selected"/e["name"]/m/f"seed{seed}"/mode/("zero" if zero else "positive")
                        if (dest/"selection.json").exists():
                            old=json.loads((dest/"selection.json").read_text())
                            assert old["chosen"]==chosen
                        else:
                            save_json(dest/"selection.json",{"chosen":chosen,"candidates":candidates,"eligible_candidates":len(eligible),
                                      "frozen_before_test_unix_time":time.time()})
                        frozen.append((e["name"],m,seed,mode,zero))
    # All selections in this requested block are now frozen before any test.
    core.parallel(test_worker,frozen,8)


if __name__=="__main__":
    main()
