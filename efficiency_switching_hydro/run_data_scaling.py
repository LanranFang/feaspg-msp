"""Run nested-fit-pool data-scaling comparisons for selected hydro policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUTS = ROOT / "outputs"
DEFAULT_BUDGETS = (250, 500)
POLICIES = (
    "myopic",
    "projected_mlp_pathwise",
    "projected_graph_neural_global",
)
RL_ALGORITHMS = ("ppo", "sac", "td3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--fit-pool-size", type=int, default=5000)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--summary-name", default="data_scaling_summary")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def output_directory(n_fit: int, seeds: list[int]) -> Path:
    seed_suffix = "-".join(str(seed) for seed in seeds)
    return OUTPUTS / f"data_scaling_n{n_fit}_seeds_{seed_suffix}"


def run_budget(n_fit: int, fit_pool_size: int, seeds: list[int], rerun: bool) -> Path:
    if n_fit <= 0:
        raise ValueError("Every data budget must be positive.")
    if n_fit > fit_pool_size:
        raise ValueError("Every data budget must be at most fit-pool-size.")
    if n_fit % 2:
        raise ValueError("Data budgets must be even because direct policies use batch size 2.")

    output = output_directory(n_fit, seeds)
    result = output / "main_results.csv"
    if result.exists() and not rerun:
        print(f"skip completed N={n_fit}: {output}", flush=True)
        return output

    command = [
        sys.executable,
        "-m",
        "efficiency_switching_hydro.run_experiment",
        "--profile",
        "decision_switching",
        "--output-dir",
        str(output),
        "--n-reservoirs",
        "12",
        "--horizon",
        "48",
        "--shock-factors",
        "4",
        "--n-fit",
        str(n_fit),
        "--fit-pool-size",
        str(fit_pool_size),
        "--n-validation",
        "256",
        "--n-test",
        "1024",
        "--n-calibration",
        "256",
        "--n-evaluation",
        "512",
        "--iterations",
        str(n_fit // 2),
        "--batch-size",
        "2",
        "--validation-every",
        "10",
        "--policies",
        *POLICIES,
        "--lr-neural",
        "0.0006",
        "--lr-mlp",
        "0.0003",
        "--rl-algorithms",
        *RL_ALGORITHMS,
        "--rl-validation-every",
        "50",
        "--rl-network-width",
        "256",
        "--rl-network-depth",
        "2",
        "--ppo-lr",
        "0.0003",
        "--ppo-rollout-episodes",
        "10",
        "--ppo-epochs",
        "10",
        "--ppo-batch-size",
        "256",
        "--sac-lr",
        "0.0003",
        "--td3-lr",
        "0.001",
        "--rl-replay-buffer-size",
        "100000",
        "--rl-learning-starts",
        "1000",
        "--rl-batch-size",
        "256",
        "--rl-gradient-steps",
        "1",
        "--rl-n-steps",
        "3",
        "--td3-action-noise",
        "0.1",
        "--rl-reward-scale",
        "1000",
        "--rl-gamma",
        "1",
        "--rl-device",
        "auto",
        "--ppo-device",
        "cpu",
        "--off-policy-device",
        "auto",
        "--train-seeds",
        *(str(seed) for seed in seeds),
        "--skip-sddp",
    ]
    print(f"start N={n_fit}: {output}", flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)
    return output


def collect(outputs: list[tuple[int, Path]]) -> pd.DataFrame:
    pieces = []
    for n_fit, output in outputs:
        path = output / "main_results.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path)
        data.insert(0, "n_fit", n_fit)
        pieces.append(data)
    if not pieces:
        raise FileNotFoundError("No completed data-scaling outputs were found.")
    return pd.concat(pieces, ignore_index=True, sort=False)


def plot_scaling(data: pd.DataFrame, path: Path) -> None:
    labels = {
        "myopic": "Myopic",
        "projected_mlp_pathwise": "Projected pathwise MLP",
        "projected_graph_neural_global": "Projected graph neural",
        "rl_ppo": "PPO",
        "rl_sac": "SAC",
        "rl_td3": "TD3",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, group in data.groupby("policy", sort=False):
        ordered = group.sort_values("n_fit")
        ax.plot(
            ordered["n_fit"],
            ordered["test_mean_cost"],
            marker="o",
            linewidth=2,
            label=labels.get(policy, policy),
        )
    budgets = sorted(data["n_fit"].unique())
    ax.set_xticks(budgets)
    if len(budgets) >= 3 and budgets[-1] / budgets[0] >= 4:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, labels=[str(value) for value in budgets])
    ax.set_xlabel("Number of fit trajectories")
    ax.set_ylabel("Mean test cost (lower is better)")
    ax.set_title("Hydro policy data scaling on a nested fit pool")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    budgets = sorted(set(args.budgets))
    outputs = [
        (budget, run_budget(budget, args.fit_pool_size, args.train_seeds, args.rerun))
        for budget in budgets
    ]
    destination = OUTPUTS / args.summary_name
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = collect(outputs)
    data.to_csv(destination / "data_scaling_results.csv", index=False)
    plot_scaling(data, figures / "data_scaling.png")
    print(f"wrote data-scaling summary to {destination}", flush=True)
    columns = [
        "n_fit",
        "policy",
        "test_mean_cost",
        "fit_wall_time_seconds_mean",
    ]
    print(data[columns].sort_values(["n_fit", "test_mean_cost"]).to_string(index=False))


if __name__ == "__main__":
    main()
