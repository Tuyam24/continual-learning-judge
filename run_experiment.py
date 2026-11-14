"""Run a continual-learning strategy on Split MNIST / Split CIFAR-10 and record the accuracy matrix.

Example:
    .venv/bin/python run_experiment.py --strategy naive --seeds 0 1 2
    .venv/bin/python run_experiment.py --dataset cifar10 --strategy derpp --seeds 0

For each seed: train on tasks 0..4 in order; after each task, evaluate test
accuracy on every task seen so far. Prints the accuracy matrix and writes
results/<strategy>_<scenario>.json with per-seed matrices and summary metrics.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cl.data import DATASETS, make_split
from cl.metrics import (SCENARIOS, avg_acc_curve, evaluate, final_avg_acc, forgetting,
                        format_matrix, new_acc_matrix)
from cl.model import MLP, ReducedResNet18
from cl.strategies import STRATEGIES, TrainConfig
import cl.ours  # noqa: F401,E402  (registers "ours" in STRATEGIES)

HERE = Path(__file__).resolve().parent


def pick_device(name: str, dataset: str = "mnist") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # The MNIST MLP is faster on CPU than on Apple MPS; the CIFAR ResNet is ~7x faster on MPS.
    if dataset == "cifar10" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(dataset: str) -> torch.nn.Module:
    return MLP() if dataset == "mnist" else ReducedResNet18()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(strategy_name: str, cfg: TrainConfig, seed: int, device: torch.device,
            n_tasks: int, hparams: dict | None = None, eval_split: str = "test",
            verbose: bool = True, return_stats: bool = False, dataset: str = "mnist"):
    set_seed(seed)
    tasks = make_split(dataset, root=str(HERE / "data"), n_tasks=n_tasks, seed=seed)
    strategy = STRATEGIES[strategy_name](make_model(dataset), cfg, device, seed=seed,
                                         input_shape=tuple(tasks[0].x_train.shape[1:]),
                                         **(hparams or {}))
    R = new_acc_matrix(n_tasks)

    for i, task in enumerate(tasks):
        t0 = time.time()
        strategy.train_task(task)
        for j in range(i + 1):
            x, y = getattr(tasks[j], f"x_{eval_split}"), getattr(tasks[j], f"y_{eval_split}")
            R[i, j] = evaluate(strategy.model, x, y, device,
                               classes=tasks[j].classes, scenario=cfg.scenario)
        if verbose:
            accs = " ".join(f"T{j}={R[i, j] * 100:5.1f}" for j in range(i + 1))
            print(f"  [seed {seed}] after task {i} {task.classes} ({time.time() - t0:4.1f}s): {accs}")
    if return_stats:
        return R, getattr(strategy, "stats", None)
    return R


def parse_hp(items: list[str]) -> dict:
    """--hp key=value pairs; values via ast.literal_eval, falling back to the raw string."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--hp expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=DATASETS)
    p.add_argument("--strategy", default="naive", choices=sorted(STRATEGIES))
    p.add_argument("--scenario", default="class", choices=SCENARIOS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--eval-split", default="test", choices=("test", "val"),
                   help="use 'val' for hyperparameter tuning so the test set stays untouched")
    p.add_argument("--mem-size", type=int, default=500, help="replay: total buffer size")
    p.add_argument("--ewc-lambda", type=float, default=10000.0, help="ewc: penalty strength")
    p.add_argument("--fisher-samples", type=int, default=1000, help="ewc: samples per Fisher estimate")
    p.add_argument("--der-alpha", type=float, default=0.5, help="derpp: weight of the logit-matching term")
    p.add_argument("--der-beta", type=float, default=0.5, help="derpp: weight of the replay CE term")
    p.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                   help="extra strategy hparam (repeatable; value parsed with ast.literal_eval), "
                        "e.g. --hp use_judge=False --hp sleep_steps=40")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    p.add_argument("--quiet", action="store_true", help="only print the summary line")
    p.add_argument("--out", default=None, help="default: results/ (mnist) or results/cifar10/")
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, scenario=args.scenario)
    hparams = {"mem_size": args.mem_size, "ewc_lambda": args.ewc_lambda,
               "fisher_samples": args.fisher_samples, "der_alpha": args.der_alpha, "der_beta": args.der_beta}
    hparams.update(parse_hp(args.hp))
    device = pick_device(args.device, args.dataset)
    print(f"dataset={args.dataset} strategy={args.strategy} scenario={args.scenario} eval={args.eval_split} device={device} "
          f"cfg={asdict(cfg)} hparams={hparams}")

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        R, stats = run_one(args.strategy, cfg, seed, device, args.n_tasks, hparams, args.eval_split,
                           verbose=not args.quiet, return_stats=True, dataset=args.dataset)
        run = {"seed": seed, "acc_matrix": R.tolist(), "avg_acc_curve": avg_acc_curve(R).tolist(),
               "final_avg_acc": final_avg_acc(R), "forgetting": forgetting(R),
               "runtime_s": time.time() - t0}
        if stats is not None:
            run["stats"] = stats
            if not args.quiet:
                print("  stats: " + json.dumps({k: v for k, v in stats.items() if k != "per_task"}))
        runs.append(run)
        if not args.quiet:
            print(format_matrix(R))
        print(f"  final avg acc = {final_avg_acc(R) * 100:.2f}%   forgetting = {forgetting(R) * 100:.2f}%\n")

    accs = np.array([r["final_avg_acc"] for r in runs])
    fgts = np.array([r["forgetting"] for r in runs])
    summary = {"final_avg_acc_mean": accs.mean(), "final_avg_acc_std": accs.std(),
               "forgetting_mean": fgts.mean(), "forgetting_std": fgts.std()}
    print(f"== {args.strategy} ({args.scenario}-IL, {len(runs)} seed(s)): "
          f"ACC {accs.mean() * 100:.2f} ± {accs.std() * 100:.2f}%  "
          f"forgetting {fgts.mean() * 100:.2f} ± {fgts.std() * 100:.2f}%  [{args.eval_split}]")

    out = Path(args.out) if args.out else (HERE / "results" if args.dataset == "mnist"
                                           else HERE / "results" / args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.tag}" if args.tag else "") + ("_val" if args.eval_split == "val" else "")
    path = out / f"{args.strategy}_{args.scenario}{suffix}.json"
    path.write_text(json.dumps({"dataset": args.dataset, "strategy": args.strategy, "config": asdict(cfg), "hparams": hparams,
                                "eval_split": args.eval_split, "device": str(device),
                                "n_tasks": args.n_tasks, "summary": summary, "runs": runs}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
