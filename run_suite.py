"""Benchmark suite v2: DER++, compute-matched baselines, and Split CIFAR-10.

Runs two chains at the same time (MNIST on CPU, CIFAR-10 on the GPU/MPS) and logs
progress to results/suite_progress.log. Every hyperparameter choice is made on the
val split; test runs only happen after the choice is fixed.

    .venv/bin/python run_suite.py            # everything
    .venv/bin/python run_suite.py --only mnist
    .venv/bin/python run_suite.py --only relax_mnist        # round 3: relaxed judge
    .venv/bin/python run_suite.py --only relax_cifar10      # (on a GPU, e.g. Kaggle)
    .venv/bin/python run_suite.py --only round4_cifar10     # round 4: keep-wake rollback (GPU)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
LOG = HERE / "results" / "suite_progress.log"
SWEEP = HERE / "results" / "sweep_v2"
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        print(line, flush=True)
        with LOG.open("a") as f:
            f.write(line + "\n")


def run(args: list[str], name: str) -> dict | None:
    """Run run_experiment.py; return the saved JSON (or None on failure)."""
    t0 = time.time()
    p = subprocess.run([PY, str(HERE / "run_experiment.py"), "--quiet", *args],
                       cwd=HERE, capture_output=True, text=True)
    saved = [l.split("saved ", 1)[1] for l in p.stdout.splitlines() if l.startswith("saved ")]
    if p.returncode or not saved:
        log(f"FAILED {name} ({time.time() - t0:.0f}s): {(p.stderr or p.stdout)[-600:]}")
        return None
    res = json.loads(Path(saved[0]).read_text())
    s = res["summary"]
    log(f"{name}: ACC {s['final_avg_acc_mean'] * 100:.2f} ± {s['final_avg_acc_std'] * 100:.2f}  "
        f"forgetting {s['forgetting_mean'] * 100:.2f}  ({time.time() - t0:.0f}s)")
    return res


def acc(res: dict | None) -> float:
    return -1.0 if res is None else res["summary"]["final_avg_acc_mean"]


def steps(res: dict | None) -> float:
    runs = (res or {}).get("runs", [])
    vals = [r["stats"]["grad_steps"] for r in runs if "stats" in r]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------- MNIST (CPU)
def mnist_chain() -> None:
    base = ["--dataset", "mnist", "--device", "cpu"]
    log("MNIST: tuning DER++ alpha/beta on val (seeds 0 1)")
    best, best_cfg = -1.0, None
    for a in (0.1, 0.2, 0.5, 1.0):
        for b in (0.5, 1.0):
            r = run(base + ["--strategy", "derpp", "--der-alpha", str(a), "--der-beta", str(b),
                            "--eval-split", "val", "--seeds", "0", "1", "--out", str(SWEEP / "mnist"),
                            "--tag", f"a{a}_b{b}"], f"mnist derpp val a={a} b={b}")
            if acc(r) > best:
                best, best_cfg = acc(r), (a, b)
    a, b = best_cfg
    log(f"MNIST: DER++ chosen alpha={a} beta={b} (val ACC {best * 100:.2f})")
    der = ["--der-alpha", str(a), "--der-beta", str(b)]

    for sc in ("class", "task"):
        run(base + ["--strategy", "derpp", "--scenario", sc, "--seeds", "0", "1", "2"] + der,
            f"mnist derpp {sc} test")
    # Compute-matched: ours uses ~2.8k grad steps vs 1.7k at 2 epochs; 4 epochs = 3.4k (>= ours).
    for st in ("replay", "derpp"):
        run(base + ["--strategy", st, "--epochs", "4", "--seeds", "0", "1", "2", "--tag", "epochs4"] + der,
            f"mnist {st} class test, 4 epochs (compute-matched)")
    log("MNIST chain done")


# ------------------------------------------------------------ CIFAR-10 (MPS)
def cifar_chain() -> None:
    E = "5"  # epochs per task (the literature uses 50; too slow for a laptop)
    base = ["--dataset", "cifar10", "--epochs", E]
    log("CIFAR: choosing learning rate with replay on val (seed 0)")
    best, lr = -1.0, None
    for cand in ("0.01", "0.03", "0.1"):
        r = run(base + ["--strategy", "replay", "--lr", cand, "--eval-split", "val", "--seeds", "0",
                        "--out", str(SWEEP / "cifar10"), "--tag", f"lr{cand}"], f"cifar replay val lr={cand}")
        if acc(r) > best:
            best, lr = acc(r), cand
    log(f"CIFAR: lr={lr} (val ACC {best * 100:.2f})")
    base += ["--lr", lr]

    log("CIFAR: choosing DER++ alpha on val (seed 0)")
    best, alpha = -1.0, None
    for a in ("0.1", "0.3", "1.0"):
        r = run(base + ["--strategy", "derpp", "--der-alpha", a, "--der-beta", "0.5", "--eval-split", "val",
                        "--seeds", "0", "--out", str(SWEEP / "cifar10"), "--tag", f"a{a}"],
                f"cifar derpp val alpha={a}")
        if acc(r) > best:
            best, alpha = acc(r), a
    log(f"CIFAR: DER++ alpha={alpha} (val ACC {best * 100:.2f})")
    der = ["--der-alpha", alpha, "--der-beta", "0.5"]

    seeds = ["--seeds", "0", "1", "2"]
    res = {}
    for st in ("naive", "replay", "derpp", "ours", "ewc"):
        res[st] = run(base + ["--strategy", st] + seeds + der, f"cifar {st} class test")
    # Only the judge ablation on CIFAR (the judge is the least-published component); the
    # full ablation set is on MNIST. Each CIFAR "ours" run is ~9 min per seed on MPS.
    for tag, hp in (("no-judge", "use_judge=False"),):
        run(base + ["--strategy", "ours", "--hp", hp, "--tag", tag] + seeds, f"cifar ours {tag} class test")

    # Compute-matched baselines: scale epochs so their grad steps >= ours.
    ratio = steps(res["ours"]) / steps(res["replay"]) if res.get("ours") and res.get("replay") else 2.0
    e2 = str(max(int(E) + 1, int(-(-int(E) * ratio // 1))))
    log(f"CIFAR: ours/replay grad-step ratio {ratio:.2f} -> compute-matched epochs {e2}")
    for st in ("replay", "derpp"):
        run(["--dataset", "cifar10", "--epochs", e2, "--lr", lr, "--strategy", st, "--tag", f"epochs{e2}"]
            + seeds + der, f"cifar {st} class test, {e2} epochs (compute-matched)")
    log("CIFAR chain done")


# ------------------------------------------- relaxed judge (round 3)
# The strict judge keeps new lessons under-learned on CIFAR (diagonal 50-68% vs 80-95% for
# the baselines). These configs relax it in three ways; "strict" is the original default.
RELAX_CONFIGS = {
    "mnist": {
        "strict": [],
        **{f"any_d{d}": ["--hp", f"judge_max_drop={d}"] for d in (5, 8)},
        **{f"mean_d{d}": ["--hp", "judge_criterion='mean'", "--hp", f"judge_max_drop={d}"] for d in (3, 5, 8)},
        **{f"until{a}": ["--hp", f"judge_until_acc={a}"] for a in (0.8, 0.9, 0.95, 0.98)},
        **{f"trade{l}": ["--hp", f"judge_tradeoff={l}"] for l in (0.25, 0.5, 1.0, 2.0)},
        "mean_d3_trade0.5": ["--hp", "judge_criterion='mean'", "--hp", "judge_tradeoff=0.5"],
    },
    # CIFAR runs are ~6.5 min per seed on a T4, so one representative per family.
    "cifar10": {
        "strict": [],
        "any_d5": ["--hp", "judge_max_drop=5"],
        "any_d8": ["--hp", "judge_max_drop=8"],
        "mean_d5": ["--hp", "judge_criterion='mean'", "--hp", "judge_max_drop=5"],
        "trade0.5": ["--hp", "judge_tradeoff=0.5"],
        "until0.8": ["--hp", "judge_until_acc=0.8"],
    },
}


def relax_chain(dataset: str) -> None:
    if dataset == "mnist":
        base, val_seeds = ["--dataset", "mnist", "--device", "cpu"], ["0", "1", "2"]
    else:  # same lr / epochs as the round-2 CIFAR runs
        base, val_seeds = ["--dataset", "cifar10", "--epochs", "5", "--lr", "0.03"], ["0", "1"]
    log(f"{dataset}: tuning the relaxed judge on val (seeds {' '.join(val_seeds)})")
    best, best_tag = -1.0, None
    for tag, hp in RELAX_CONFIGS[dataset].items():
        r = run(base + ["--strategy", "ours", "--eval-split", "val", "--seeds", *val_seeds,
                        "--out", str(HERE / "results" / "sweep_relax" / dataset), "--tag", tag] + hp,
                f"{dataset} ours val {tag}")
        if acc(r) > best:
            best, best_tag = acc(r), tag
    log(f"{dataset}: relaxed judge chosen = {best_tag} (val ACC {best * 100:.2f})")
    if best_tag == "strict":
        log(f"{dataset}: the strict judge won on val; no new test run needed")
        return
    run(base + ["--strategy", "ours", "--seeds", "0", "1", "2", "--tag", f"relaxed-{best_tag}"]
        + RELAX_CONFIGS[dataset][best_tag], f"{dataset} ours class test, relaxed judge ({best_tag})")
    log(f"{dataset} relax chain done")


# ------------------------------------------- round 4: keep new learning on rollback
# Round 3 showed rollbacks discard 80% of the new-data (wake) training on CIFAR-10, so new
# tasks stall near 65%. rollback_mode="sleep_only" keeps the wake steps and only redoes the
# sleep phase. Also tests the surprise gate (the other suspect). Variants are chosen on val
# (seeds 0 1); the strict default scored 27.75 on the same val split in round 3
# (results/sweep_relax/cifar10/ours_class_strict_val.json), so a variant is only run on test
# if it beats that.
ROUND4_CONFIGS = {
    "no-surprise": ["--hp", "use_surprise=False"],
    "keep-wake": ["--hp", "rollback_mode='sleep_only'"],
    "keep-wake-wr64": ["--hp", "rollback_mode='sleep_only'", "--hp", "wake_replay=64"],
}
ROUND4_STRICT_VAL = 0.2775


def round4_chain() -> None:
    base = ["--dataset", "cifar10", "--epochs", "5", "--lr", "0.03", "--strategy", "ours"]
    log("cifar10 round 4: keep-wake rollback and surprise gate, on val (seeds 0 1)")
    best, best_tag = -1.0, None
    for tag, hp in ROUND4_CONFIGS.items():
        r = run(base + ["--eval-split", "val", "--seeds", "0", "1",
                        "--out", str(HERE / "results" / "sweep_round4" / "cifar10"), "--tag", tag] + hp,
                f"cifar10 ours val {tag}")
        if acc(r) > best:
            best, best_tag = acc(r), tag
    log(f"cifar10 round 4: best variant = {best_tag} (val ACC {best * 100:.2f}; strict was "
        f"{ROUND4_STRICT_VAL * 100:.2f})")
    if best <= ROUND4_STRICT_VAL:
        log("cifar10 round 4: no variant beat the strict judge on val; no test run")
    else:
        run(base + ["--seeds", "0", "1", "2", "--tag", best_tag] + ROUND4_CONFIGS[best_tag],
            f"cifar10 ours class test, {best_tag}")
    log("cifar10 round 4 chain done")


CHAINS = {
    "mnist": mnist_chain,
    "cifar10": cifar_chain,
    "relax_mnist": lambda: relax_chain("mnist"),
    "relax_cifar10": lambda: relax_chain("cifar10"),
    "round4_cifar10": round4_chain,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=sorted(CHAINS),
                   help="chains to run (default: mnist and cifar10, i.e. round 2)")
    args = p.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    only = args.only or ["mnist", "cifar10"]
    log(f"suite start (chains={' '.join(only)})")
    chains = [CHAINS[n] for n in only]
    threads = [threading.Thread(target=c) for c in chains]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("suite done")


if __name__ == "__main__":
    main()
