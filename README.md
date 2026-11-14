# Catastrophic forgetting on Split MNIST and Split CIFAR-10

Small continual-learning testbed. A 10-class dataset is split into 5 tasks of two
classes (MNIST digits 0-1, 2-3, ...; CIFAR-10 airplane/car, bird/cat, ...) learned in
sequence; after each task we record test accuracy on every task seen so far. MNIST runs
on a laptop CPU in seconds; CIFAR-10 needs a GPU (Apple MPS, or Kaggle/Colab via
`kaggle/`).

**Where it stands:** across both datasets, our method forgets the least, but DER++ is
the better learner overall: roughly tied on MNIST, about 10 points ahead on CIFAR-10.
See [Round 2](#round-2-der-compute-matched-baselines-split-cifar-10) and
[Round 3](#round-3-relaxing-the-judge). A plain-language walkthrough of the whole project
is in `docs/the_forgetting_experiment.html`.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python run_experiment.py --strategy naive --seeds 0 1 2
.venv/bin/python run_experiment.py --strategy naive --scenario task --seeds 0 1 2
```

Results go to `results/<strategy>_<scenario>.json` (CIFAR-10: `results/cifar10/`).

```bash
.venv/bin/python run_experiment.py --dataset cifar10 --strategy derpp --seeds 0   # needs a GPU
.venv/bin/python run_suite.py --only relax_mnist   # a whole experiment round, unattended
.venv/bin/python make_report.py --results results/cifar10
```

## Scenarios

- `class` (default): class-incremental. One 10-way head, no task ID at test time.
  This is the hard setting, where forgetting is most visible.
- `task`: task-incremental. Logits are masked to the task's own 2 classes; much easier.

## Layout

| File | Role |
|---|---|
| `cl/data.py` | Builds the 5 tasks (MNIST or CIFAR-10), each with train / judge / val / test splits |
| `cl/model.py` | 2x400 MLP (MNIST) and reduced ResNet-18 (CIFAR-10), shared 10-way head |
| `cl/metrics.py` | Evaluation, accuracy matrix, average accuracy, forgetting |
| `cl/strategies.py` | `Strategy` training loop with hooks; `Naive`, `Replay`, `EWC`, `DERPP` |
| `cl/ours.py` | Our method (surprise gate, sleep, judge with rollback, targeted replay) |
| `run_experiment.py` | CLI: runs one strategy over seeds, prints and saves results |
| `run_suite.py` | Runs a whole experiment round unattended (`--only mnist cifar10 relax_mnist relax_cifar10`) |
| `make_report.py` | Builds `summary.md` and the two figures from a results folder |
| `kaggle/make_notebook.py` | Builds self-contained Kaggle/Colab notebooks for the CIFAR-10 rounds |
| `docs/the_forgetting_experiment.html` | Plain-language course on the whole project |

## Baselines (2 epochs/task, SGD lr 0.01 momentum 0.9, test set, 3 seeds)

| Method | class-IL ACC | class-IL forgetting | task-IL ACC | task-IL forgetting |
|---|---|---|---|---|
| Naive fine-tuning | 19.64 ± 0.04% | 99.23 ± 0.09% | 98.29 ± 0.26% | 1.51 ± 0.28% |
| Replay (500-sample reservoir) | 84.76 ± 1.61% | 17.36 ± 1.99% | 98.85 ± 0.07% | 0.55 ± 0.02% |
| Online EWC (λ = 1e4) | 22.80 ± 1.44% | 95.39 ± 1.84% | 99.16 ± 0.07% | 0.07 ± 0.04% |

- **Replay** stores a uniform random sample of the stream (reservoir sampling, 500
  samples total = ~1% of MNIST) and mixes an equal-size replay batch into every step.
- **EWC** penalises moving weights that had high Fisher information on earlier tasks.
  λ was picked on the **val** split (seed 0, sweep 1e2–1e6), not on test:
  `--strategy ewc --ewc-lambda L --eval-split val`. Each weight's penalty stiffness
  λ·F is capped at 1/lr; without the cap, λ ≥ 1e4 makes SGD diverge to NaN.
- EWC failing in class-IL is the expected result (van de Ven et al. 2019): it keeps
  old *weights* in place but nothing stops the shared head from always predicting
  the newest classes. Replay fixes exactly that, because old classes keep showing up
  as negatives and positives in the softmax.

## Our method (`--strategy ours`, `cl/ours.py`)

A replay method with four extra pieces around the wake/learn loop. Every piece can be
switched off with `--hp use_<piece>=False` for ablations.

| Component | Flag | What it does |
|---|---|---|
| Surprise gate | `use_surprise` | Each incoming batch's per-sample losses are compared with an EMA estimate of the loss distribution (threshold mean + `surprise_z`·std, `surprise_z=-0.5`, `surprise_ema=0.05`). Only samples above it give gradient and get stored; at least `min_learn=16` highest-loss samples per batch are always kept. Off: learn from and store every sample. |
| Fast buffer + long-term memory (LTM) | always | Gated samples go into a FIFO fast buffer (`fast_size=50`) and are reservoir-inserted into a **class-balanced** LTM at the same moment (first epoch only, once per sample, as in Replay). |
| Sleep consolidation | `use_sleep` | Every `sleep_every=25` wake steps and at task end: `sleep_steps=10` SGD steps on batches of `sleep_new=64` new samples (fast buffer + current-task LTM) plus `sleep_old=96` old-task LTM samples. Off: each wake step instead gets a plain replay batch of 64 old-task LTM samples. |
| Judge with rollback | `use_judge` | After each consolidation, count correct predictions on small stored judge sets (40 held-out samples per finished task). If any old task lost more than `judge_max_drop=3` of 40 versus its count at the start of the current task (`judge_ref=task_start`), restore model **and** optimizer to the last accepted checkpoint and retry the update from memory: a sleep phase whose old-replay share is multiplied by `retry_boost=2.0` per retry. After `max_retries=2` the retry is accepted anyway (a "forced accept") so the new task is still learned. |
| Targeted replay | `use_targeted` | Old-task sampling weight ∝ 1 + `target_k`·drop_t (`target_k=5`), drop_t = latest measured accuracy drop of task t; on a retry, tasks that broke the judge get weight 1 + `retry_boost`. Off: uniform over old tasks everywhere. |

**Memory accounting.** Fast buffer + LTM + stored judge sets ≤ `mem_size` = 500 samples at
all times, asserted after every change (`stats.max_mem_used` is 500 in every run). That is
the same budget as the replay baseline: by the end, 160 slots are judge samples (4 × 40),
50 are the fast buffer slot reservation, and the remaining 290 are LTM (29 per class).
With `use_judge=False` the judge's share goes back to LTM, so every ablation uses 500
samples too. Model/optimizer checkpoints for rollback are parameters, not samples, and are
not counted. The judge set is the first 40 samples of each task's held-out slice
(`cl/data.py`); val is the rest of that slice, so judge data never overlaps with val or test,
and train/test are unchanged from the baselines.

**How the hyperparameters were chosen.** Only on the **val** split, never on test. The
defaults were first tuned on val seeds 0-2; a later sweep (36 configs × 2 seeds, plus
confirmation runs on seeds 0-4) found nothing clearly better. Per-config std was 0.4-1.4
points, larger than most differences between the top configs. `sleep_steps` was lowered
from 25 to 10: same val class-IL ACC (5-seed mean 88.08 vs 88.04), ~35% fewer gradient
steps, lower forgetting and lower seed variance. On val (seeds 0-4, same 500-sample budget)
ours scored 88.08 vs replay 85.06. Robust findings from the sweep: retries are essential
(`max_retries=0` falls to replay level, 83.6); `judge_max_drop=3` beat 1, 2, 4 and 5;
`target_k=0` hurt (86.9); `sleep_every=50` with `sleep_steps=10` was unstable (one seed
79.1). Caveat: val seeds 0-2 were already used to set the first defaults, so only val
seeds 3-4 were fully fresh. The test numbers below were run once, after all choices were
fixed. Sweep files and logs: `results/sweep_ours/`.

**CLI**

```bash
# final numbers (test split, defaults = tuned config)
.venv/bin/python run_experiment.py --strategy ours --seeds 0 1 2
.venv/bin/python run_experiment.py --strategy ours --scenario task --seeds 0 1 2
# ablations (one flag off each)
.venv/bin/python run_experiment.py --strategy ours --seeds 0 1 2 --hp use_judge=False --tag no-judge
#   ... use_surprise / use_sleep / use_targeted with --tag no-surprise / no-sleep / no-targeted
# tuning: always on val
.venv/bin/python run_experiment.py --strategy ours --seeds 0 1 --eval-split val --hp sleep_steps=15 --tag ss15
# tables and figures from results/*.json (val files and subfolders are skipped)
.venv/bin/python make_report.py
```

Any hparam in `Ours.__init__` can be set with `--hp key=value`. Each seed's `stats` in the
output JSON records `grad_steps` (wake + sleep + retry), rollbacks, forced accepts,
`discarded_*_steps` (steps thrown away by rollbacks) and `max_judge_drift` (largest
within-task judge drop vs task start, which forced accepts can push past the threshold).

Ablation semantics: `use_sleep=False` removes periodic sleep; judge retries then use plain
wake-style replay (64 new-from-memory + 64 old, no boost). `use_targeted=False` is uniform
everywhere, retries included.

## Results (test split, 3 seeds: 0 1 2)

Produced by `make_report.py` (`results/summary.md`, `results/avg_acc.png`,
`results/forgetting.png`). Mean ± std over seeds; ACC = final average accuracy over all 5
tasks; forgetting = mean drop from best to final accuracy over the first 4 tasks. All
methods: 2 epochs/task, batch 64, SGD lr 0.01 momentum 0.9, replay-type methods 500
samples. "grad steps" counts optimizer steps per run (baselines: one per batch).

### Class-incremental (primary)

| Method | File | ACC | Forgetting | Seeds | grad steps | rollbacks |
|---|---|---:|---:|---:|---:|---:|
| Naive fine-tuning | `naive_class` | 19.64 ± 0.04% | 99.23 ± 0.09% | 3 | 1,694 ± 0 | – |
| Online EWC | `ewc_class` | 22.80 ± 1.44% | 95.39 ± 1.84% | 3 | 1,694 ± 0 | – |
| Replay | `replay_class` | 84.76 ± 1.61% | 17.36 ± 1.99% | 3 | 1,694 ± 0 | – |
| Ours | `ours_class` | 89.05 ± 0.44% | 7.08 ± 0.82% | 3 | 2,824 ± 42 | 43.0 ± 4.2 |

### Task-incremental

| Method | File | ACC | Forgetting | Seeds | grad steps | rollbacks |
|---|---|---:|---:|---:|---:|---:|
| Naive fine-tuning | `naive_task` | 98.29 ± 0.26% | 1.51 ± 0.28% | 3 | 1,694 ± 0 | – |
| Online EWC | `ewc_task` | 99.16 ± 0.07% | 0.07 ± 0.04% | 3 | 1,694 ± 0 | – |
| Replay | `replay_task` | 98.85 ± 0.07% | 0.55 ± 0.02% | 3 | 1,694 ± 0 | – |
| Ours | `ours_task` | 99.01 ± 0.09% | 0.74 ± 0.11% | 3 | 2,394 ± 0 | 0.0 ± 0.0 |

### Ablations (class-IL, test, seeds 0 1 2)

| Variant | ACC | Forgetting | grad steps | rollbacks | forced accepts |
|---|---:|---:|---:|---:|---:|
| Ours (full) | 89.05 ± 0.44% | 7.08 ± 0.82% | 2,824 ± 42 | 43.0 ± 4.2 | 4.3 |
| no surprise gate (`use_surprise=False`) | 87.65 ± 1.95% | 5.25 ± 1.79% | 2,764 ± 67 | 37.0 ± 6.7 | 0.3 |
| no sleep (`use_sleep=False`) | 86.09 ± 0.30% | 13.64 ± 0.25% | 2,467 ± 85 | 77.3 ± 8.5 | 35.7 |
| no judge (`use_judge=False`) | 87.27 ± 0.60% | 14.45 ± 0.63% | 2,394 ± 0 | 0 | 0 |
| no targeted replay (`use_targeted=False`) | 88.44 ± 1.18% | 7.95 ± 2.18% | 2,744 ± 22 | 35.0 ± 2.2 | 3.0 |

Per-seed ACC, full method: 88.43 / 89.45 / 89.26; no-surprise: 89.61 / 88.35 / 84.98;
no-targeted: 88.54 / 86.96 / 89.84.

### Interpretation

In class-IL, ours reaches 89.05 ± 0.44% against replay's 84.76 ± 1.61% at the same
500-sample budget, with less than half the forgetting (7.1 vs 17.4 points) and lower seed
variance, consistent with the +3 points seen on val; with only 3 seeds the gap is clear but
its size is not precisely known. The ablations say the judge-with-rollback and sleep do the
real work (removing either costs 1.8-3.0 points and doubles forgetting), while the surprise
gate and targeted replay give at most ~0.6-1.4 points, which is within seed noise (removing
the gate even lowers forgetting); note also that the no-judge run uses ~430 fewer gradient
steps, so it is not a compute-matched comparison. The gains are not free: a run takes 2,824
gradient steps vs 1,694 for every baseline (1.67×, and sleep batches are 160 samples vs
replay's 128), plus judge evaluations; 43 rollbacks per run throw away ~1,270 of those steps
(45%), and ~4 consolidations per run end in a forced accept. In task-IL every method except
naive sits at ~99% and ours (99.01%) is no better than EWC (99.16%) or replay (98.85%), so the
method only matters in the class-IL setting.

## Round 2: DER++, compute-matched baselines, Split CIFAR-10

Added after checking prior work: published simple replay scores ~87% and DER++ ~92% on
class-IL Split MNIST with 500 samples, so beating our own replay baseline was not enough.

- **DER++** (`--strategy derpp`, Buzzega et al. 2020): reservoir replay that also stores the
  logits each sample had when seen, and adds `der_alpha * MSE(logits)` + `der_beta * CE(labels)`
  on two buffer batches. Same 500-sample budget.
- **Compute-matched baselines:** ours uses more gradient steps (1.7x on MNIST, 2.05x on
  CIFAR-10), so replay and DER++ were also run with more epochs, giving them *more* steps than ours.
- **Split CIFAR-10:** reduced ResNet-18 (nf=20, ~1.1M params), 5 epochs/task (the literature
  uses 50; too slow for a laptop), no augmentation, class-IL only. Absolute numbers are
  therefore lower than published ones; only the comparison between methods means anything.
  Run on a Kaggle T4 with `kaggle/cl_cifar10_suite.ipynb`.

All hyperparameters were chosen on val: DER++ `alpha=0.2, beta=1.0` on MNIST (seeds 0-1,
8 configs); CIFAR-10 `lr=0.03` (with replay) and DER++ `alpha=0.3, beta=0.5`. Our method
kept its MNIST-tuned settings on CIFAR-10, which tilts the CIFAR comparison towards DER++.

### Split MNIST, class-IL (test, seeds 0 1 2)

| Method | ACC | Forgetting | grad steps |
|---|---:|---:|---:|
| Replay | 84.76 ± 1.61% | 17.36% | 1,694 |
| Replay, 4 epochs (compute-matched) | 85.43 ± 0.95% | 17.24% | 3,388 |
| Ours | 89.05 ± 0.44% | **7.08%** | 2,824 |
| **DER++** | **89.90 ± 0.81%** | 11.40% | 1,694 |
| DER++, 4 epochs (compute-matched) | 89.64 ± 0.52% | 12.14% | 3,388 |

DER++ task-IL: 99.49 ± 0.05%.

### Split CIFAR-10, class-IL (test, seeds 0 1 2; `results/cifar10/`)

| Method | ACC | Forgetting | grad steps | per-seed ACC |
|---|---:|---:|---:|---|
| Naive | 18.69 ± 0.26% | 86.89% | 3,525 | 18.3 / 19.0 / 18.7 |
| Online EWC | 16.30 ± 0.52% | 76.55% | 3,525 | 16.7 / 16.6 / 15.6 |
| Replay | 25.56 ± 2.11% | 78.75% | 3,525 | 23.6 / 28.5 / 24.6 |
| Replay, 11 epochs (compute-matched) | 27.94 ± 0.29% | 79.18% | 7,755 | 28.0 / 28.3 / 27.6 |
| Ours | 28.56 ± 1.69% | **45.36%** | 7,235 | 30.2 / 26.2 / 29.2 |
| Ours, no judge | 24.54 ± 9.72% | 59.05% | 4,975 | 18.3 / 38.3 / 17.0 |
| **DER++** | **38.64 ± 2.03%** | 58.24% | 3,525 | 37.0 / 41.5 / 37.4 |
| DER++, 11 epochs (compute-matched) | 43.04 ± 1.05% | 54.21% | 7,755 | 42.8 / 44.4 / 41.9 |

**Reading it.** DER++ wins clearly on CIFAR-10. Ours has the lowest forgetting, but mostly
because it never learns the new tasks well: accuracy on each task right after learning it
(the accuracy-matrix diagonal) is 92 / 62 / 50 / 54 / 68% for ours against 96 / 70 / 79 /
88 / 92% for DER++. Forgetting is measured from a task's best accuracy, so a task that only
reached 50% has less to lose. The judge keeps rejecting updates that hurt old tasks, which
also blocks learning the new one (the stability-plasticity trade-off, tipped too far
towards stability). Without the judge, ours collapses to naive level on 2 of 3 seeds, so the
judge's job here is preventing collapse.

## Round 3: relaxing the judge

Three new `ours` switches, all off by default (defaults reproduce every earlier result
bit for bit):

| Hparam | Effect |
|---|---|
| `judge_criterion='mean'` | reject only if the *average* old-task loss exceeds `judge_max_drop` (default `'any'`: any single task) |
| `judge_max_drop=N` | allowed loss in correct judge samples out of 40 (default 3) |
| `judge_until_acc=X` | stop judging for the rest of a task once its accuracy (on its samples in memory) reaches X |
| `judge_tradeoff=λ` | also accept an update if mean old-task accuracy lost since the last checkpoint <= λ × new-task accuracy gained |

### Split MNIST: tuning on val (seeds 0 1 2) and test

"New-task" is the mean of the accuracy-matrix diagonal: how well each task is learned
right after training on it.

| Config | val ACC | val forgetting | new-task | rollbacks |
|---|---:|---:|---:|---:|
| `judge_max_drop=5` (chosen) | 88.82 ± 0.69% | 9.85% | 96.70% | 27.3 |
| strict (original) | 88.42 ± 0.57% | 7.87% | 94.56% | 43.0 |
| `judge_tradeoff=0.25` | 88.16 ± 0.31% | 10.95% | 96.93% | 31.0 |
| `judge_criterion='mean'` | 87.79 ± 1.69% | 11.75% | 97.19% | 21.3 |
| `judge_tradeoff=0.5` | 86.89 ± 1.53% | 12.65% | 96.91% | 23.7 |
| `judge_max_drop=8` | 85.52 ± 0.47% | 16.49% | 98.71% | 17.0 |
| `judge_until_acc=0.98` | 85.34 ± 1.42% | 13.64% | 96.25% | 26.7 |
| `judge_until_acc=0.9` | 83.81 ± 0.91% | 18.63% | 98.71% | 15.3 |
| `judge_criterion='mean' judge_max_drop=8` | 82.72 ± 0.75% | 20.19% | 98.87% | 6.0 |

Full table: `results/sweep_relax/mnist/`. **Test (seeds 0 1 2), `judge_max_drop=5`: 89.57 ±
0.73% ACC, 9.20% forgetting** (strict: 89.05 / 7.08; DER++: 89.90 / 11.40). On MNIST,
relaxing does what it should for new tasks (94.6% → up to 98.9%) but forgetting rises
faster, so final accuracy barely moves; the three methods are tied within noise. Stopping the
judge once the new task is learned never helps: every threshold tried (0.8-0.98) scores
below the strict judge (83.8-85.3% vs 88.4%), because it lets forgetting through for the
rest of the task.

### Split CIFAR-10: relaxing does not help

Run on a Kaggle T4 with `kaggle/cl_relax_cifar10.ipynb` (`results/sweep_relax/cifar10/`,
`results/cifar10/ours_class_relaxed-any_d8.json`). Same lr / epochs as round 2.

| Config (val, seeds 0 1) | val ACC | val forgetting | new-task | rollbacks | forced accepts |
|---|---:|---:|---:|---:|---:|
| `judge_max_drop=8` (chosen) | 28.21 ± 2.38% | 43.79% | 63.2% | 195 | 69 |
| `judge_criterion='mean' judge_max_drop=5` | 27.96 ± 1.88% | 48.02% | 66.4% | 199 | 72 |
| strict (original) | 27.75 ± 0.63% | 44.06% | 63.0% | 228 | 107 |
| `judge_max_drop=5` | 27.72 ± 1.70% | 43.10% | 62.2% | 226 | 105 |
| `judge_until_acc=0.8` | 25.17 ± 2.90% | 41.43% | 58.3% | 223 | 105 |
| `judge_tradeoff=0.5` | 23.21 ± 6.71% | 53.26% | 65.8% | 158 | 37 |

**Test (seeds 0 1 2), `judge_max_drop=8`: 27.90 ± 2.39% ACC, 47.80% forgetting**, against
28.56 ± 1.69% for the strict judge and 38.64% for DER++. The top four configs are within
0.5 points of each other on val, well inside seed noise, and new-task accuracy barely moves
(64.8% → 66.1% on test, vs 85-89% for DER++ and replay).

### Why: rollbacks discard the new-task learning

On CIFAR-10 the judge rejects almost every update: ~226 rollbacks and ~107 forced accepts
per run (MNIST: 43 and 4). A rollback restores the last checkpoint, which **also throws away
the wake steps on new data** taken since then; the retry then trains from memory only (the
50-slot fast buffer plus ~90 stored samples per class). Measured from the run stats:

| Run | wake steps | wake steps discarded by rollbacks |
|---|---:|---:|
| MNIST, strict | 1,694 | 844 (50%) |
| CIFAR-10, strict | 3,525 | 2,820 (80%) |
| CIFAR-10, `judge_max_drop=8` | 3,525 | 2,820 (80%) |

So on CIFAR-10 four fifths of the new-task training is thrown away and each new task is
learned mostly from a few hundred stored samples, which is why its accuracy stalls near 65%.
Relaxing the threshold cannot fix this: in class-IL, learning new classes always costs old
classes some accuracy, and on CIFAR-10 that cost exceeds even 8 of 40 judge samples. A 40-sample
judge set is also noisy: at 30-50% accuracy it swings by about ±3 correct answers (1 std)
from sampling alone, the same size as the default threshold.

**Implication for the design:** the judge should not undo the learning on new data. A
rollback that restores only the sleep phase (checkpoint after wake, before sleep), or a retry
that also trains on new data, would keep the new-task learning while still guarding old
tasks. Larger judge sets would also reduce false rejections, at a memory cost.

## Round 4: keep new learning on rollback

New `ours` switch `rollback_mode` (default `"full"` reproduces every earlier result).
`"sleep_only"` checkpoints right after the wake steps, before sleep, so a judge rejection only
undoes the sleep phase; the new-data learning is kept and the retries repair old tasks on top
of it.

### Split MNIST (val, seeds 0 1 2; `results/sweep_round4/mnist/`)

| Config | val ACC | forgetting | new-task | wake steps discarded | rollbacks |
|---|---:|---:|---:|---:|---:|
| strict (`rollback_mode='full'`) | 88.42 ± 0.57% | 7.87% | 94.6% | 49.8% | 43 |
| `use_surprise=False` | 86.98 ± 2.30% | 5.88% | 91.7% | 48.8% | 37 |
| `wake_replay=64` (full rollback) | 86.91 ± 0.16% | 11.69% | 96.3% | 52.8% | 61 |
| `rollback_mode='sleep_only' wake_replay=32` | 83.48 ± 0.15% | 19.11% | 98.8% | 0% | 74 |
| `rollback_mode='sleep_only' wake_replay=64 max_retries=4` | 83.01 ± 1.76% | 19.53% | 98.6% | 0% | 113 |
| `rollback_mode='sleep_only'` | 82.64 ± 1.01% | 20.03% | 98.7% | 0% | 80 |
| `rollback_mode='sleep_only' use_surprise=False` | 81.46 ± 1.73% | 20.96% | 98.2% | 0% | 77 |
| `rollback_mode='sleep_only' wake_replay=64` | 81.09 ± 0.59% | 22.01% | 98.7% | 0% | 64 |

The fix works mechanically (no new-data training is discarded, new-task accuracy ~98.7%),
but on MNIST it costs ~6 points: forgetting rises to ~20%, replay-baseline level. So on MNIST
the method's advantage over replay *was* the full rollback: discarding new learning whenever
old tasks slipped. Without it the method is replay with a smaller long-term memory (the judge
sets take 200 of the 500 slots). On MNIST new tasks are easy, so that trade paid off; on
CIFAR-10 it is the bottleneck. Mixing old samples into the wake steps (`wake_replay`) does not
change the picture.

### Split CIFAR-10: no variant beats the strict judge

Run on a Kaggle T4 with `kaggle/cl_round4_cifar10.ipynb`; val split, seeds 0 1. The strict
judge's val score from round 3 (27.75%) was the bar for a test run.

| Variant | val ACC | val forgetting |
|---|---:|---:|
| strict (round 3 reference) | 27.75 ± 0.63% | 44.06% |
| `use_surprise=False` | 25.64% | not recorded |
| `rollback_mode='sleep_only'` | 19.74 ± 5.74% | 26.42% |
| `rollback_mode='sleep_only' wake_replay=64` | not recorded | 75.92% |

Nothing beat the strict judge, so no test run was made. The per-run result files were lost:
the notebook's report cell crashed when there were no test results, which stopped the Kaggle
commit before the zip was saved. These numbers come from the run's log as relayed from the
Kaggle page; "not recorded" values were not in that relay. The notebooks now save the zip
before the report and skip the report when there is nothing to show.

**Reading it.** Keeping the new-data learning does lower forgetting on CIFAR-10, but accuracy
collapses to naive level (18.7%) with large seed variance, and adding wake replay makes
forgetting far worse. Removing the surprise gate costs ~2 points. Together with round 3: the
method's full rollback is both what protects old tasks and what blocks new ones, and neither
loosening it nor removing it fixes CIFAR-10. DER++ (38.6% test, 43.0% compute-matched) remains
clearly ahead.
