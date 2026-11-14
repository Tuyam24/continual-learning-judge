"""Our method: surprise-gated wake learning + sleep consolidation + judge with rollback.

Components (each can be switched off for ablations):

  use_surprise  Wake phase. Per-sample loss on each incoming batch is compared with
                an EMA estimate of the loss distribution (mean + surprise_z * std).
                Only samples above it ("surprising") give gradient and are stored in
                the fast buffer. A floor of `min_learn` highest-loss samples per batch
                keeps learning from stalling. Off: learn from / store every sample.
  (always)      Fast buffer: FIFO of the last `fast_size` stored (surprising) samples.
  (always)      Long-term memory (LTM): class-balanced reservoir. Every gated (stored)
                sample is reservoir-inserted into LTM the moment it is pushed to the fast
                buffer, so each first-epoch stream sample that passes the gate is an LTM
                storage candidate exactly once (as in Replay, but only gated samples).
  use_sleep     Every `sleep_every` wake steps and at task end: `sleep_steps` SGD steps
                on batches mixing new samples (fast buffer + current-task LTM) with
                old-task LTM samples. Off: no periodic sleep; wake steps instead append a
                plain replay batch (batch_size samples) drawn from old-task LTM.
  use_judge     After each update (each consolidation point) count correct predictions on
                the stored judge sets of OLD tasks. Criterion (integer counts, no float
                rounding): reject if ANY old task lost more than `judge_max_drop` correct
                judge samples versus the reference. Reference (`judge_ref`): "last_accept"
                = the last accepted checkpoint (ratchets), "task_start" = the state at the
                start of the current task (bounds drift within a task). Reject -> restore
                model + optimizer (deep copies) to the last accepted checkpoint and retry
                the update from memory. With use_sleep the retry is a sleep phase whose
                old-replay share is multiplied by `retry_boost` per retry (and, with
                use_targeted, degrading tasks get weight 1+retry_boost); without
                use_sleep the retry is plain wake-style replay (batch_size new from
                memory + batch_size old, no boost). After `max_retries` the retry result
                is accepted anyway so the new task is still learned. Off: never evaluate,
                never roll back, no judge storage.
  use_targeted  Old-task replay sampling weights w_t proportional to 1 + target_k * drop_t,
                drop_t = latest measured accuracy drop of task t vs its reference
                (judge sets if use_judge, else that task's LTM samples). Off: uniform
                everywhere, including retries.

Memory budget: fast buffer + LTM + stored judge sets <= mem_size at all times
(asserted after every change). LTM capacity is whatever is left, so turning the judge
off hands its share to LTM, keeping every ablation at the same 500-sample budget as
the replay baseline. Model / optimizer checkpoints are parameters, not samples.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .data import TaskData
from .metrics import evaluate
from .strategies import STRATEGIES, Strategy

# Shared CLI hparams meant for other strategies; accepted and ignored.
_FOREIGN_HPARAMS = {"seed", "input_shape", "ewc_lambda", "fisher_samples", "ewc_gamma",
                    "der_alpha", "der_beta"}  # seed goes to the base class


class Ours(Strategy):
    name = "ours"

    def __init__(self, *args, mem_size: int = 500, fast_size: int = 50, judge_per_task: int = 40,
                 use_surprise: bool = True, use_sleep: bool = True, use_judge: bool = True,
                 use_targeted: bool = True,
                 surprise_z: float = -0.5, surprise_ema: float = 0.05, min_learn: int = 16,
                 sleep_every: int = 25, sleep_steps: int = 10, sleep_new: int = 64, sleep_old: int = 96,
                 wake_replay: int = 0,
                 judge_max_drop: int = 3, judge_threshold: float | None = None,
                 judge_ref: str = "task_start",
                 judge_criterion: str = "any", judge_until_acc: float | None = None,
                 judge_tradeoff: float = 0.0, rollback_mode: str = "full",
                 max_retries: int = 2, retry_boost: float = 2.0,
                 target_k: float = 5.0, **kw):
        unknown = set(kw) - _FOREIGN_HPARAMS
        if unknown:
            raise TypeError(f"Ours: unknown hparams {sorted(unknown)}")
        super().__init__(*args, **kw)
        self.mem_size, self.fast_size, self.judge_per_task = mem_size, fast_size, judge_per_task
        self.use_surprise, self.use_sleep = use_surprise, use_sleep
        self.use_judge, self.use_targeted = use_judge, use_targeted
        self.surprise_z, self.surprise_ema, self.min_learn = surprise_z, surprise_ema, min_learn
        self.sleep_every, self.sleep_steps = sleep_every, sleep_steps
        self.sleep_new, self.sleep_old, self.wake_replay = sleep_new, sleep_old, wake_replay
        # The judge compares integer counts. judge_threshold (a fraction), if given, overrides
        # judge_max_drop and is converted to samples with floor + epsilon, so e.g. 0.05 * 40 = 2
        # always means "reject a loss of 3+ samples", independent of float rounding.
        # Defaults (max_drop 3, task_start reference) were picked on val, seeds 0-2;
        # sleep_steps=10 (was 25) was picked on val, seeds 0-4 (same ACC, fewer steps, less forgetting).
        if judge_threshold is not None:
            judge_max_drop = int(math.floor(judge_threshold * judge_per_task + 1e-9))
        if judge_ref not in ("last_accept", "task_start"):
            raise ValueError(f"judge_ref must be last_accept or task_start, got {judge_ref!r}")
        self.judge_max_drop, self.judge_ref = int(judge_max_drop), judge_ref
        if judge_criterion not in ("any", "mean"):
            raise ValueError(f"judge_criterion must be any or mean, got {judge_criterion!r}")
        # Relaxed judge (defaults = original strict judge):
        #   judge_criterion "mean": reject only if the AVERAGE old-task loss exceeds judge_max_drop
        #   judge_until_acc X:      stop judging for the rest of a task once its accuracy reaches X
        #   judge_tradeoff λ > 0:   also accept if mean old-task accuracy lost since the last
        #                           checkpoint <= λ * new-task accuracy gained since it
        self.judge_criterion, self.judge_until_acc = judge_criterion, judge_until_acc
        self.judge_tradeoff = float(judge_tradeoff)
        # rollback_mode "full": a rejected update is undone back to the last accepted checkpoint,
        #   which also discards the wake (new-data) steps since then. On CIFAR-10 this threw away
        #   80% of all new-task training.
        # rollback_mode "sleep_only": checkpoint right after the wake steps, before sleep, so a
        #   rejection only undoes the sleep phase; the new-data learning is kept and the retries
        #   repair old tasks on top of it.
        if rollback_mode not in ("full", "sleep_only"):
            raise ValueError(f"rollback_mode must be full or sleep_only, got {rollback_mode!r}")
        self.rollback_mode = rollback_mode
        self.judge_off = False        # set when judge_until_acc is reached; reset per task
        self.ckpt_correct: dict[int, int] = {}
        self.ckpt_new_acc = 0.0
        self.max_retries, self.retry_boost = max_retries, retry_boost
        self.target_k = target_k
        if fast_size >= mem_size:
            raise ValueError("fast_size must leave room for long-term memory")

        d = self.device
        # fast buffer (FIFO ring)
        self.fx = torch.empty(fast_size, *self.input_shape, device=d)
        self.fy = torch.empty(fast_size, dtype=torch.long, device=d)
        self.f_moved = torch.zeros(fast_size, dtype=torch.bool, device=d)
        self.f_ptr = self.f_count = 0
        # long-term memory: per-class reservoirs {class: (x, y)}, plus stream counts
        self.ltm: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ltm_seen: dict[int, int] = {}
        self.class_task: dict[int, int] = {}
        self._flat = None  # cached (x, y, task) view of LTM
        # judge sets of finished tasks {task_id: (x, y)} and reference accuracies
        self.judge: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ref_correct: dict[int, int] = {}    # judge reference (counts), see judge_ref
        self.start_correct: dict[int, int] = {}  # judge counts at the start of the current task
        self.ltm_ref_acc: dict[int, float] = {}
        self.drop: dict[int, float] = {}
        # surprise statistics
        self.mu = self.var = None
        self.ckpt = None
        self.since_ckpt = {"wake": 0, "sleep": 0, "retry": 0}  # grad steps since last checkpoint
        self.stats = {"wake_steps": 0, "sleep_steps": 0, "retry_steps": 0, "grad_steps": 0,
                      "discarded_wake_steps": 0, "discarded_sleep_steps": 0, "discarded_retry_steps": 0,
                      "consolidations": 0, "judge_evals": 0, "rollbacks": 0, "retries": 0,
                      "forced_accepts": 0, "samples_seen": 0, "samples_learned": 0,
                      "samples_stored_fast": 0, "ltm_candidates": 0, "max_mem_used": 0,
                      "max_judge_drift": 0.0, "tradeoff_accepts": 0, "judge_stopped_tasks": 0,
                      "per_task": []}

    # ---- memory ----------------------------------------------------------
    def mem_used(self) -> int:
        return (self.f_count + sum(len(y) for _, y in self.ltm.values())
                + sum(len(y) for _, y in self.judge.values()))

    def _check_mem(self) -> None:
        used = self.mem_used()
        assert used <= self.mem_size, f"memory budget exceeded: {used} > {self.mem_size}"
        self.stats["max_mem_used"] = max(self.stats["max_mem_used"], used)

    def _ltm_cap_per_class(self) -> int:
        judge_n = sum(len(y) for _, y in self.judge.values())
        cap = self.mem_size - self.fast_size - judge_n
        return cap // max(1, len(self.ltm_seen))

    def _rebalance_ltm(self) -> None:
        cap = self._ltm_cap_per_class()
        for c, (x, y) in self.ltm.items():
            if len(y) > cap:  # reservoir contents are a random sample; keep a random subset
                keep = torch.randperm(len(y), generator=self.gen)[:cap].to(self.device)
                self.ltm[c] = (x[keep], y[keep])
        self._flat = None

    def _fast_push(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Store gated samples: FIFO fast buffer (recent pool for sleep) and, at the same
        time, a reservoir insert into LTM, so no stored sample is overwritten in the fast
        buffer before it has had its chance to enter LTM."""
        for i in range(len(y)):
            self.fx[self.f_ptr], self.fy[self.f_ptr] = x[i], y[i]
            self.f_moved[self.f_ptr] = False
            self._ltm_insert(self.f_ptr)
            self.f_ptr = (self.f_ptr + 1) % self.fast_size
            self.f_count = min(self.f_count + 1, self.fast_size)
        self.stats["samples_stored_fast"] += len(y)
        if len(y):
            self._flat = None
        self._check_mem()

    def _ltm_insert(self, i: int) -> None:
        """Reservoir-insert fast slot i into its class's LTM reservoir (once per sample)."""
        cap = self._ltm_cap_per_class()
        c = int(self.fy[i])
        x, y = self.ltm.get(c, (self.fx[:0], self.fy[:0]))
        self.ltm_seen[c] += 1
        self.stats["ltm_candidates"] += 1
        if len(y) < cap:
            self.ltm[c] = (torch.cat([x, self.fx[i:i + 1]]), torch.cat([y, self.fy[i:i + 1]]))
        else:
            j = int(torch.randint(self.ltm_seen[c], (1,), generator=self.gen))
            if j < cap:
                x[j], y[j] = self.fx[i], self.fy[i]
        self.f_moved[i] = True

    def _ltm_flat(self):
        if self._flat is None:
            if self.ltm:
                xs, ys = zip(*self.ltm.values())
                x, y = torch.cat(xs), torch.cat(ys)
            else:
                x, y = self.fx[:0], self.fy[:0]
            t = torch.tensor([self.class_task[int(c)] for c in y.tolist()], dtype=torch.long,
                             device=self.device)
            self._flat = (x, y, t)
        return self._flat

    def _rand(self, n: int, k: int) -> torch.Tensor:
        return torch.randint(n, (k,), generator=self.gen).to(self.device)

    def _sample_old(self, k: int, cur: int, boost: dict[int, float] | None = None):
        """k samples from LTM of tasks < cur, task chosen by (targeted) weights, uniform inside."""
        x, y, t = self._ltm_flat()
        old = sorted({int(v) for v in t.tolist() if v < cur})
        if k <= 0 or not old:
            return None
        w = torch.ones(len(old))
        if self.use_targeted:
            w = torch.tensor([1.0 + self.target_k * max(0.0, self.drop.get(o, 0.0)) for o in old])
        if boost:
            w = w * torch.tensor([boost.get(o, 1.0) for o in old])
        pick = torch.multinomial(w / w.sum(), k, replacement=True, generator=self.gen)
        out = torch.empty(k, dtype=torch.long)
        for a, o in enumerate(old):
            sel = (pick == a).nonzero().flatten()
            if len(sel):
                pool = (t == o).nonzero().flatten().cpu()
                out[sel] = pool[torch.randint(len(pool), (len(sel),), generator=self.gen)]
        out = out.to(self.device)
        return x[out], y[out]

    def _sample_new(self, k: int, cur: int):
        """k samples from the fast buffer plus the current task's share of LTM."""
        x, y, t = self._ltm_flat()
        m = t == cur
        px = torch.cat([self.fx[:self.f_count], x[m]])
        py = torch.cat([self.fy[:self.f_count], y[m]])
        if k <= 0 or len(py) == 0:
            return None
        i = self._rand(len(py), k)
        return px[i], py[i]

    # ---- training pieces ---------------------------------------------------
    def _step(self, loss: torch.Tensor, kind: str) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.stats[f"{kind}_steps"] += 1
        self.stats["grad_steps"] += 1
        self.since_ckpt[kind] += 1

    def _gate(self, losses: torch.Tensor) -> torch.Tensor:
        """Boolean mask of samples to learn from / store."""
        l = losses.detach()
        if not self.use_surprise:
            return torch.ones_like(l, dtype=torch.bool)
        if self.mu is None:
            self.mu, self.var = l.mean().item(), l.var().item() if len(l) > 1 else 0.0
        thr = self.mu + self.surprise_z * self.var ** 0.5
        sel = l > thr
        k = min(self.min_learn, len(l))
        if int(sel.sum()) < k:
            sel = torch.zeros_like(sel)
            sel[l.topk(k).indices] = True
        a = self.surprise_ema
        self.mu = (1 - a) * self.mu + a * l.mean().item()
        self.var = (1 - a) * self.var + a * (l.var().item() if len(l) > 1 else 0.0)
        return sel

    def _wake_step(self, x, y, task: TaskData) -> None:
        losses = F.cross_entropy(self.masked_logits(self.model(x), y), y, reduction="none")
        sel = self._gate(losses)
        total, n = losses[sel].sum(), int(sel.sum())
        n_rep = self.wake_replay if self.use_sleep else len(y)
        rep = self._sample_old(n_rep, task.task_id)
        if rep is not None:
            rx, ry = rep
            total = total + F.cross_entropy(self.masked_logits(self.model(rx), ry), ry, reduction="sum")
            n += len(ry)
        self._step(total / n, "wake")
        self.stats["samples_seen"] += len(y)
        self.stats["samples_learned"] += int(sel.sum())
        if self.epoch == 0:  # each stream sample is a storage candidate once, as in Replay
            self._fast_push(x[sel], y[sel])

    def _sleep(self, task: TaskData, steps: int, old_mult: float = 1.0,
               boost: dict[int, float] | None = None, kind: str = "sleep",
               n_new: int | None = None, n_old: int | None = None) -> None:
        n_new = self.sleep_new if n_new is None else n_new
        n_old = self.sleep_old if n_old is None else n_old
        for _ in range(steps):
            parts = [self._sample_new(n_new, task.task_id),
                     self._sample_old(int(round(n_old * old_mult)), task.task_id, boost)]
            parts = [p for p in parts if p is not None]
            if not parts:
                return
            x = torch.cat([p[0] for p in parts])
            y = torch.cat([p[1] for p in parts])
            self._step(F.cross_entropy(self.masked_logits(self.model(x), y), y), kind)

    # ---- judge -------------------------------------------------------------
    def _judge_correct(self) -> dict[int, int]:
        """Number of correctly classified judge samples per old task (integer)."""
        return {t: int(round(evaluate(self.model, x, y, self.device, classes=self.seen_tasks[t].classes,
                                      scenario=self.cfg.scenario) * len(y)))
                for t, (x, y) in self.judge.items()}

    def _ltm_accs(self, cur: int) -> dict[int, float]:
        x, y, t = self._ltm_flat()
        out = {}
        for o in sorted({int(v) for v in t.tolist() if v < cur}):
            m = t == o
            out[o] = evaluate(self.model, x[m], y[m], self.device, classes=self.seen_tasks[o].classes,
                              scenario=self.cfg.scenario)
        return out

    def _new_acc(self, task: TaskData) -> float:
        """Accuracy on the current task's samples held in memory (fast buffer + its LTM share).
        These are training samples, so it is an optimistic estimate; it is only used as a
        relative signal (did the update help the new task?)."""
        x, y, t = self._ltm_flat()
        m = t == task.task_id
        px = torch.cat([self.fx[:self.f_count], x[m]])
        py = torch.cat([self.fy[:self.f_count], y[m]])
        if len(py) == 0:
            return 0.0
        return evaluate(self.model, px, py, self.device, classes=task.classes, scenario=self.cfg.scenario)

    def _relaxed(self) -> bool:
        return self.judge_until_acc is not None or self.judge_tradeoff > 0

    def _save_ckpt(self) -> None:
        self.ckpt = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()))
        self.since_ckpt = dict.fromkeys(self.since_ckpt, 0)

    def _restore_ckpt(self) -> None:
        # model.load_state_dict copies into the parameters, but Optimizer.load_state_dict puts
        # the checkpoint's own momentum tensors into the live optimizer, which then mutates them
        # in place. Load a deep copy so the checkpoint stays clean across repeated rollbacks.
        self.model.load_state_dict(self.ckpt[0])
        self.optimizer.load_state_dict(copy.deepcopy(self.ckpt[1]))
        for k, v in self.since_ckpt.items():
            self.stats[f"discarded_{k}_steps"] += v
        self.since_ckpt = dict.fromkeys(self.since_ckpt, 0)

    def _consolidate(self, task: TaskData) -> None:
        """One consolidation point: transfer, sleep, judge (with rollback / retry)."""
        if self.rollback_mode == "sleep_only" and self.use_judge and self.judge and not self.judge_off:
            self._save_ckpt()  # keep the wake steps: a rollback only undoes what follows
        if self.use_sleep:
            self._sleep(task, self.sleep_steps)
        self.stats["consolidations"] += 1
        pt = self.stats["per_task"][-1]

        if not self.use_judge:
            if self.use_targeted:
                accs = self._ltm_accs(task.task_id)
                self.drop = {t: self.ltm_ref_acc.get(t, a) - a for t, a in accs.items()}
            return
        if not self.judge:
            self._save_ckpt()
            return
        n_j = {t: len(y) for t, (_, y) in self.judge.items()}
        new_acc = self._new_acc(task) if self._relaxed() else 0.0
        if self.judge_off or (self.judge_until_acc is not None and new_acc >= self.judge_until_acc):
            # new lesson is learned well enough: stop judging (no rollback) for the rest of it
            if not self.judge_off:
                self.judge_off = True
                self.stats["judge_stopped_tasks"] += 1
            correct = self._judge_correct()  # still measured, for targeted replay weights
            self.drop = {t: (self.ref_correct[t] - c) / n_j[t] for t, c in correct.items()}
            return
        for attempt in range(self.max_retries + 1):
            correct = self._judge_correct()
            self.stats["judge_evals"] += 1
            lost = {t: self.ref_correct[t] - c for t, c in correct.items()}
            self.drop = {t: d / n_j[t] for t, d in lost.items()}
            worst = max(lost.values()) if self.judge_criterion == "any" else sum(lost.values()) / len(lost)
            if worst <= self.judge_max_drop:
                break
            if self.judge_tradeoff > 0:
                new_acc = self._new_acc(task)
                old_cost = sum((self.ckpt_correct.get(t, c) - c) / n_j[t] for t, c in correct.items()) / len(correct)
                if old_cost <= self.judge_tradeoff * (new_acc - self.ckpt_new_acc):
                    self.stats["tradeoff_accepts"] += 1
                    break
            if attempt == self.max_retries:
                self.stats["forced_accepts"] += 1
                pt["forced_accepts"] += 1
                break
            # reject: roll back and retry from memory with more replay of degrading tasks
            self._restore_ckpt()
            self.stats["rollbacks"] += 1
            self.stats["retries"] += 1
            pt["rollbacks"] += 1
            if self.use_sleep:
                boost = None
                if self.use_targeted:
                    boost = {t: (1.0 + self.retry_boost) if d > self.judge_max_drop else 1.0
                             for t, d in lost.items()}
                self._sleep(task, self.sleep_steps, old_mult=self.retry_boost ** (attempt + 1),
                            boost=boost, kind="retry")
            else:  # no sleep: retry with plain wake-style replay from memory, no boost
                bs = self.cfg.batch_size
                self._sleep(task, self.sleep_steps, n_new=bs, n_old=bs, kind="retry")
        # accept: new checkpoint; with judge_ref="last_accept" the reference ratchets
        if self.judge_ref == "last_accept":
            self.ref_correct.update(correct)
        drift = max((self.start_correct[t] - c) / n_j[t] for t, c in correct.items())
        pt["max_judge_drift"] = max(pt["max_judge_drift"], drift)
        self.stats["max_judge_drift"] = max(self.stats["max_judge_drift"], drift)
        self.ckpt_correct = dict(correct)
        if self.judge_tradeoff > 0:
            self.ckpt_new_acc = self._new_acc(task)
        self._save_ckpt()

    # ---- task loop ---------------------------------------------------------
    def before_task(self, task: TaskData) -> None:
        super().before_task(task)
        for c in task.classes:
            self.class_task[c] = task.task_id
            self.ltm_seen.setdefault(c, 0)
        self._rebalance_ltm()
        self._check_mem()
        self.stats["per_task"].append({"task": task.task_id, "rollbacks": 0, "forced_accepts": 0,
                                       "max_judge_drift": 0.0})
        self.judge_off = False
        self.ckpt_correct = dict(self.start_correct)
        self.ckpt_new_acc = 0.0
        if self.use_judge:
            self._save_ckpt()  # fresh optimizer -> checkpoint must hold its (empty) state

    def after_task(self, task: TaskData) -> None:
        self._consolidate(task)
        self.f_count = self.f_ptr = 0  # everything is in LTM now; fast buffer is per-stream
        self.f_moved.zero_()
        if self.use_judge and self.judge_per_task > 0:
            if task.y_judge is None or len(task.y_judge) < self.judge_per_task:
                raise ValueError("task has fewer judge samples than judge_per_task")
            self.judge[task.task_id] = (task.x_judge[:self.judge_per_task].to(self.device),
                                        task.y_judge[:self.judge_per_task].to(self.device))
            self._rebalance_ltm()  # make room before the budget check
        self._check_mem()

    def train_task(self, task: TaskData) -> None:
        self.before_task(task)
        self.model.train()
        pt = self.stats["per_task"][-1]
        g0 = self.stats["grad_steps"]
        # the task becomes visible in seen_tasks only at the end, but judge/LTM helpers
        # index seen_tasks by task id, which is fine because they only touch old tasks.
        n, wake = len(task.y_train), 0
        for self.epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                self._wake_step(task.x_train[idx].to(self.device), task.y_train[idx].to(self.device), task)
                wake += 1
                if wake % self.sleep_every == 0:
                    self._consolidate(task)
        self.after_task(task)
        self.seen_tasks.append(task)
        if self.use_judge:  # references for the next task: counts at the accepted end state
            self.start_correct = self._judge_correct()  # (checkpoint is taken in before_task)
            self.ref_correct = dict(self.start_correct)
        self.ltm_ref_acc = self._ltm_accs(task.task_id + 1)
        self.drop = {}
        pt["grad_steps"] = self.stats["grad_steps"] - g0
        pt["mem_used_end"] = self.mem_used()


STRATEGIES["ours"] = Ours
