"""Continual-learning strategies.

`Strategy` owns the model and the training loop over one task. Subclasses
change behaviour through hooks rather than rewriting the loop:

  before_task(task)            e.g. build a fresh optimizer
  make_batch(x, y, task)       e.g. append replayed samples to the batch
  extra_loss(x, y, task)       e.g. an EWC penalty
  after_step(x, y, task)       e.g. update a memory buffer (gets the *un-augmented* batch)
  after_task(task)             e.g. compute Fisher information

Implemented:
  Naive   plain sequential fine-tuning (forgetting lower bound)
  Replay  experience replay from a fixed-size reservoir-sampled buffer
  EWC     online Elastic Weight Consolidation (diagonal Fisher penalty)
  DERPP   DER++: replay of stored labels *and* stored logits (Buzzega et al. 2020)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .data import N_CLASSES, TaskData


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    scenario: str = "class"  # "class" or "task" (see cl.metrics.active_logits)


class Strategy:
    name = "base"

    def __init__(self, model: torch.nn.Module, cfg: TrainConfig, device: torch.device, seed: int = 0,
                 input_shape: tuple[int, ...] = (784,), **hparams):
        self.model = model.to(device)
        self.input_shape = tuple(input_shape)  # shape of one sample, for memory buffers
        self.cfg = cfg
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)  # shuffling/sampling, independent of model init
        self.optimizer: torch.optim.Optimizer | None = None
        self.seen_tasks: list[TaskData] = []
        # same_task[c] = bool mask over the classes in c's task (task-IL masking per sample).
        self.same_task = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.bool, device=device)
        self.stats: dict = {"grad_steps": 0}  # saved per seed by run_experiment.py

    # ---- helpers --------------------------------------------------------
    def masked_logits(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Task-IL: each sample only competes among its own task's classes.
        Class-IL: unchanged. Per-sample (not per-batch) so replayed old samples are masked correctly."""
        if self.cfg.scenario == "class":
            return logits
        return logits.masked_fill(~self.same_task[y], float("-inf"))

    def task_loss(self, x: torch.Tensor, y: torch.Tensor, task: TaskData) -> torch.Tensor:
        return F.cross_entropy(self.masked_logits(self.model(x), y), y)

    # ---- hooks ----------------------------------------------------------
    def before_task(self, task: TaskData) -> None:
        cls = list(task.classes)
        for c in cls:
            self.same_task[c, cls] = True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr,
                                         momentum=self.cfg.momentum)

    def make_batch(self, x, y, task: TaskData):
        return x, y

    def extra_loss(self, x, y, task: TaskData) -> torch.Tensor | float:
        return 0.0

    def after_step(self, x, y, task: TaskData) -> None:
        pass

    def after_task(self, task: TaskData) -> None:
        pass

    # ---- core loop ------------------------------------------------------
    def train_task(self, task: TaskData) -> None:
        self.before_task(task)
        self.model.train()
        n = len(task.y_train)
        for self.epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                x = task.x_train[idx].to(self.device)
                y = task.y_train[idx].to(self.device)
                xb, yb = self.make_batch(x, y, task)
                loss = self.task_loss(xb, yb, task) + self.extra_loss(xb, yb, task)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.stats["grad_steps"] += 1
                self.after_step(x, y, task)
        self.after_task(task)
        self.seen_tasks.append(task)


class Naive(Strategy):
    """Fine-tune on each task in turn with no protection against forgetting."""
    name = "naive"


class Replay(Strategy):
    """Experience replay (ER) with a reservoir-sampled buffer.

    The buffer holds at most `mem_size` samples in total. Reservoir sampling keeps
    a uniform random sample of everything seen so far, so old tasks keep a share of
    the buffer without needing task boundaries. Each training step concatenates the
    current batch with an equally sized batch drawn uniformly from the buffer.
    """
    name = "replay"

    def __init__(self, *args, mem_size: int = 500, **kw):
        super().__init__(*args, **kw)
        self.mem_size = mem_size
        self.buf_x = torch.empty(mem_size, *self.input_shape, device=self.device)
        self.buf_y = torch.empty(mem_size, dtype=torch.long, device=self.device)
        self.n_filled = 0
        self.n_seen = 0  # stream count for reservoir sampling

    def sample_buffer(self, n: int) -> torch.Tensor:
        return torch.randint(self.n_filled, (n,), generator=self.gen).to(self.device)

    def make_batch(self, x, y, task):
        if self.n_filled == 0:
            return x, y
        idx = self.sample_buffer(len(y))
        return torch.cat([x, self.buf_x[idx]]), torch.cat([y, self.buf_y[idx]])

    def reservoir_slots(self, n: int) -> list[tuple[int, int]]:
        """Reservoir sampling for the next n stream samples: (batch index, buffer slot) pairs
        for the samples that get stored."""
        out = []
        for i in range(n):
            if self.n_filled < self.mem_size:
                out.append((i, self.n_filled))
                self.n_filled += 1
            else:
                j = int(torch.randint(self.n_seen + 1, (1,), generator=self.gen))
                if j < self.mem_size:
                    out.append((i, j))
            self.n_seen += 1
        return out

    def after_step(self, x, y, task):
        # Only add samples on the first epoch, so each sample enters the stream once.
        if self.epoch > 0:
            return
        for i, j in self.reservoir_slots(len(y)):
            self.buf_x[j], self.buf_y[j] = x[i], y[i]

    def buffer_class_counts(self) -> list[int]:
        return torch.bincount(self.buf_y[:self.n_filled], minlength=N_CLASSES).tolist()


class DERPP(Replay):
    """DER++ (Buzzega et al., NeurIPS 2020, "Dark Experience for General Continual Learning").

    Same reservoir buffer as Replay, but each stored sample also keeps the logits the
    model produced for it when it was seen. Loss per step:
        CE(current batch)
      + der_alpha * MSE(model(buffer batch A), stored logits A)      # "dark knowledge"
      + der_beta  * CE(model(buffer batch B), labels B)              # plain replay term
    with A and B drawn independently from the buffer, each of batch_size samples.
    """
    name = "derpp"

    def __init__(self, *args, der_alpha: float = 0.5, der_beta: float = 0.5, **kw):
        super().__init__(*args, **kw)
        self.alpha, self.beta = der_alpha, der_beta
        self.buf_logits = torch.empty(self.mem_size, N_CLASSES, device=self.device)
        self._cur_logits: torch.Tensor | None = None

    def make_batch(self, x, y, task):
        return x, y  # replay enters through extra_loss, not by concatenation

    def task_loss(self, x, y, task):
        logits = self.model(x)
        self._cur_logits = logits.detach()  # stored with the sample (pre-update, as in the paper)
        return F.cross_entropy(self.masked_logits(logits, y), y)

    def extra_loss(self, x, y, task):
        if self.n_filled == 0:
            return 0.0
        a = self.sample_buffer(self.cfg.batch_size)
        loss = self.alpha * F.mse_loss(self.model(self.buf_x[a]), self.buf_logits[a])
        b = self.sample_buffer(self.cfg.batch_size)
        logits_b = self.masked_logits(self.model(self.buf_x[b]), self.buf_y[b])
        return loss + self.beta * F.cross_entropy(logits_b, self.buf_y[b])

    def after_step(self, x, y, task):
        if self.epoch > 0:
            return
        for i, j in self.reservoir_slots(len(y)):
            self.buf_x[j], self.buf_y[j], self.buf_logits[j] = x[i], y[i], self._cur_logits[i]


class EWC(Strategy):
    """Online EWC (Kirkpatrick et al. 2017; online variant of Schwarz et al. 2018).

    After each task, estimate the diagonal Fisher information F of the parameters on
    that task's data, add it to a running total, and anchor to the current weights θ*.
    While training later tasks, add  (λ/2) Σ F_i (θ_i - θ*_i)^2  to the loss, which
    makes weights that mattered for old tasks expensive to move.

    Fisher uses labels sampled from the model's own predictive distribution (the
    "true" Fisher, not the empirical one), with per-sample gradients via torch.func.
    """
    name = "ewc"

    def __init__(self, *args, ewc_lambda: float = 10000.0, fisher_samples: int = 1000,
                 ewc_gamma: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.lam = ewc_lambda
        self.fisher_samples = fisher_samples
        self.gamma = ewc_gamma  # decay of older Fisher terms; 1.0 = plain accumulation
        self.fisher: dict[str, torch.Tensor] | None = None
        self.anchor: dict[str, torch.Tensor] | None = None

    def extra_loss(self, x, y, task):
        if self.fisher is None:
            return 0.0
        pen = sum((self.stiffness[k] * (p - self.anchor[k]).pow(2)).sum()
                  for k, p in self.model.named_parameters())
        return 0.5 * pen

    def _update_stiffness(self) -> None:
        # Per-weight curvature of the penalty is λ·F. Gradient descent on a quadratic
        # diverges once lr·λ·F exceeds ~2, which happens for large λ (weights go NaN).
        # Cap it at 1/lr: at that point one step already pulls the weight straight back
        # to its anchor, so stiffer gives no extra protection, only instability.
        cap = 1.0 / self.cfg.lr
        self.stiffness = {k: (self.lam * f).clamp(max=cap) for k, f in self.fisher.items()}

    def after_task(self, task):
        n = min(self.fisher_samples, len(task.y_train))
        idx = torch.randperm(len(task.y_train), generator=self.gen)[:n]
        x = task.x_train[idx].to(self.device)
        y_task = task.y_train[idx].to(self.device)  # only used to pick the task-IL mask

        self.model.eval()
        params = {k: p.detach() for k, p in self.model.named_parameters()}
        with torch.no_grad():
            logits = self.masked_logits(self.model(x), y_task)
            y_sampled = torch.distributions.Categorical(logits=logits).sample()
        mask = self.same_task[y_task] if self.cfg.scenario == "task" else torch.ones_like(logits, dtype=torch.bool)

        def nll(p, xi, yi, mi):
            out = functional_call(self.model, p, (xi.unsqueeze(0),))
            out = out.masked_fill(~mi.unsqueeze(0), float("-inf"))
            return F.cross_entropy(out, yi.unsqueeze(0))

        fisher = {k: torch.zeros_like(v) for k, v in params.items()}
        # Per-sample gradients take chunk * n_params floats; keep that around 0.5 GB.
        n_params = sum(v.numel() for v in params.values())
        chunk = max(8, min(256, int(1.3e8 // n_params)))
        for i in range(0, n, chunk):
            g = vmap(grad(nll), in_dims=(None, 0, 0, 0))(params, x[i:i + chunk], y_sampled[i:i + chunk],
                                                         mask[i:i + chunk])
            for k in fisher:
                fisher[k] += g[k].pow(2).sum(0)
        self.model.train()

        if self.fisher is None:
            self.fisher = {k: f / n for k, f in fisher.items()}
        else:
            self.fisher = {k: self.gamma * self.fisher[k] + fisher[k] / n for k in fisher}
        self.anchor = {k: v.clone() for k, v in params.items()}
        self._update_stiffness()


STRATEGIES: dict[str, type[Strategy]] = {
    "naive": Naive,
    "replay": Replay,
    "ewc": EWC,
    "derpp": DERPP,
}
