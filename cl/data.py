"""Split MNIST / Split CIFAR-10: a 10-class dataset cut into sequential tasks of
disjoint class pairs (task 0 = classes 0-1, task 1 = classes 2-3, ...).

Each task carries four splits:
  - train: what the learner sees while that task is "current"
  - judge: the first `judge_per_task` samples of the held-out slice (default 40).
           Only for in-loop decisions of a strategy (the judge / rollback in our
           method). Counted against that strategy's memory budget when stored.
  - val:   the rest of the held-out slice of the *training* set (never trained on).
           Used for hyperparameter / design selection (--eval-split val); disjoint
           from judge, so tuning never reuses the data a strategy makes decisions on.
  - test:  the official test set for that task's classes; used only for reporting.

Everything is loaded into memory as float tensors, which is much faster than a
DataLoader for models this small. MNIST is flattened to 784 (for the MLP); CIFAR-10
stays (3, 32, 32) and is normalised per channel. No data augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision import datasets

N_CLASSES = 10
DATASETS = ("mnist", "cifar10")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TaskData:
    task_id: int
    classes: tuple[int, ...]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_judge: torch.Tensor | None = None
    y_judge: torch.Tensor | None = None

    def __repr__(self) -> str:
        n_judge = 0 if self.y_judge is None else len(self.y_judge)
        return (f"Task {self.task_id} classes={self.classes} train={len(self.y_train)} "
                f"judge={n_judge} val={len(self.y_val)} test={len(self.y_test)}")


def _load_mnist(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.MNIST(root, train=train, download=True)
    x = ds.data.float().div_(255).sub_(MNIST_MEAN).div_(MNIST_STD).view(-1, 28 * 28)
    return x, ds.targets.clone()


def _load_cifar10(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255)  # N,3,32,32
    x.sub_(torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)).div_(torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
    return x, torch.tensor(np.array(ds.targets), dtype=torch.long)


_LOADERS = {"mnist": _load_mnist, "cifar10": _load_cifar10}


def make_split(dataset: str = "mnist", root: str = "data", n_tasks: int = 5, val_fraction: float = 0.1,
               seed: int = 0, judge_per_task: int = 40) -> list[TaskData]:
    """Return `n_tasks` tasks; task t holds classes [t*k, ..., t*k + k-1], k = 10 / n_tasks.

    The judge set is carved from the held-out (val) slice, not from train, so the
    train split -- and hence every test number -- is unaffected by it."""
    if N_CLASSES % n_tasks:
        raise ValueError(f"n_tasks must divide {N_CLASSES}, got {n_tasks}")
    load = _LOADERS[dataset]
    k = N_CLASSES // n_tasks
    x_tr, y_tr = load(root, train=True)
    x_te, y_te = load(root, train=False)
    gen = torch.Generator().manual_seed(seed)

    tasks = []
    for t in range(n_tasks):
        classes = tuple(range(t * k, (t + 1) * k))
        cls = torch.tensor(classes)

        m = torch.isin(y_tr, cls)
        x, y = x_tr[m], y_tr[m]
        perm = torch.randperm(len(y), generator=gen)
        n_val = int(len(y) * val_fraction)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        if not 0 <= judge_per_task < n_val:
            raise ValueError(f"judge_per_task must be in [0, {n_val}), got {judge_per_task}")
        judge_idx, val_idx = val_idx[:judge_per_task], val_idx[judge_per_task:]

        mt = torch.isin(y_te, cls)
        tasks.append(TaskData(t, classes, x[tr_idx], y[tr_idx], x[val_idx], y[val_idx],
                              x_te[mt], y_te[mt], x[judge_idx], y[judge_idx]))
    return tasks


def make_split_mnist(root: str = "data", n_tasks: int = 5, val_fraction: float = 0.1,
                     seed: int = 0, judge_per_task: int = 40) -> list[TaskData]:
    return make_split("mnist", root, n_tasks, val_fraction, seed, judge_per_task)
