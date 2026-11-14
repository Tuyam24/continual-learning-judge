"""Evaluation and continual-learning metrics.

Accuracy matrix R: R[i, j] = test accuracy on task j right after training on
task i. Only j <= i is filled (seen tasks); the rest is NaN.

  average accuracy after task i  = mean(R[i, :i+1])
  final average accuracy (ACC)   = mean(R[T-1, :])
  forgetting (Chaudhry et al. 2018), for each old task j < T-1:
      f_j = max_{l in [j, T-2]} R[l, j] - R[T-1, j]
  i.e. the drop from the best accuracy ever reached on j to its final accuracy.
  Reported forgetting is the mean of f_j.
"""
from __future__ import annotations

import numpy as np
import torch

SCENARIOS = ("class", "task")


def active_logits(logits: torch.Tensor, classes: tuple[int, ...] | None, scenario: str) -> torch.Tensor:
    """Class-incremental: all 10 logits compete (no task identity at test time).
    Task-incremental: logits outside the given task's classes are masked out,
    so the model only has to pick between that task's digits."""
    if scenario == "class" or classes is None:
        return logits
    mask = torch.full_like(logits, float("-inf"))
    mask[:, list(classes)] = 0.0
    return logits + mask


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device,
             classes: tuple[int, ...] | None = None, scenario: str = "class",
             batch_size: int = 2000) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        pred = active_logits(model(xb), classes, scenario).argmax(1)
        correct += (pred == yb).sum().item()
    model.train(was_training)
    return correct / len(y)


def new_acc_matrix(n_tasks: int) -> np.ndarray:
    return np.full((n_tasks, n_tasks), np.nan)


def avg_acc_curve(R: np.ndarray) -> np.ndarray:
    """Average accuracy over seen tasks after each task."""
    return np.array([np.nanmean(R[i, :i + 1]) for i in range(len(R))])


def final_avg_acc(R: np.ndarray) -> float:
    return float(np.mean(R[-1]))


def forgetting(R: np.ndarray) -> float:
    T = len(R)
    if T < 2:
        return 0.0
    drops = [np.nanmax(R[j:T - 1, j]) - R[T - 1, j] for j in range(T - 1)]
    return float(np.mean(drops))


def format_matrix(R: np.ndarray) -> str:
    T = len(R)
    head = "after\\eval " + " ".join(f"  T{j}  " for j in range(T)) + "  | avg"
    lines = [head]
    for i in range(T):
        cells = " ".join("   -  " if np.isnan(v) else f"{v * 100:5.1f} " for v in R[i])
        lines.append(f"  task {i}   {cells} | {np.nanmean(R[i, :i + 1]) * 100:5.1f}")
    return "\n".join(lines)
