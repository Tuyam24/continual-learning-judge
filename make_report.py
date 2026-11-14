"""Build the results report from results/*.json.

    .venv/bin/python make_report.py [--results results]

Reads the top-level JSON files in the results folder (subfolders and files with
eval_split == "val" are skipped) and writes:

  results/summary.md      one markdown table per scenario (also printed to stdout)
  results/avg_acc.png     average accuracy over seen tasks after each task, per scenario
  results/forgetting.png  final accuracy and forgetting per main method, per scenario

Each file is labelled by its filename stem (e.g. replay_class, ours_class_no-judge).
A file is a "main" method when its stem is exactly <strategy>_<scenario>; anything
after that is a tag and marks the file as a variant/ablation. Variants appear in
the table only, after the main methods.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENARIO_ORDER = ("class", "task")
SCENARIO_TITLE = {"class": "Class-incremental", "task": "Task-incremental"}
MAIN_ORDER = ("naive", "ewc", "replay", "derpp", "ours")
METHOD_NAME = {"naive": "Naive fine-tuning", "ewc": "Online EWC", "replay": "Replay", "derpp": "DER++", "ours": "Ours"}
# Okabe-Ito palette (colour-blind safe); fixed per method so both figures agree.
METHOD_COLOR = {"naive": "#999999", "ewc": "#56B4E9", "replay": "#E69F00", "derpp": "#009E73", "ours": "#D55E00"}
EXTRA_COLORS = ["#CC79A7", "#0072B2", "#F0E442", "#000000"]
# Optional stats columns: key in the JSON -> column header.
STAT_COLUMNS = {"grad_steps": "grad steps", "rollbacks": "rollbacks"}


@dataclass
class Result:
    label: str          # filename stem
    strategy: str
    scenario: str
    tag: str            # "" for main methods
    runs: list
    stats: dict = field(default_factory=dict)  # stat key -> list of per-seed values (or one value)

    @property
    def is_main(self) -> bool:
        return self.tag == ""

    def per_seed(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.runs], dtype=float)

    @property
    def curves(self) -> np.ndarray:
        return np.array([r["avg_acc_curve"] for r in self.runs], dtype=float)


def _stat_values(raw) -> list[float]:
    """Accept a scalar, a list of per-seed values, or {'mean': .., 'std': ..}."""
    if isinstance(raw, dict):
        if "values" in raw:
            return [float(v) for v in raw["values"]]
        if "mean" in raw:
            return [float(raw["mean"])]
        return []
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw if isinstance(v, (int, float))]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return []


def collect_stats(d: dict) -> dict:
    stats: dict[str, list[float]] = {}
    top = d.get("stats") or {}
    for key in STAT_COLUMNS:
        if key in top:
            stats[key] = _stat_values(top[key])
    # Per-run stats take precedence when present (gives a real std over seeds).
    for key in STAT_COLUMNS:
        vals = [float(r["stats"][key]) for r in d.get("runs", [])
                if isinstance(r.get("stats"), dict) and isinstance(r["stats"].get(key), (int, float))]
        if vals:
            stats[key] = vals
    return {k: v for k, v in stats.items() if v}


def load_results(folder: Path) -> list[Result]:
    out = []
    for path in sorted(folder.glob("*.json")):  # glob on the folder itself: top level only
        if not path.is_file():
            continue
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {path.name}: {e}")
            continue
        if d.get("eval_split") == "val":
            continue
        if not d.get("runs"):
            print(f"skip {path.name}: no runs")
            continue
        strategy = d.get("strategy", path.stem.split("_")[0])
        scenario = (d.get("config") or {}).get("scenario", "class")
        base = f"{strategy}_{scenario}"
        stem = path.stem
        tag = stem[len(base):].lstrip("_") if stem.startswith(base) else stem
        out.append(Result(stem, strategy, scenario, tag, d["runs"], collect_stats(d)))
    return out


def sort_key(r: Result):
    rank = MAIN_ORDER.index(r.strategy) if r.strategy in MAIN_ORDER else len(MAIN_ORDER)
    return (not r.is_main, rank, r.strategy, r.tag)


def pm(vals: np.ndarray, pct: bool = True, digits: int = 2) -> str:
    vals = np.asarray(vals, dtype=float)
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    if len(vals) == 1:
        return f"{vals[0] * scale:.{digits}f}{unit}"
    return f"{vals.mean() * scale:.{digits}f} ± {vals.std() * scale:.{digits}f}{unit}"


def fmt_stat(vals: list[float]) -> str:
    a = np.array(vals, dtype=float)
    integral = np.allclose(a, np.round(a))
    if len(a) == 1:
        return f"{int(a[0]):,}" if integral else f"{a[0]:.2f}"
    digits = 0 if integral and a.mean() >= 100 else 1
    return f"{a.mean():,.{digits}f} ± {a.std():,.{digits}f}"


def method_label(r: Result) -> str:
    name = METHOD_NAME.get(r.strategy, r.strategy)
    return f"{name} [{r.tag}]" if r.tag else name


DATASET_TITLE = {"mnist": "Split MNIST", "cifar10": "Split CIFAR-10"}
TITLE = "Split MNIST"  # set in main() from the result files' "dataset" field


def dataset_title(folder: Path) -> str:
    """Name of the benchmark in `folder`. Files written before the --dataset flag existed
    have no "dataset" field; those are all MNIST."""
    names = set()
    for path in folder.glob("*.json"):
        try:
            names.add(json.loads(path.read_text()).get("dataset", "mnist"))
        except (ValueError, OSError):
            continue
    return " + ".join(DATASET_TITLE.get(n, n) for n in sorted(names)) or "Split MNIST"


def make_tables(results: list[Result]) -> str:
    scenarios = [s for s in SCENARIO_ORDER if any(r.scenario == s for r in results)]
    scenarios += sorted({r.scenario for r in results} - set(scenarios))
    parts = [f"# {TITLE} results (test split)", "",
             "Mean ± std over seeds. ACC = final average accuracy over all 5 tasks; "
             "forgetting = mean drop from best to final accuracy on tasks 1-4 "
             "(Chaudhry et al. 2018). Rows below the rule are tagged variants/ablations.", ""]
    for sc in scenarios:
        rows = sorted([r for r in results if r.scenario == sc], key=sort_key)
        stat_keys = [k for k in STAT_COLUMNS if any(k in r.stats for r in rows)]
        header = ["Method", "File", "ACC", "Forgetting", "Seeds"] + [STAT_COLUMNS[k] for k in stat_keys]
        parts += [f"## {SCENARIO_TITLE.get(sc, sc)} ({sc}-IL)", "",
                  "| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"]
        prev_main = None
        for r in rows:
            if prev_main is True and not r.is_main:
                parts.append("| " + " | ".join(["*variants*"] + [""] * (len(header) - 1)) + " |")
            prev_main = r.is_main
            cells = [method_label(r), f"`{r.label}`", pm(r.per_seed("final_avg_acc")),
                     pm(r.per_seed("forgetting")), str(len(r.runs))]
            cells += [fmt_stat(r.stats[k]) if k in r.stats else "–" for k in stat_keys]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")
    return "\n".join(parts)


def color_for(strategy: str, extra: dict) -> str:
    if strategy in METHOD_COLOR:
        return METHOD_COLOR[strategy]
    if strategy not in extra:
        extra[strategy] = EXTRA_COLORS[len(extra) % len(EXTRA_COLORS)]
    return extra[strategy]


def main_methods(results: list[Result]) -> list[str]:
    present = {r.strategy for r in results if r.is_main}
    return [m for m in MAIN_ORDER if m in present] + sorted(present - set(MAIN_ORDER))


def present_scenarios(results: list[Result]) -> list[str]:
    """Scenarios that have main-method results, so no chart gets an empty panel."""
    return [s for s in SCENARIO_ORDER if any(r.is_main and r.scenario == s for r in results)] or ["class"]


def plot_avg_acc(results: list[Result], path: Path, extra: dict) -> None:
    scs = present_scenarios(results)
    fig, axes = plt.subplots(1, len(scs), figsize=(max(9.5, 5.5 * len(scs)), 4.4), squeeze=False)
    axes = axes[0]
    for ax, sc in zip(axes, scs):
        rows = {r.strategy: r for r in results if r.is_main and r.scenario == sc}
        lo = 100.0
        for m in main_methods(results):
            if m not in rows:
                continue
            c = rows[m].curves * 100
            x = np.arange(1, c.shape[1] + 1)
            mean, std = c.mean(0), c.std(0)
            lo = min(lo, float((mean - std).min()))
            col = color_for(m, extra)
            ax.plot(x, mean, marker="o", ms=4, lw=2, color=col,
                    label=f"{METHOD_NAME.get(m, m)} (n={len(c)})")
            ax.fill_between(x, mean - std, np.minimum(mean + std, 100), color=col, alpha=0.2, lw=0)
        ax.set_title(SCENARIO_TITLE[sc] + (" (primary)" if sc == "class" else ""))
        ax.set_xlabel("Tasks learned")
        ax.set_xticks(range(1, 6))
        # Zoom in when every curve stays high (task-IL), so differences are visible.
        ax.set_ylim(0 if lo < 60 else 10 * math.floor((lo - 1) / 10), 101)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(alpha=0.3)
        if not rows:
            ax.text(0.5, 0.5, "no results", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(loc="lower left", fontsize=9, frameon=False)
    for ax in axes:
        ax.set_ylabel("Average accuracy on seen tasks")
    fig.suptitle(f"{TITLE}: average accuracy after each task (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(results: list[Result], path: Path, extra: dict) -> None:
    methods = main_methods(results)
    metrics = [("final_avg_acc", "Final avg accuracy"), ("forgetting", "Forgetting")]
    scs = present_scenarios(results)
    fig, axes = plt.subplots(1, len(scs), figsize=(max(9.5, 5.5 * len(scs)), 4.4), sharey=True, squeeze=False)
    axes = axes[0]
    width = 0.8 / max(len(methods), 1)
    for ax, sc in zip(axes, scs):
        rows = {r.strategy: r for r in results if r.is_main and r.scenario == sc}
        for k, m in enumerate(methods):
            if m not in rows:
                continue
            vals = [rows[m].per_seed(key) * 100 for key, _ in metrics]
            means = [v.mean() for v in vals]
            stds = [v.std() for v in vals]
            x = np.arange(len(metrics)) + (k - (len(methods) - 1) / 2) * width
            bars = ax.bar(x, means, width * 0.92, yerr=stds, capsize=3, color=color_for(m, extra),
                          edgecolor="black", linewidth=0.5, label=METHOD_NAME.get(m, m),
                          error_kw={"elinewidth": 1})
            for b, mu, sd in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, mu + sd + 1.5, f"{mu:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([f"{lbl}\n({'higher' if key == 'final_avg_acc' else 'lower'} is better)"
                            for key, lbl in metrics])
        ax.set_title(SCENARIO_TITLE[sc] + (" (primary)" if sc == "class" else ""))
        ax.set_ylim(0, 112)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", alpha=0.3)
        if not rows:
            ax.text(0.5, 0.5, "no results", ha="center", va="center", transform=ax.transAxes)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if labels:  # a folder with only tagged variants has no main methods to plot
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"{TITLE}: final accuracy and forgetting (mean ± std over seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(HERE / "results"), help="folder with the result JSONs")
    p.add_argument("--out", default=None, help="output folder (default: same as --results)")
    args = p.parse_args()
    src = Path(args.results)
    out = Path(args.out) if args.out else src
    out.mkdir(parents=True, exist_ok=True)

    global TITLE
    TITLE = dataset_title(src)
    results = load_results(src)
    if not results:
        raise SystemExit(f"no test-split result files in {src}")
    md = make_tables(results)
    (out / "summary.md").write_text(md + "\n")
    print(md)
    extra: dict[str, str] = {}
    plot_avg_acc(results, out / "avg_acc.png", extra)
    plot_bars(results, out / "forgetting.png", extra)
    for name in ("summary.md", "avg_acc.png", "forgetting.png"):
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
