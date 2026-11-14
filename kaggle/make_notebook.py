"""Build a self-contained notebook that runs the CIFAR-10 suite on Kaggle or Colab.

    .venv/bin/python kaggle/make_notebook.py                        # round 2: kaggle/cl_cifar10_suite.ipynb
    .venv/bin/python kaggle/make_notebook.py --chain relax_cifar10  # round 3: kaggle/cl_relax_cifar10.ipynb
    .venv/bin/python kaggle/make_notebook.py --chain round4_cifar10 # round 4: kaggle/cl_round4_cifar10.ipynb

The notebook embeds every source file (via %%writefile), so it is the only thing you
upload. It downloads CIFAR-10 itself, runs `run_suite.py --only cifar10` on the GPU,
builds the report, and zips the results for download. Re-run this script after
changing the code so the notebook picks up the changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--chain", default="cifar10", choices=("cifar10", "relax_cifar10", "round4_cifar10"))
CHAIN = ap.parse_args().chain
NAME = {"cifar10": "cl_cifar10_suite", "relax_cifar10": "cl_relax_cifar10", "round4_cifar10": "cl_round4_cifar10"}[CHAIN]
ZIP = {"cifar10": "cl_cifar10_results", "relax_cifar10": "cl_relax_cifar10_results",
       "round4_cifar10": "cl_round4_cifar10_results"}[CHAIN]
INTRO = {
    "cifar10": ("Continual learning on Split CIFAR-10: naive, EWC, replay, DER++ and ours",
                "learning-rate and DER++ tuning on val, then 5 methods + 1 ablation + 2 compute-matched\n"
                "baselines, 3 seeds each) takes roughly 2 hours on a T4"),
    "round4_cifar10": ("Round 4: keep new learning on rollback (Split CIFAR-10)",
                       "3 variants on val with 2 seeds each, then the best on test with 3 seeds if it\n"
                       "beats the strict judge) takes roughly 1 hour on a T4"),
    "relax_cifar10": ("Round 3: relaxed judge on Split CIFAR-10",
                      "6 judge settings tuned on val with 2 seeds each, then the winner on test with\n"
                      "3 seeds) takes roughly 1.5-2 hours on a T4"),
}[CHAIN]

FILES = ["cl/__init__.py", "cl/data.py", "cl/model.py", "cl/metrics.py", "cl/strategies.py",
         "cl/ours.py", "run_experiment.py", "run_suite.py", "make_report.py"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"""
# {INTRO[0]}

Self-contained copy of the `continual_learning` project. Run all cells in order.

**Before running:** turn on a GPU.
- **Kaggle:** Settings → Accelerator → GPU T4 x2 (or P100), and Settings → Internet → On (needed to download CIFAR-10).
- **Colab:** Runtime → Change runtime type → T4 GPU.

The run ({INTRO[1]}. Progress is printed as it goes and also written to
`results/suite_progress.log`. Tip: use Save Version -> Save & Run All (Commit) so it keeps running
with the browser closed.
"""),
    code("""
import os, torch
os.makedirs("/tmp/cl_project/cl", exist_ok=True)
os.chdir("/tmp/cl_project")
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Turn one on (see the first cell) or this will take many hours.")
"""),
    md("## Project files\nEach cell below writes one file of the project."),
]
for f in FILES:
    # Kaggle rejects a %%writefile cell with no body, so empty files (cl/__init__.py) get a comment.
    src = (ROOT / f).read_text() or "# package marker\n"
    cells.append(code(f"%%writefile {f}\n{src}"))

cells += [
    md("## Run the CIFAR-10 suite\nRuns as a subprocess so its progress lines stream into the output below."),
    code("""
import subprocess, sys
p = subprocess.Popen([sys.executable, "-u", "run_suite.py", "--only", "CHAIN"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
print("exit code", p.wait())
"""),
    # Save the zip BEFORE the report: a report failure must never lose the results.
    md(f"## Download the results\nDownload `{ZIP}.zip` (Kaggle: Output panel on the right; Colab: Files panel on the left) "
       "and unzip it inside the `continual_learning` folder on your computer."),
    code("""
import shutil
dest = "/kaggle/working/ZIP" if os.path.isdir("/kaggle/working") else "/content/ZIP"
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.make_archive(dest, "zip", root_dir="/tmp/cl_project", base_dir="results")
print("saved", dest + ".zip")
"""),
    md("## Report\nOnly drawn when this run produced test results (a tuning-only run may not)."),
    code("""
from IPython.display import Image, Markdown, display
import glob, subprocess, sys
if glob.glob("results/cifar10/*.json"):
    subprocess.run([sys.executable, "make_report.py", "--results", "results/cifar10"], check=False)
if os.path.exists("results/cifar10/summary.md"):
    display(Markdown(open("results/cifar10/summary.md").read()))
    for img in ("avg_acc.png", "forgetting.png"):
        if os.path.exists(f"results/cifar10/{img}"):
            display(Image(f"results/cifar10/{img}"))
else:
    print("No test results in this run, so no report. Tuning results are in the zip and the log above.")
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for c in cells:  # fill in the chain / zip names in the run and download cells
    c["source"] = [l.replace('"CHAIN"', f'"{CHAIN}"').replace("/ZIP", f"/{ZIP}") for l in c["source"]]
out = ROOT / "kaggle" / f"{NAME}.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
