"""
benchmark_baselines.py
======================
Comparative evaluation benchmarking SNN memory performance against classical 
symbolic template matchers and binary ART-1 models across noise sweeps.
"""

import argparse, itertools, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment_core import (CFG, NOISE_TYPES, run_snn, run_template_baseline, run_art1_baseline)
from stats_utils import mean_ci, paired_compare

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
RAW = os.path.join(args.out, "benchmark_raw.json")
SEEDS = list(range(args.seeds))
cache = json.load(open(RAW)) if os.path.exists(RAW) else {}

CELLS = {"default": {}}
SWEEPS = {"density (active bits of 16)": [("3", dict(active_bits=3)), ("5", {}), ("8", dict(active_bits=8))],
          "overlap with pattern 0": [("none", dict(overlap_frac=0.0)), ("0.5", {}), ("0.8", dict(overlap_frac=0.8))],
          "capacity (patterns; 12 native slots)": [("8", dict(n_patterns=8)), ("16", {}), ("24", dict(n_patterns=24))],
          "training order": [("fixed", {}), ("random", dict(order="random")), ("reverse", dict(order="reverse"))]}
for sw, levels in SWEEPS.items():
    for lvl, kw in levels:
        CELLS[f"{sw}|{lvl}"] = kw
SNN_METHODS = {"SNN full": (True, True, True), "SNN no-attention": (True, True, False),
               "SNN random-alloc": (True, False, True), "SNN nothing": (False, False, False)}
SNN_ALL_CELLS = ("SNN full", "SNN no-attention")        # run in every sweep cell
BASELINES = {"Template matcher": lambda s, c: run_template_baseline(s, c),
             "ART-1 (rho=0.6)": lambda s, c: run_art1_baseline(s, 0.6, c),
             "ART-1 (rho=0.8)": lambda s, c: run_art1_baseline(s, 0.8, c)}
TESTS = ("auroc", "auroc_swap1", "auroc_swap2", "auroc_drop2")
TEST_LABEL = {"auroc": "clean", "auroc_swap1": "swap1", "auroc_swap2": "swap2", "auroc_drop2": "drop2"}


def cell_cfg(name):
    cfg = dict(CFG)
    if name == "default":
        return cfg
    sw, lvl = name.split("|")
    cfg.update(dict(SWEEPS[sw])[lvl])
    return cfg


def cell_default_equiv(name):
    """Sweep cells whose parameters equal the default (e.g. 'density|5') reuse the default cell's runs."""
    return "default" if (name != "default" and cell_cfg(name) == CFG) else name


t0 = time.time()
jobs = []
for cell in CELLS:
    canon = cell_default_equiv(cell)
    if canon != cell:
        continue
    methods = list(SNN_METHODS) if cell == "default" else list(SNN_ALL_CELLS)
    for m in methods:
        for sd in SEEDS:
            jobs.append((cell, m, sd))
for i, (cell, m, sd) in enumerate(jobs):
    k = f"{cell}|{m}|{sd}"
    if k not in cache:
        s, d, a = SNN_METHODS[m]
        cache[k] = run_snn(sd, s, d, a, cfg=cell_cfg(cell), with_noise=True)
        cache[k].pop("per_pattern", None)
        json.dump(cache, open(RAW, "w"))
    if (i + 1) % 6 == 0:
        print(f"  [{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
for cell in CELLS:
    if cell_default_equiv(cell) != cell:
        continue
    for bn, fn in BASELINES.items():
        for sd in SEEDS:
            k = f"{cell}|{bn}|{sd}"
            if k not in cache:
                cache[k] = fn(sd, cell_cfg(cell))
json.dump(cache, open(RAW, "w"))
print(f"runs complete in {time.time() - t0:.0f}s", flush=True)


def vals(cell, method, test):
    cell = cell_default_equiv(cell)
    return np.array([cache[f"{cell}|{method}|{sd}"][test] for sd in SEEDS])


def primary(cell, method):
    return np.mean([vals(cell, method, t) for t in TESTS], axis=0)       # per-seed mean over 4 conditions


ALL_METHODS = list(SNN_METHODS) + list(BASELINES)
print("\n" + "=" * 104)
print(f"DEFAULT CELL: AUROC (novel vs known) by method and known-probe condition, {len(SEEDS)} matched seeds")
print("mean [95% bootstrap CI]. PRIMARY = mean over the 4 conditions.")
print("=" * 104)
print(f"{'method':20s}" + "".join(f"{TEST_LABEL[t]:>19s}" for t in TESTS) + f"{'PRIMARY (mean of 4)':>24s}  slots")
for m in ALL_METHODS:
    cells = []
    for t in TESTS:
        mm, lo, hi = mean_ci(vals("default", m, t))
        cells.append(f"{mm:.3f} [{lo:.2f},{hi:.2f}]")
    mm, lo, hi = mean_ci(primary("default", m))
    ns = np.mean(vals("default", m, "n_slots"))
    print(f"{m:20s}" + "".join(f"{c:>19s}" for c in cells) + f"{mm:.3f} [{lo:.2f},{hi:.2f}]".rjust(24) + f"  {ns:.1f}")

print("\nPaired comparisons at the default cell (primary AUROC; difference, 95% CI, sign-flip p):")
for a, b in [("SNN full", "Template matcher"), ("SNN full", "ART-1 (rho=0.6)"), ("SNN full", "ART-1 (rho=0.8)"),
             ("SNN full", "SNN no-attention"), ("SNN full", "SNN nothing"), ("SNN full", "SNN random-alloc")]:
    r = paired_compare(primary("default", a), primary("default", b))
    print(f"  {a:18s} - {b:18s}: {r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  d_z={r['dz']:+.2f}  p={r['p']:.3f}")
print("  by condition, SNN full minus Template matcher:")
for t in TESTS:
    r = paired_compare(vals("default", "SNN full", t), vals("default", "Template matcher", t))
    print(f"    {TEST_LABEL[t]:6s} {r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p']:.3f}")

print("\n" + "=" * 104)
print("SWEEPS: primary AUROC (mean of 4 conditions) per level; mean [95% CI]; slots = categories created")
print("=" * 104)
sweep_methods = ["SNN full", "SNN no-attention", "Template matcher", "ART-1 (rho=0.6)", "ART-1 (rho=0.8)"]
for sw, levels in SWEEPS.items():
    print(f"\n  {sw}")
    print(f"  {'level':8s}" + "".join(f"{m:>22s}" for m in sweep_methods))
    for lvl, _ in levels:
        row = []
        for m in sweep_methods:
            mm, lo, hi = mean_ci(primary(f"{sw}|{lvl}", m))
            row.append(f"{mm:.3f} [{lo:.2f},{hi:.2f}]")
        print(f"  {lvl:8s}" + "".join(f"{r:>22s}" for r in row))
print("\n  SNN full minus Template matcher, paired, per sweep level (primary AUROC):")
for sw, levels in SWEEPS.items():
    parts = []
    for lvl, _ in levels:
        r = paired_compare(primary(f"{sw}|{lvl}", "SNN full"), primary(f"{sw}|{lvl}", "Template matcher"))
        parts.append(f"{lvl}:{r['mean']:+.3f}(p={r['p']:.2f})")
    print(f"   {sw:38s} " + "  ".join(parts))

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
colors = {"SNN full": "green", "SNN no-attention": "crimson", "Template matcher": "navy",
          "ART-1 (rho=0.6)": "darkorange", "ART-1 (rho=0.8)": "goldenrod"}
for a_, (sw, levels) in zip(ax.ravel()[:4], SWEEPS.items()):
    xs = np.arange(len(levels))
    for m in sweep_methods:
        ms, los, his = [], [], []
        for lvl, _ in levels:
            mm, lo, hi = mean_ci(primary(f"{sw}|{lvl}", m))
            ms.append(mm), los.append(mm - lo), his.append(hi - mm)
        a_.errorbar(xs + 0.04 * sweep_methods.index(m), ms, yerr=[los, his], marker="o", ms=4, capsize=2,
                    color=colors[m], label=m)
    a_.set_xticks(xs)
    a_.set_xticklabels([l for l, _ in levels])
    a_.set_title(sw)
    a_.set_ylabel("primary AUROC")
    a_.set_ylim(0.3, 1.02)
    a_.grid(True, alpha=0.3)
ax[0, 0].legend(fontsize=7, loc="lower left")
a_ = ax[1, 1]
w = 0.15
for j, m in enumerate(sweep_methods):
    ms = [mean_ci(vals("default", m, t))[0] for t in TESTS]
    a_.bar(np.arange(4) + (j - 2) * w, ms, w, color=colors[m], label=m)
a_.set_xticks(np.arange(4))
a_.set_xticklabels([TEST_LABEL[t] for t in TESTS])
a_.set_ylim(0.3, 1.02)
a_.set_title("default cell, by known-probe condition\n(clean = stored template; others = not the stored template)")
a_.grid(True, alpha=0.3, axis="y")
a_ = ax[1, 2]
for j, m in enumerate(list(SNN_METHODS)):
    mm, lo, hi = mean_ci(primary("default", m))
    a_.bar(j, mm, yerr=[[mm - lo], [hi - mm]], color="seagreen", capsize=3)
a_.set_xticks(range(len(SNN_METHODS)))
a_.set_xticklabels(list(SNN_METHODS), rotation=15, fontsize=8)
a_.set_ylim(0.3, 1.02)
a_.set_title("SNN mechanisms at the default cell\n(primary AUROC)")
a_.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "benchmark_results.png"), dpi=140)
print(f"\nFigure saved -> {args.out}/benchmark_results.png")
