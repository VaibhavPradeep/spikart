"""
noise_fix_study.py
==================
Diagnostic study evaluating the impact of post-training probe window length 
and normalized learned keys on swapped-bit noise resolution.
"""

import argparse, json, os, warnings
import numpy as np
from snn_lib_changing import build_art_network, run_schedule, probe_only
from experiment_core import make_task, NOISE_TYPES
from stats_utils import auroc, mean_ci, paired_compare

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)
RAW = os.path.join(args.out, "noise_fix_raw.json")
cache = json.load(open(RAW)) if os.path.exists(RAW) else {}
TESTS = ("clean",) + NOISE_TYPES


def evaluate(system, task, steps):
    pr = lambda p: probe_only(system, p, steps=steps, warmup=400)
    nov = [pr(p) for p in task["novel"]]
    out = {"clean": auroc(nov, [pr(p) for p in task["train"]])}
    for k in NOISE_TYPES:
        out[k] = auroc(nov, [pr(p) for p in task["noisy"][k]])
    return out


def train(seed, learned):
    task = make_task(seed)
    s = build_art_network(16, 144, 12, 0.6, True, True, True, seed=seed, boost=4.0, use_learned_keys=learned)
    run_schedule(s, task["schedule"], steps_per_pattern=500, probe_steps=200, probe_warmup=400)
    return task, s


for sd in range(args.seeds):
    if f"A200|{sd}" not in cache:
        task, s = train(sd, False)
        for w in (200, 400, 600, 1000):
            cache[f"A{w}|{sd}"] = evaluate(s, task, w)
        s["art"].use_learned_keys = True
        cache[f"B600|{sd}"] = evaluate(s, task, 600)
        task, s = train(sd, True)
        cache[f"C600|{sd}"] = evaluate(s, task, 600)
        json.dump(cache, open(RAW, "w"))
        print(f"seed {sd} done", flush=True)

names = ["A200", "A400", "A600", "A1000", "B600", "C600"]
print(f"\nAUROC (novel vs known) mean [95% CI over {args.seeds} matched seeds]")
print(f"{'variant':8s}" + "".join(f"{t:>20s}" for t in TESTS))
for n in names:
    cells = []
    for t in TESTS:
        m, lo, hi = mean_ci([cache[f'{n}|{sd}'][t] for sd in range(args.seeds)])
        cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}]")
    print(f"{n:8s}" + "".join(f"{c:>20s}" for c in cells))
print("\nPaired differences vs A200 (current read-out), mean [95% CI], sign-flip p:")
for n in names[1:]:
    parts = []
    for t in ("swap1", "swap2"):
        r = paired_compare([cache[f'{n}|{sd}'][t] for sd in range(args.seeds)],
                           [cache[f'A200|{sd}'][t] for sd in range(args.seeds)])
        parts.append(f"{t}: {r['mean']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.3f}")
    print(f"  {n:7s} " + "   ".join(parts))
