"""
excerpt_retrieval.py
====================
Evaluates partial-pattern completion (excerpt retrieval) comparing masked 
containment matching, naive Jaccard, and spiking activity readouts.
"""

import argparse, json, os, warnings
import numpy as np
from snn_lib_changing import build_art_network, run_schedule, retrieve_spiking
from experiment_core import make_task, CFG
from stats_utils import mean_ci, paired_compare

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)
RAW = os.path.join(args.out, "excerpt_raw.json")
cache = json.load(open(RAW)) if os.path.exists(RAW) else {}
KINDS = ["bits1", "bits2", "bits3", "bits4", "block6"]
METHODS = ["containment", "naive Jaccard", "spiking", "ceiling"]


def run_seed(seed):
    task = make_task(seed)
    s = build_art_network(CFG["pattern_size"], CFG["substrate_size"], CFG["slot_size"], CFG["vigilance"],
                          True, True, True, seed=seed, boost=CFG["boost"])
    run_schedule(s, task["schedule"], steps_per_pattern=CFG["steps_per_pattern"], probe_steps=50, probe_warmup=50)
    art = s["art"]
    slot_of = {}
    for i, p in enumerate(task["train"]):          # slot whose template equals the trained pattern
        for j, sl in enumerate(art.slots):
            if np.array_equal(sl["template"], p):
                slot_of[i] = j
                break
    rng = np.random.default_rng(777 + seed)
    acc = {k: {m: {"top1": [], "top3": []} for m in METHODS} for k in KINDS}
    T = np.array([sl["template"] for sl in art.slots])
    for i, p in enumerate(task["train"]):
        if i not in slot_of:
            continue
        target = slot_of[i]
        on = np.where(p == 1)[0]
        for kind in KINDS:
            for _ in range(args.draws):
                frag, known = np.zeros_like(p), None
                if kind.startswith("bits"):
                    k = min(int(kind[-1]), len(on))
                    frag[rng.choice(on, k, replace=False)] = 1.0
                else:
                    st = int(rng.integers(0, len(p) - 6 + 1))
                    known = np.zeros(len(p), bool)
                    known[st:st + 6] = True
                    frag = p * known
                    if frag.sum() == 0:
                        continue                       # no active evidence in the block: skip
                rank = {}
                rank["containment"] = [r[0] for r in art.retrieve(frag, known=known, top_k=3)]
                sc = art.match_scores(frag)
                rank["naive Jaccard"] = list(np.lexsort((np.arange(len(sc)), -sc))[:3])
                rank["spiking"] = [r[0] for r in retrieve_spiking(s, frag, top_k=3, steps=300, warmup=300)]
                fv = frag * (np.ones(len(p), bool) if known is None else known)
                cand = np.where((np.minimum(fv[None, :], T).sum(1) >= fv.sum() - 1e-9))[0]
                for m in METHODS[:3]:
                    acc[kind][m]["top1"].append(float(rank[m][0] == target))
                    acc[kind][m]["top3"].append(float(target in rank[m]))
                acc[kind]["ceiling"]["top1"].append(1.0 / len(cand) if target in cand else 0.0)
                acc[kind]["ceiling"]["top3"].append(min(1.0, 3.0 / len(cand)) if target in cand else 0.0)
    return {k: {m: {a: float(np.mean(v)) if v else float("nan") for a, v in d.items()} for m, d in kd.items()}
            for k, kd in acc.items()}


for sd in range(args.seeds):
    if str(sd) not in cache:
        cache[str(sd)] = run_seed(sd)
        json.dump(cache, open(RAW, "w"))
        print(f"seed {sd} done", flush=True)

print(f"\nTop-1 accuracy by fragment type, mean [95% CI] over {args.seeds} seeds "
      f"({args.draws} fragments x 16 patterns per seed per type)")
print(f"{'fragment':9s}" + "".join(f"{m:>22s}" for m in METHODS))
for kind in KINDS:
    row = []
    for m in METHODS:
        mm, lo, hi = mean_ci([cache[str(sd)][kind][m]["top1"] for sd in range(args.seeds)])
        row.append(f"{mm:.3f} [{lo:.2f},{hi:.2f}]")
    print(f"{kind:9s}" + "".join(f"{r:>22s}" for r in row))
print("\nTop-3 accuracy")
for kind in KINDS:
    row = []
    for m in METHODS:
        mm, lo, hi = mean_ci([cache[str(sd)][kind][m]["top3"] for sd in range(args.seeds)])
        row.append(f"{mm:.3f} [{lo:.2f},{hi:.2f}]")
    print(f"{kind:9s}" + "".join(f"{r:>22s}" for r in row))
print("\nPaired top-1 differences, spiking minus containment (mean [CI], p):")
for kind in KINDS:
    r = paired_compare([cache[str(sd)][kind]["spiking"]["top1"] for sd in range(args.seeds)],
                       [cache[str(sd)][kind]["containment"]["top1"] for sd in range(args.seeds)])
    print(f"  {kind:7s} {r['mean']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.3f}")
