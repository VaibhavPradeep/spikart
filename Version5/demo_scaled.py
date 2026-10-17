"""
demo_scaled.py
==============
Scalability capacity curve evaluation at ~600 neurons and 48 categories using
read-only pre-commit and post-training probes.
"""
import argparse, json, os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_lib_changing import (build_art_network, run_schedule, probe_only,
                              well_separated_patterns, force_overlap)
from stats_utils import mean_ci, paired_compare, auroc

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)
RAW = os.path.join(args.out, "demo_scaled_raw.json")

PATTERN_SIZE, ACTIVE_BITS, N_PATTERNS, N_NOVEL = 24, 6, 48, 8
SUBSTRATE_SIZE, SLOT_SIZE = 480, 12
VIGILANCE, BOOST = 0.55, 4.0
TRAIN_STEPS, PROBE_STEPS, PROBE_WARMUP = 350, 200, 400
DT = 0.1
OVERLAP_IDX = [3, 12, 25, 40]
SEEDS = list(range(args.seeds))
CONFIGS = [
    ("Full (DPP + Attention)", dict(use_stdp=True, use_dpp=True, use_attn=True)),
    ("No DPP (random allocation)", dict(use_stdp=True, use_dpp=False, use_attn=True)),
    ("No attention (uniform inhibition)", dict(use_stdp=True, use_dpp=True, use_attn=False)),
    ("Nothing (no STDP/DPP/attention)", dict(use_stdp=False, use_dpp=False, use_attn=False)),
]
cache = json.load(open(RAW)) if os.path.exists(RAW) else {}


def make_patterns(seed):
    pats = well_separated_patterns(PATTERN_SIZE, N_PATTERNS + N_NOVEL, active_bits=ACTIVE_BITS, seed=9 + seed)
    train, novel = pats[:N_PATTERNS], pats[N_PATTERNS:]
    return force_overlap(train, 0, OVERLAP_IDX, frac=0.5), novel


def run_capacity_curve(use_stdp, use_dpp, use_attn, seed):
    train, novel = make_patterns(seed)
    t0 = time.time()
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn, use_stdp=use_stdp,
        seed=3 + seed, boost=BOOST)
    build_time = time.time() - t0
    ex = run_schedule(system, [(f"train p{i}", p) for i, p in enumerate(train)],
                      steps_per_pattern=TRAIN_STEPS, dt=DT, probe_steps=PROBE_STEPS,
                      probe_warmup=PROBE_WARMUP)
    first = np.array([e["probe_mismatch"] for e in ex])
    curve = np.array([probe_only(system, p, steps=PROBE_STEPS, dt=DT, warmup=PROBE_WARMUP) for p in train])
    nov = np.array([probe_only(system, p, steps=PROBE_STEPS, dt=DT, warmup=PROBE_WARMUP) for p in novel])
    return {"curve": curve.tolist(), "first_exposure": first.tolist(), "novel": nov.tolist(),
            "auroc": auroc(nov, curve), "n_slots": len(system["art"].slots),
            "n_fallback": system["art"].n_fallback_allocations, "build_time": build_time}


print("=" * 100)
print(f"Scaled demo: pattern_size={PATTERN_SIZE} substrate_size={SUBSTRATE_SIZE} "
      f"({SUBSTRATE_SIZE // SLOT_SIZE} native slots) n_patterns={N_PATTERNS}, {len(SEEDS)} seeds")
print(f"Patterns {OVERLAP_IDX} share 50% of pattern 0's active bits (bits are added, so they grow).")
print("All reads are read-only from-rest probes (400 warm-up + 200 measured ticks).")
print("=" * 100)
t0 = time.time()
for name, kw in CONFIGS:
    for sd in SEEDS:
        k = f"{name}|{sd}"
        if k not in cache:
            print(f"Running: {name}  seed {sd} ...", flush=True)
            cache[k] = run_capacity_curve(seed=sd, **kw)
            json.dump(cache, open(RAW, "w"))
print(f"runs complete in {time.time() - t0:.0f}s")


def arr(name, field):
    return np.array([cache[f"{name}|{sd}"][field] for sd in SEEDS])


print("\n" + f"{'configuration':36s} {'mean retest':>22s} {'first 10':>20s} {'last 10':>20s} "
      f"{'novel':>20s} {'AUROC':>8s} {'slots':>6s} {'build ms':>8s}")
for name, _ in CONFIGS:
    C = arr(name, "curve")
    cells = [mean_ci(C.mean(1)), mean_ci(C[:, :10].mean(1)), mean_ci(C[:, -10:].mean(1)),
             mean_ci(arr(name, "novel").mean(1))]
    f = lambda t: f"{t[0]:.4f} [{t[1]:.4f},{t[2]:.4f}]"
    print(f"{name:36s} {f(cells[0]):>22s} {f(cells[1]):>20s} {f(cells[2]):>20s} {f(cells[3]):>20s} "
          f"{arr(name, 'auroc').mean():8.3f} {arr(name, 'n_slots').mean():6.1f} {1000 * arr(name, 'build_time').mean():8.1f}")

print("\nPaired comparisons vs 'Full' (matched seeds), mean retest mismatch and AUROC:")
base = CONFIGS[0][0]
for name, _ in CONFIGS[1:]:
    r1 = paired_compare(arr(name, "curve").mean(1), arr(base, "curve").mean(1))
    r2 = paired_compare(arr(name, "auroc"), arr(base, "auroc"))
    print(f"  {name:36s} retest {r1['mean']:+.4f} [{r1['ci_lo']:+.4f},{r1['ci_hi']:+.4f}] p={r1['p']:.3f}   "
          f"AUROC {r2['mean']:+.3f} [{r2['ci_lo']:+.3f},{r2['ci_hi']:+.3f}] p={r2['p']:.3f}   (n={len(SEEDS)} seeds)")

print("\nSpecificity check -- early vs late-learned patterns (retest, read-only probes):")
for name, _ in CONFIGS:
    C = arr(name, "curve")
    r = paired_compare(C[:, -10:].mean(1), C[:, :10].mean(1))
    print(f"  {name:36s} last10 - first10 = {r['mean']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")

neighbor_idx = [i for i in range(N_PATTERNS) if i not in OVERLAP_IDX][:len(OVERLAP_IDX)]
print("\nFirst-exposure (pre-commit) mismatch, deliberately-overlapping patterns vs non-overlap neighbors:")
for name, _ in CONFIGS:
    F = arr(name, "first_exposure")
    r = paired_compare(F[:, OVERLAP_IDX].mean(1), F[:, neighbor_idx].mean(1))
    print(f"  {name:36s} overlap={F[:, OVERLAP_IDX].mean():.4f}  neighbors={F[:, neighbor_idx].mean():.4f}  "
          f"diff {r['mean']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")

fig, axes = plt.subplots(2, 1, figsize=(13, 10))
colors = {"Full (DPP + Attention)": "green", "No DPP (random allocation)": "steelblue",
          "No attention (uniform inhibition)": "crimson", "Nothing (no STDP/DPP/attention)": "gray"}
x = np.arange(N_PATTERNS)
for name, _ in CONFIGS:
    C = arr(name, "curve")
    m = C.mean(0)
    lo, hi = np.array([mean_ci(C[:, i]) [1:] for i in x]).T
    axes[0].plot(x, m, marker="o", ms=3, lw=1.2, color=colors[name], label=name)
    axes[0].fill_between(x, lo, hi, color=colors[name], alpha=0.12)
for oi in OVERLAP_IDX:
    axes[0].axvline(oi, color="purple", ls=":", lw=1, alpha=0.5)
axes[0].axhline(np.mean([arr(n, "novel").mean() for n, _ in CONFIGS]), color="black", ls="--", lw=1, alpha=0.6,
                label="avg. never-trained mismatch")
axes[0].axvline(SUBSTRATE_SIZE // SLOT_SIZE - 0.5, color="gold", ls=":", lw=1.5, label="native-slot capacity")
axes[0].set_title(f"Capacity curve: read-only retest mismatch vs training order ({N_PATTERNS} patterns, "
                  f"{len(SEEDS)} seeds, band = 95% CI); purple = overlapping patterns")
axes[0].set_xlabel("pattern index (training order)")
axes[0].set_ylabel("retest mismatch (probe, from rest)")
axes[0].legend(loc="upper left", fontsize=8)
axes[0].grid(True, alpha=0.3)

names = [n for n, _ in CONFIGS]
xw, width = np.arange(len(names)), 0.25
for off, lab, col, fn in ((-width, "first 10 (retest)", "seagreen", lambda C, N: C[:, :10].mean(1)),
                          (0, "last 10 (retest)", "darkorange", lambda C, N: C[:, -10:].mean(1)),
                          (width, "never-trained", "crimson", lambda C, N: N.mean(1))):
    ms, es = [], [[], []]
    for n in names:
        v = fn(arr(n, "curve"), arr(n, "novel"))
        m, lo, hi = mean_ci(v)
        ms.append(m), es[0].append(m - lo), es[1].append(hi - m)
    axes[1].bar(xw + off, ms, width, yerr=es, capsize=3, label=lab, color=col)
axes[1].set_xticks(xw)
axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[1].set_title("Specificity (first vs last 10) and sensitivity (never-trained) by configuration, 95% CI over seeds")
axes[1].set_ylabel("mean mismatch")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "demo_scaled_results.png"), dpi=140)
print(f"Figure saved -> {args.out}/demo_scaled_results.png")
