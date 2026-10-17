"""
demo_scaled_merged.py -- merged edition
==================================

Scalability showcase for the snn_lib ART-style slotted novelty detector.
This Script runs the same as the ablation code underlying snn_lib_merged at a larger scale reporting a cpacity curve.

"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_lib_merged import build_art_network, run_schedule, probe_only, well_separated_patterns, force_overlap

# -----------------------------------------------------------------------
# Scale configuration
# -----------------------------------------------------------------------
PATTERN_SIZE = 24
ACTIVE_BITS = 6
N_PATTERNS = 48
SUBSTRATE_SIZE = 480          # 40 "native" slots at slot_size=12
SLOT_SIZE = 12
VIGILANCE = 0.55
BOOST = 4.0
TRAIN_STEPS = 350
RETEST_STEPS = 200
PROBE_STEPS = 100
DT = 0.1
SEED = 3

PATTERNS = well_separated_patterns(PATTERN_SIZE, N_PATTERNS, active_bits=ACTIVE_BITS, seed=9)
PATTERNS = force_overlap(PATTERNS, source_idx=0, target_indices=[3, 12, 25, 40], frac=0.5)
NOVEL_PROBE = well_separated_patterns(PATTERN_SIZE, N_PATTERNS + 1, active_bits=ACTIVE_BITS, seed=9)[-1]

CONFIGS = [
    ("Full (DPP + Attention)", dict(use_stdp=True, use_dpp=True, use_attn=True)),
    ("No DPP (random allocation)", dict(use_stdp=True, use_dpp=False, use_attn=True)),
    ("No attention (uniform inhibition)", dict(use_stdp=True, use_dpp=True, use_attn=False)),
    ("Nothing (no STDP/DPP/attention)", dict(use_stdp=False, use_dpp=False, use_attn=False)),
]


def run_capacity_curve(use_stdp, use_dpp, use_attn, seed):
    t0 = time.time()
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn,
        use_stdp=use_stdp, seed=seed, boost=BOOST)
    build_time = time.time() - t0

    # 1. Train every pattern once, in order. run_schedule's built-in probe
    #    phase gives us a genuine "how novel did this look on first
    #    contact" reading for free, before this same call's own commit.
    schedule = [(f"train p{i}", p) for i, p in enumerate(PATTERNS)]
    exposures = run_schedule(system, schedule, steps_per_pattern=TRAIN_STEPS,
                              dt=DT, probe_steps=PROBE_STEPS)
    first_exposure = np.array([r["probe_mismatch"] for r in exposures])

    # 2. Capacity curve: re-test every pattern, in ORIGINAL training order,
    #    with a genuinely zero-side-effect read (no allocation, learning
    #    hard-disabled) so degradation from later patterns crowding the
    #    substrate shows up as a pure function of training order.
    curve = np.array([probe_only(system, p, steps=RETEST_STEPS, dt=DT) for p in PATTERNS])

    # 3. Sensitivity check: a pattern that was never trained at all.
    novel_mismatch = probe_only(system, NOVEL_PROBE, steps=RETEST_STEPS, dt=DT)

    return {"curve": curve, "first_exposure": first_exposure,
            "n_slots": len(system["art"].slots), "novel_mismatch": novel_mismatch,
            "build_time": build_time}


print("=" * 92)
print(f"Scaled demo: pattern_size={PATTERN_SIZE} substrate_size={SUBSTRATE_SIZE} "
      f"({SUBSTRATE_SIZE // SLOT_SIZE} native slots) n_patterns={N_PATTERNS}")
print("Patterns 3, 12, 25, 40 were deliberately given 50% overlap with pattern 0's active bits,")
print("so content-driven interference has something to actually detect (well-separated random")
print("patterns alone rarely overlap enough to stress this at this scale).")
print("=" * 92)

results = {}
for name, kwargs in CONFIGS:
    print(f"Running: {name} ...")
    results[name] = run_capacity_curve(seed=SEED, **kwargs)
    r = results[name]
    c = r["curve"]
    print(f"  -> build={r['build_time']*1000:5.1f} ms  slots allocated={r['n_slots']:3d}  "
          f"mean retest mismatch={c.mean():.4f} "
          f"(first 10 patterns={c[:10].mean():.4f}, last 10={c[-10:].mean():.4f}) "
          f"novel-probe mismatch={r['novel_mismatch']:.4f}")

print()
print("Reading the numbers: 'first 10' vs 'last 10' patterns tests whether EARLY-learned")
print("categories degrade as more categories are added later (interference/specificity), and")
print("novel-probe mismatch should stay clearly above the trained-pattern average (sensitivity:")
print("correctly flagging something genuinely new instead of habituating to everything).")
print("Patterns 3/12/25/40 (forced overlap with p0) are worth comparing on first_exposure vs")
print("their well-separated neighbors -- shared bits with an already-learned pattern should")
print("make them look measurably less novel on first contact.")

overlap_idx = [3, 12, 25, 40]
neighbor_idx = [i for i in range(N_PATTERNS) if i not in overlap_idx][:len(overlap_idx)]
for name, _ in CONFIGS:
    fe = results[name]["first_exposure"]
    print(f"  {name:36s} overlap patterns first-exposure={fe[overlap_idx].mean():.4f}  "
          f"vs. non-overlap neighbors={fe[neighbor_idx].mean():.4f}")

# -----------------------------------------------------------------------
# Figure: capacity curve per configuration + summary bars
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
colors = {"Full (DPP + Attention)": "green", "No DPP (random allocation)": "steelblue",
          "No attention (uniform inhibition)": "crimson", "Nothing (no STDP/DPP/attention)": "gray"}
x = np.arange(N_PATTERNS)
for name, _ in CONFIGS:
    axes[0].plot(x, results[name]["curve"], marker="o", ms=3, lw=1.2,
                 color=colors[name], label=name)
for oi in overlap_idx:
    axes[0].axvline(oi, color="purple", ls=":", lw=1, alpha=0.5)
axes[0].axhline(np.mean([results[n]["novel_mismatch"] for n, _ in CONFIGS]),
                 color="black", ls="--", lw=1, alpha=0.6, label="avg. novel-probe mismatch")
axes[0].set_title(f"Capacity curve: retest mismatch vs. training order ({N_PATTERNS} patterns, "
                   f"{SUBSTRATE_SIZE // SLOT_SIZE} native slots)\n"
                   f"gold = capacity boundary (reuse forced beyond here); "
                   f"purple = deliberately-overlapping patterns", fontsize=11)
axes[0].set_xlabel("Pattern index (training order)")
axes[0].set_ylabel("Retest mismatch (zero-side-effect probe)")
axes[0].axvline(SUBSTRATE_SIZE // SLOT_SIZE - 0.5, color="gold", ls=":", lw=1.5,
                 label="substrate capacity boundary")
axes[0].legend(loc="upper left", fontsize=8)
axes[0].grid(True, alpha=0.3)

names = [n for n, _ in CONFIGS]
first10 = [results[n]["curve"][:10].mean() for n in names]
last10 = [results[n]["curve"][-10:].mean() for n in names]
novelp = [results[n]["novel_mismatch"] for n in names]
xw = np.arange(len(names))
width = 0.25
axes[1].bar(xw - width, first10, width, label="first 10 patterns (retest)", color="seagreen")
axes[1].bar(xw, last10, width, label="last 10 patterns (retest)", color="darkorange")
axes[1].bar(xw + width, novelp, width, label="never-trained probe", color="crimson")
axes[1].set_xticks(xw)
axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[1].set_title("Specificity (first vs. last 10) and sensitivity (novel probe) by configuration")
axes[1].set_ylabel("Mean mismatch")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("demo_scaled_results.png", dpi=150)
print("Figure saved -> demo_scaled_results.png")