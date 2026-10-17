"""
demo_scaled_dpp_attention_3.py
==============================
Scalability benchmark evaluating memory capacity and crosstalk across 
larger pattern sets in the Version 3 framework.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_lib_3 import build_art_network, well_separated_patterns

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
STEPS_PER_PATTERN = 350
DT = 0.1
SEED = 3

PATTERNS = well_separated_patterns(PATTERN_SIZE, N_PATTERNS, active_bits=ACTIVE_BITS, seed=9)
NOVEL_PROBE = well_separated_patterns(PATTERN_SIZE, N_PATTERNS + 1, active_bits=ACTIVE_BITS, seed=9)[-1]

CONFIGS = [
    ("Full (DPP + Attention)", dict(use_stdp=True, use_dpp=True, use_attn=True)),
    ("No DPP (random allocation)", dict(use_stdp=True, use_dpp=False, use_attn=True)),
    ("No attention (uniform inhibition)", dict(use_stdp=True, use_dpp=True, use_attn=False)),
    ("Nothing (no STDP/DPP/attention)", dict(use_stdp=False, use_dpp=False, use_attn=False)),
]


def run_capacity_curve(use_stdp, use_dpp, use_attn, seed):
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn,
        use_stdp=use_stdp, seed=seed, boost=BOOST)
    net, art = system["net"], system["art"]
    inp, mem, ev = system["input"], system["memory"], system["evaluator"]
    proj = system["proj_in_mem"]
    ssz = system["substrate_size"]

    def present(pattern, steps=STEPS_PER_PATTERN, allow_allocate=True):
        pattern = np.asarray(pattern, dtype=float)
        mism = np.zeros(steps)
        allocated = False
        for t in range(steps):
            needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:ssz], 20.0)
            if needs_alloc and allow_allocate and not allocated:
                art.allocate(pattern, proj.weights)
                allocated = True
                needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:ssz], 20.0)
            ev_drive = np.maximum(pattern * 55.0 - inhib, 0.0)
            ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
            net.step({"Input": pattern * 180.0, "Evaluator": ext_ev}, dt=DT)
            mism[t] = ev.spiked[:PATTERN_SIZE].sum()
        return float(mism.mean())

    # 1. Train every pattern once, in order.
    for p in PATTERNS:
        present(p)

    # 2. Capacity curve: re-test every pattern, in its ORIGINAL training
    #    order, so degradation on early patterns from later ones crowding
    #    the substrate shows up directly as a function of training order.
    curve = [present(p, steps=200, allow_allocate=False) for p in PATTERNS]

    # 3. Sensitivity check: a pattern that was never trained at all.
    novel_mismatch = present(NOVEL_PROBE, steps=200, allow_allocate=False)

    return {"curve": np.array(curve), "n_slots": len(art.slots),
            "novel_mismatch": novel_mismatch}


print("=" * 88)
print(f"Scaled demo: pattern_size={PATTERN_SIZE}  substrate_size={SUBSTRATE_SIZE} "
      f"({SUBSTRATE_SIZE // SLOT_SIZE} native slots)  n_patterns={N_PATTERNS}")
print(f"({SUBSTRATE_SIZE + SUBSTRATE_SIZE // 4} total memory neurons incl. shared inhibitory pool "
      f"vs. ~10 in the original two-column script)")
print("=" * 88)

results = {}
for name, kwargs in CONFIGS:
    print(f"Running: {name} ...")
    results[name] = run_capacity_curve(seed=SEED, **kwargs)
    c = results[name]["curve"]
    print(f"  -> slots allocated={results[name]['n_slots']:3d}  "
          f"mean retest mismatch={c.mean():.4f}  "
          f"(first 10 patterns={c[:10].mean():.4f}, last 10={c[-10:].mean():.4f})  "
          f"novel-probe mismatch={results[name]['novel_mismatch']:.4f}")

print()
print("Reading the numbers: 'first 10' vs 'last 10' patterns tests whether EARLY-learned")
print("categories degrade as more categories are added later (interference/specificity),")
print("and novel-probe mismatch should stay clearly above the trained-pattern average")
print("(sensitivity: correctly flagging something genuinely new instead of habituating")
print("to everything indiscriminately).")

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
axes[0].axhline(np.mean([results[n]["novel_mismatch"] for n, _ in CONFIGS]),
                color="black", ls="--", lw=1, alpha=0.6, label="avg. novel-probe mismatch")
axes[0].set_title(f"Capacity curve: retest mismatch vs. training order, {N_PATTERNS} patterns "
                   f"into {SUBSTRATE_SIZE // SLOT_SIZE} native slots (reuse forced beyond pattern "
                   f"{SUBSTRATE_SIZE // SLOT_SIZE})")
axes[0].set_xlabel("Pattern index (training order)")
axes[0].set_ylabel("Retest mismatch")
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
plt.savefig("/home/claude/dpp_art/demo_scaled_dpp_attention_results.png", dpi=150)
print("Figure saved -> demo_scaled_dpp_attention_results.png")
