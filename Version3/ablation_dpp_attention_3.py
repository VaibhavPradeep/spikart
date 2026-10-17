"""
ablation_dpp_attention_3.py
===========================
Multi-seed ablation testing STDP plasticity, attention routing, and DPP 
slot allocation across scaled pattern sets.
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_lib_3 import build_art_network, run_schedule, well_separated_patterns

# -----------------------------------------------------------------------
# Configuration (scaled up substantially vs. the original 8-neuron,
# 2-column script: this substrate is ~15x more neurons and supports many
# more simultaneously-learnable categories/columns via slot allocation)
# -----------------------------------------------------------------------
PATTERN_SIZE = 16
ACTIVE_BITS = 5
N_PATTERNS = 16          # > n_slots below, so allocation is forced to reuse
SUBSTRATE_SIZE = 144
SLOT_SIZE = 12           # -> 12 "native" slots; patterns 12-15 force reuse
VIGILANCE = 0.6
BOOST = 4.0
STEPS_PER_PATTERN = 500
N_SEEDS = 4
DT = 0.1

PATTERNS = well_separated_patterns(PATTERN_SIZE, N_PATTERNS, active_bits=ACTIVE_BITS, seed=7)
NOVEL_PROBE = well_separated_patterns(PATTERN_SIZE, N_PATTERNS + 1, active_bits=ACTIVE_BITS, seed=7)[-1]


def make_schedule():
    sched = []
    for i, p in enumerate(PATTERNS):
        sched.append((f"p{i} 1st", p))
        sched.append((f"p{i} retest", p))
    sched.append(("p0 after all learned", PATTERNS[0]))
    sched.append(("NOVEL probe (never trained)", NOVEL_PROBE))
    return sched


SCHEDULE = make_schedule()


def run_condition(use_stdp, use_dpp, use_attn, seed, record_series=False):
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn,
        use_stdp=use_stdp, seed=seed, boost=BOOST)

    if not record_series:
        results = run_schedule(system, SCHEDULE, steps_per_pattern=STEPS_PER_PATTERN, dt=DT)
    else:
        # Re-implemented per-tick so we can also keep the full mismatch
        # time series for the summary figure (run_schedule only returns
        # per-exposure aggregates).
        net, art = system["net"], system["art"]
        inp, mem, ev = system["input"], system["memory"], system["evaluator"]
        proj = system["proj_in_mem"]
        ssz = system["substrate_size"]
        results, series = [], []
        for label, pattern in SCHEDULE:
            pattern = np.asarray(pattern, dtype=float)
            mism = np.zeros(STEPS_PER_PATTERN)
            allocated = False
            for t in range(STEPS_PER_PATTERN):
                needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:ssz], 20.0)
                if needs_alloc and not allocated:
                    art.allocate(pattern, proj.weights)
                    allocated = True
                    needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:ssz], 20.0)
                ev_drive = np.maximum(pattern * 55.0 - inhib, 0.0)
                ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
                net.step({"Input": pattern * 180.0, "Evaluator": ext_ev}, dt=DT)
                mism[t] = ev.spiked[:PATTERN_SIZE].sum()
            series.append(mism)
            results.append({"label": label, "mean_mismatch": float(mism.mean()),
                             "allocated": allocated, "n_slots": len(art.slots)})

    firsts = np.mean([r["mean_mismatch"] for r in results if "1st" in r["label"]])
    retests = np.mean([r["mean_mismatch"] for r in results if "retest" in r["label"]])
    after = next(r["mean_mismatch"] for r in results if r["label"] == "p0 after all learned")
    novel = next(r["mean_mismatch"] for r in results if "NOVEL probe" in r["label"])
    n_slots = results[-1]["n_slots"]
    diag = system["art"].allocation_diagnostics()

    out = {"first": firsts, "retest": retests, "after": after, "novel": novel,
           "n_slots": n_slots, "alloc_quality": diag["mean_quality"],
           "alloc_redundancy": diag["mean_redundancy"]}
    if record_series:
        out["series"] = np.concatenate(series)
    return out


# =============================================================================
# 1. FULL 2x2x2 FACTORIAL, MULTIPLE SEEDS
# =============================================================================
print("=" * 92)
print(f"Full factorial ablation: STDP x DPP x Attention, {N_SEEDS} seeds each")
print(f"  pattern_size={PATTERN_SIZE}  substrate_size={SUBSTRATE_SIZE}  slot_size={SLOT_SIZE} "
      f"(-> {SUBSTRATE_SIZE // SLOT_SIZE} native slots for {N_PATTERNS} patterns: reuse is forced)")
print("=" * 92)

header = (f"{'STDP':>5s} {'DPP':>5s} {'ATTN':>5s} | {'1st':>7s} {'retest':>7s} "
          f"{'after*':>7s} {'novel*':>7s} | {'slots':>5s} | {'q_dpp':>6s} {'redund':>6s}")
print(header)
print("-" * len(header))

factorial = {}
for use_stdp, use_dpp, use_attn in itertools.product([True, False], repeat=3):
    metrics = [run_condition(use_stdp, use_dpp, use_attn, seed) for seed in range(N_SEEDS)]
    agg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    std = {k: np.std([m[k] for m in metrics]) for k in metrics[0]}
    factorial[(use_stdp, use_dpp, use_attn)] = (agg, std)
    print(f"{str(use_stdp):>5s} {str(use_dpp):>5s} {str(use_attn):>5s} | "
          f"{agg['first']:7.4f} {agg['retest']:7.4f} {agg['after']:7.4f} {agg['novel']:7.4f} | "
          f"{agg['n_slots']:5.1f} | {agg['alloc_quality']:6.3f} {agg['alloc_redundancy']:6.3f}")

print("(* 'after' = specificity probe: p0 re-tested once every other pattern has since been")
print("   learned. 'novel' = sensitivity probe: a pattern the network was never trained on.)")

# -----------------------------------------------------------------------
# Main-effect summary: average over the OTHER two factors to isolate each
# switch's marginal effect on retest mismatch (the core habituation metric).
# -----------------------------------------------------------------------
def main_effect(factor_index):
    on = np.mean([factorial[c][0]["retest"] for c in factorial if c[factor_index]])
    off = np.mean([factorial[c][0]["retest"] for c in factorial if not c[factor_index]])
    return on, off

print()
print("Main effects on retest mismatch (lower = better recognition), averaged over the other two factors:")
for name, idx in [("STDP", 0), ("DPP", 1), ("Attention", 2)]:
    on, off = main_effect(idx)
    rel = 100.0 * (off - on) / off if off > 1e-9 else 0.0
    print(f"  {name:10s}: ON={on:.4f}  OFF={off:.4f}  (ON reduces mismatch by {rel:.1f}% vs OFF)")

# DPP's construction-level effect, isolated cleanly regardless of the other factors
q_dpp = np.mean([factorial[c][0]["alloc_quality"] for c in factorial if c[1]])
q_rand = np.mean([factorial[c][0]["alloc_quality"] for c in factorial if not c[1]])
r_dpp = np.mean([factorial[c][0]["alloc_redundancy"] for c in factorial if c[1]])
r_rand = np.mean([factorial[c][0]["alloc_redundancy"] for c in factorial if not c[1]])
print()
print("DPP's direct, construction-level effect on WHICH neurons get allocated")
print("(independent of downstream spiking dynamics):")
print(f"  mean quality    (higher=better): DPP={q_dpp:.4f}  random={q_rand:.4f}")
print(f"  mean redundancy (lower=better):  DPP={r_dpp:.4f}  random={r_rand:.4f}")

# =============================================================================
# 2. TIME-SERIES RUNS FOR THE SUMMARY FIGURE (one seed each, for legibility)
# =============================================================================
print()
print("Running representative time-series conditions for the figure...")
series_conditions = [
    ("Full (STDP+DPP+Attn)", dict(use_stdp=True, use_dpp=True, use_attn=True)),
    ("No attention", dict(use_stdp=True, use_dpp=True, use_attn=False)),
    ("No STDP", dict(use_stdp=False, use_dpp=True, use_attn=True)),
    ("Nothing (all off)", dict(use_stdp=False, use_dpp=False, use_attn=False)),
]
series_data = {}
for name, kwargs in series_conditions:
    r = run_condition(seed=42, record_series=True, **kwargs)
    series_data[name] = r["series"]

# =============================================================================
# 3. FIGURE
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(13, 12))
time_axis = np.arange(len(next(iter(series_data.values())))) * DT

def rolling_mean(x, window=150):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

colors = {"Full (STDP+DPP+Attn)": "green", "No attention": "crimson",
          "No STDP": "orange", "Nothing (all off)": "gray"}
for name, mism in series_data.items():
    axes[0].plot(time_axis, rolling_mean(mism), lw=1.4, color=colors[name], label=name)
n_exposures = len(SCHEDULE)
for i in range(1, n_exposures):
    axes[0].axvline(i * STEPS_PER_PATTERN * DT, color="black", lw=0.3, alpha=0.15)
axes[0].set_title("Mismatch (novelty signal), 150-step rolling mean, across the full training schedule")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Mismatch (rolling-mean # evaluator spikes)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.3)

labels = [f"S{int(s)}D{int(d)}A{int(a)}" for (s, d, a) in factorial]
retest_means = [factorial[c][0]["retest"] for c in factorial]
retest_stds = [factorial[c][1]["retest"] for c in factorial]
bar_colors = ["seagreen" if c[2] else "indianred" for c in factorial]
axes[1].bar(labels, retest_means, yerr=retest_stds, color=bar_colors, capsize=3)
axes[1].set_title(f"Retest mismatch, all 8 conditions ({N_SEEDS} seeds, mean +/- std)\n"
                   "(green = attention ON, red = attention OFF -- S/D/A = STDP/DPP/Attention)")
axes[1].set_ylabel("Mean mismatch")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(True, alpha=0.3, axis="y")

x = np.arange(2)
axes[2].bar(x - 0.15, [q_dpp, q_rand], width=0.3, label="mean quality (higher=better)", color="steelblue")
axes[2].bar(x + 0.15, [r_dpp, r_rand], width=0.3, label="mean redundancy (lower=better)", color="salmon")
axes[2].set_xticks(x)
axes[2].set_xticklabels(["DPP allocation", "Random allocation"])
axes[2].set_title("DPP's direct effect on allocation quality/diversity\n(construction-level; independent of network dynamics)")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/home/claude/dpp_art/ablation_dpp_attention_results.png", dpi=150)
print("Figure saved -> ablation_dpp_attention_results.png")
