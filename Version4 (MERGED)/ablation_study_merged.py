"""
ablation_study_merged.py
========================
Standardized factorial ablation suite for the Version 4 unified memory engine.
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_lib_merged import (build_art_network, run_schedule, probe_only,
                      well_separated_patterns, force_overlap)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
PATTERN_SIZE = 16
ACTIVE_BITS = 5
N_PATTERNS = 16           # > n_slots below, so late allocations are forced to reuse
SUBSTRATE_SIZE = 144
SLOT_SIZE = 12             # -> 12 "native" slots; patterns 12-15 force reuse
VIGILANCE = 0.6
BOOST = 4.0
STEPS_PER_PATTERN = 500
PROBE_STEPS = 120
N_SEEDS = 5
DT = 0.1

PATTERNS = well_separated_patterns(PATTERN_SIZE, N_PATTERNS,
                                    active_bits=ACTIVE_BITS, seed=7)
PATTERNS = force_overlap(PATTERNS, source_idx=0, target_indices=[3, 7, 11], frac=0.5)
NOVEL_PROBE = well_separated_patterns(PATTERN_SIZE, N_PATTERNS + 1,
                                       active_bits=ACTIVE_BITS, seed=7)[-1]

SCHEDULE = []
for i, p in enumerate(PATTERNS):
    SCHEDULE.append((f"p{i} 1st", p))
    SCHEDULE.append((f"p{i} retest", p))


def run_condition(use_stdp, use_dpp, use_attn, seed):
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn,
        use_stdp=use_stdp, seed=seed, boost=BOOST)
    exposures = run_schedule(system, SCHEDULE, steps_per_pattern=STEPS_PER_PATTERN,
                              dt=DT, probe_steps=PROBE_STEPS)

    firsts = [r["probe_mismatch"] for r in exposures if "1st" in r["label"]]
    retests = [r["probe_mismatch"] for r in exposures if "retest" in r["label"]]

    after_all = [probe_only(system, p, steps=PROBE_STEPS, dt=DT) for p in PATTERNS]
    interference = [a - r for a, r in zip(after_all, retests)]
    novel = probe_only(system, NOVEL_PROBE, steps=PROBE_STEPS, dt=DT)

    diag = system["art"].allocation_diagnostics()
    return {
        "first": float(np.mean(firsts)), "retest": float(np.mean(retests)),
        "after": float(np.mean(after_all)), "novel": float(novel),
        "mean_interference": float(np.mean(np.abs(interference))),
        "n_slots": len(system["art"].slots),
        "alloc_quality": diag["mean_quality"], "alloc_redundancy": diag["mean_redundancy"],
        "per_pattern": {"first": firsts, "retest": retests,
                         "after": after_all, "interference": interference},
    }


# =========================================================================
# 1. FULL 2x2x2 FACTORIAL, MULTIPLE SEEDS
# =========================================================================
print("=" * 96)
print(f"Full factorial ablation: STDP x DPP x Attention, {N_SEEDS} seeds each")
print(f"  pattern_size={PATTERN_SIZE} substrate_size={SUBSTRATE_SIZE} slot_size={SLOT_SIZE} "
      f"(-> {SUBSTRATE_SIZE // SLOT_SIZE} native slots for {N_PATTERNS} patterns: reuse is "
      f"forced, plus deliberate content overlap on patterns 3/7/11 with pattern 0)")
print("  first/retest/after/novel are all genuine pre-commit PROBE reads (see snn_lib.run_schedule")
print("  and probe_only) -- none of them are contaminated by the fast-commit that a novel")
print("  pattern's own allocation triggers.")
print("=" * 96)

header = (f"{'STDP':>5s} {'DPP':>5s} {'ATTN':>5s} | {'1st':>7s} {'retest':>7s} "
          f"{'after*':>7s} {'novel*':>7s} | {'slots':>5s} | {'q_dpp':>6s} {'redund':>6s}")
print(header)
print("-" * len(header))

factorial = {}
for use_stdp, use_dpp, use_attn in itertools.product([True, False], repeat=3):
    metrics = [run_condition(use_stdp, use_dpp, use_attn, seed) for seed in range(N_SEEDS)]
    agg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0] if k != "per_pattern"}
    std = {k: np.std([m[k] for m in metrics]) for k in metrics[0] if k != "per_pattern"}
    factorial[(use_stdp, use_dpp, use_attn)] = (agg, std, metrics)
    print(f"{str(use_stdp):>5s} {str(use_dpp):>5s} {str(use_attn):>5s} | "
          f"{agg['first']:7.4f} {agg['retest']:7.4f} {agg['after']:7.4f} {agg['novel']:7.4f} | "
          f"{agg['n_slots']:5.1f} | {agg['alloc_quality']:6.3f} {agg['alloc_redundancy']:6.3f}")
print("(* 'after' = specificity probe: every pattern re-tested once every other pattern has")
print(" since been learned, averaged. 'novel' = sensitivity probe: a pattern never trained on.)")

# -----------------------------------------------------------------------
# Main-effect summary
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

q_dpp = np.mean([factorial[c][0]["alloc_quality"] for c in factorial if c[1]])
q_rand = np.mean([factorial[c][0]["alloc_quality"] for c in factorial if not c[1]])
r_dpp = np.mean([factorial[c][0]["alloc_redundancy"] for c in factorial if c[1]])
r_rand = np.mean([factorial[c][0]["alloc_redundancy"] for c in factorial if not c[1]])
print()
print("DPP's direct, construction-level effect on WHICH neurons get allocated")
print("(independent of downstream spiking dynamics):")
print(f"  mean quality (higher=better):    DPP={q_dpp:.4f}  random={q_rand:.4f}")
print(f"  mean redundancy (lower=better):  DPP={r_dpp:.4f}  random={r_rand:.4f}")

# =========================================================================
# 2. PER-PATTERN BREAKDOWN, "FULL" CONDITION
# =========================================================================
print()
print("=" * 96)
print("Per-pattern breakdown, full condition (STDP+DPP+Attention), seed 0")
print("(patterns 3, 7, 11 were deliberately given 50% overlap with pattern 0's active bits;")
print(" patterns 12-15 were forced to reuse neurons since only 12 native slots exist)")
print("=" * 96)
full_metrics0 = factorial[(True, True, True)][2][0]
pp = full_metrics0["per_pattern"]
print(f"  {'pattern':8s} {'1st':>8s} {'retest':>8s} {'after-all':>10s} {'interf.':>8s}")
for i in range(N_PATTERNS):
    flag = " *overlap" if i in (3, 7, 11) else (" (reused slot)" if i >= 12 else "")
    print(f"  p{i:<7d} {pp['first'][i]:8.4f} {pp['retest'][i]:8.4f} "
          f"{pp['after'][i]:10.4f} {pp['interference'][i]:8.4f}{flag}")

# =========================================================================
# 3. TIME-SERIES RUNS FOR THE SUMMARY FIGURE (one seed each, for legibility)
# =========================================================================
def run_condition_series(use_stdp, use_dpp, use_attn, seed):
    """Re-implements run_schedule's per-exposure logic inline so the full
    per-tick mismatch trace can be kept (run_schedule only returns
    per-exposure aggregates, by design, to keep its return value small)."""
    system = build_art_network(
        pattern_size=PATTERN_SIZE, substrate_size=SUBSTRATE_SIZE, slot_size=SLOT_SIZE,
        vigilance=VIGILANCE, use_dpp=use_dpp, use_attention=use_attn,
        use_stdp=use_stdp, seed=seed, boost=BOOST)
    net, art = system["net"], system["art"]
    mem, ev = system["memory"], system["evaluator"]
    proj = system["proj_in_mem"]
    ssz = system["substrate_size"]
    series = []
    for label, pattern in SCHEDULE:
        pattern = np.asarray(pattern, dtype=float)
        needs_alloc, _ = art.route(pattern)
        if needs_alloc:
            art.allocate(pattern, proj.weights)
        # ownership for STDP gating must come from genuine match (match_scores),
        # not from attn -- attn is uniform when use_attention=False, and
        # np.argmax's tie-break would then always pick slot 0 regardless of
        # which slot actually matches (see snn_lib.run_schedule for the full
        # explanation; this loop duplicates that logic for series recording).
        if needs_alloc:
            owner_idx = len(art.slots) - 1
        else:
            raw_scores = art.match_scores(pattern)
            owner_idx = int(np.argmax(raw_scores)) if raw_scores.size else None
        proj.set_active_mask(art.owner_mask(mem.n, owner_idx) if owner_idx is not None else None)
        mism = np.zeros(STEPS_PER_PATTERN)
        for t in range(STEPS_PER_PATTERN):
            _, inhib = art.inhibition_for(pattern, mem.trace[:ssz], 20.0)
            ev_drive = np.maximum(pattern * 55.0 - inhib, 0.0)
            ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
            net.step({"Input": pattern * 180.0, "Evaluator": ext_ev}, dt=DT)
            mism[t] = ev.spiked[:PATTERN_SIZE].sum()
        series.append(mism)
    return np.concatenate(series)


print()
print("Running representative time-series conditions for the figure...")
series_conditions = [
    ("Full (STDP+DPP+Attn)", dict(use_stdp=True, use_dpp=True, use_attn=True)),
    ("No attention", dict(use_stdp=True, use_dpp=True, use_attn=False)),
    ("No STDP", dict(use_stdp=False, use_dpp=True, use_attn=True)),
    ("Nothing (all off)", dict(use_stdp=False, use_dpp=False, use_attn=False)),
]
series_data = {name: run_condition_series(seed=42, **kwargs) for name, kwargs in series_conditions}

# =========================================================================
# 4. FIGURE
# =========================================================================
fig, axes = plt.subplots(3, 1, figsize=(13, 12))
time_axis = np.arange(len(next(iter(series_data.values())))) * DT

def rolling_mean(x, window=150):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

colors = {"Full (STDP+DPP+Attn)": "green", "No attention": "crimson",
          "No STDP": "orange", "Nothing (all off)": "gray"}
for name, mism in series_data.items():
    axes[0].plot(time_axis, rolling_mean(mism), lw=1.4, color=colors[name], label=name)
for i in range(1, len(SCHEDULE)):
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
axes[1].set_title(f"Retest mismatch (probe-based), all 8 conditions ({N_SEEDS} seeds, mean +/- std)\n"
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
plt.savefig("ablation_results.png", dpi=150)
print("Figure saved -> ablation_results.png")