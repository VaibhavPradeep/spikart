snn_lib_changing.py -- general-purpose spiking network library (unified edition)
================================================================================

ART-style, slot-allocated, attention-routed spiking novelty detector.

Layers
  LIFPopulation / Projection / Network   -- spiking substrate (current-based
                                            synapses, optional pair-based STDP)
  greedy_dpp_map*                        -- greedy determinant-maximizing
                                            selection (Chen, Zhang & Zhou, 2018)
  AttentionRouter                        -- softmax routing of inhibition
  SlottedARTMemory                       -- Jaccard vigilance test, slot
                                            allocation, routed inhibition
  build_art_network / run_schedule /
  probe_only                             -- experiment helpers

Terminology (kept consistent everywhere)
  * "Greedy DPP MAP selection" is the ESTABLISHED algorithm: greedy residual-
    norm maximization on a quality-weighted feature matrix B (implied kernel
    L = B B^T). Implemented twice (kernel form, feature form); tests check the
    two agree.
  * Two SOFT quality modifiers are OUR additions, not part of the standard
    algorithm, and are now independently switchable:
        reuse damping     q <- q / (1 + times_claimed)
        overlap penalty   q <- q * 1 / (1 + overlap_energy_with_earlier_slots)
    Neither projects anything out; both only discount DPP quality.
  * Also non-standard: random top-up when the greedy selection stops early
    (rank(B) <= pattern_size), and the additive "resonance boost" applied to
    the chosen rows after selection (runs whatever the selection method).

State semantics 
  * Dynamical state = voltages, refractory timers, spike flags, activity
    traces, synaptic current traces (psc), plastic-eligibility masks, and the
    attention router's last weights. Synaptic weights, slots and RNG state are
    "learned state" and are only changed by run_schedule()/allocate().
  * probe_only(state_mode="rest") is a genuinely read-only measurement: it
    snapshots all state, resets the network to rest, optionally runs a short
    discarded warm-up with the test pattern, measures, and restores the
    snapshot. Because it always starts from rest, a probe cannot depend on
    what was presented before it, and repeated identical probes are bit-for-bit
    identical. state_digest() fingerprints everything for tests.
  * run_schedule(state_mode="rest") starts every exposure from rest, runs the
    pre-commit probe from that rest state, returns to the SAME rest state, then
    commits/learns for `steps_per_pattern` ticks. Probe and main window
    therefore start from identical initial conditions, and the probe costs no
    learning time.
  * state_mode="legacy" reproduces the pre-fix behaviour (probes advance and
    leave the state they ended in; main window continues from the probed
    state; the probe is carved out of steps_per_pattern). It is kept ONLY so
    the size of the old contamination can be measured.

Not merged: block-diagonal cortical columns (see earlier notes). Slots are a
bookkeeping grouping on ONE flat, shared recurrent E/I matrix.





ablation_study.py -- statistically-powered ablation
==========================================================================================

Sections
  1. 2x2x2 factorial (STDP x DPP x Attention), matched seeds, per-seed effects
     with bootstrap CIs, d_z, sign-flip p, Holm correction.
       PRIMARY metric (declared before the new runs): retest mismatch.
       Everything else is secondary / exploratory.
       "DPP" here = the whole bundle (greedy MAP + reuse damping + overlap
       penalty), exactly as before.
  2. DPP decomposition: at STDP on / Attention on, random vs greedy
     MAP only vs +reuse damping vs +overlap penalty vs both. Paired comparisons
     on downstream metrics AND construction-level diagnostics.
  3. Effect of the pre-fix state handling: the same factorial in
     state_mode="legacy" vs "rest", to measure how much probe contamination
     moved the numbers.
  4. Per-pattern table (seed 0, full condition).
  5. Figures. The time-series figure now comes from run_schedule itself
     (same loop as training), with the pre-commit PROBE and the post-commit
     main window drawn separately.

Every seed regenerates the pattern set, novel probes and network initialization,
so seed-to-seed variance includes benchmark variance, not just init noise.

Usage:  python ablation_study.py [--seeds 10] [--out results] [--skip-legacy]
Raw per-run results are cached in <out>/ablation_raw.json (resumable).






experiment_core.py -- shared task generation, SNN runner and spike-free
baselines used by ablation_study.py, contamination_check.py and
benchmark_baselines.py. One implementation, so scripts cannot drift apart.

A "task" is generated PER SEED (pattern set, novel probes, noisy variants,
training order all depend on the seed), so matched-seed conditions see
identical data and seed-to-seed variation includes benchmark variation, not
just network initialization.






benchmark_baselines.py -- baselines and benchmark variety
====================================================================

Question: does the spiking ART detector beat -- or even need -- simpler
novelty detectors, and how does it degrade as the benchmark gets harder?

Task (identical for every method; one task per seed)
  Train on N patterns (each shown twice, in the swept order). Then score
  * KNOWN probes : trained patterns, either CLEAN (the exact stored template)
                   or NOISY (not the stored template): swap1 / swap2 (replace 1 / 2
                   active bits with inactive ones) and drop2 (occlude 2 active bits)
  * NOVEL probes : 8 never-trained patterns (the generator's last, i.e. most
                   overlapping-with-training, patterns -> hard negatives)
  Metric: AUROC = P(score(novel) > score(known)). 1.0 = perfect separation,
  0.5 = chance. PRIMARY (declared before running): mean AUROC over the four
  known-probe conditions {clean, swap1, swap2, drop2}. The noisy conditions are
  the part of the benchmark that does NOT reuse the architecture's own stored
  templates verbatim.

Methods (all see the same training sequence and the same probes)
  SNN full / no-attention / random-allocation / nothing  -- the spiking system
      (score = evaluator mismatch from a read-only probe)
  Template matcher -- spike-free symbolic Jaccard, same vigilance/allocation
      rule as the architecture's route(); score = 1 - best Jaccard
  ART-1 (rho=0.6, 0.8) -- classic binary ART-1, fast learning, subset-match
      vigilance, template intersection; score = 1 - best vigilance match

Sweeps (one factor at a time around the default cell: 5 active bits of 16,
overlap 0.5, 16 patterns / 12 native slots, fixed order)
  density: 3 / 5 / 8 active bits;  overlap: none / 0.5 / 0.8 of pattern 0's bits
  force-shared into patterns 3, 7, 11 (bits are ADDED, so patterns grow);
  capacity: 8 / 16 / 24 patterns (> 12 forces neuron reuse);
  order: fixed / random / reverse training order.

Usage: python benchmark_baselines.py [--seeds 6] [--out results]
Resumable (results/benchmark_raw.json). Seeds are matched across methods.






demo_scaled.py -- capacity curve at ~600 neurons / 48 categories (updated)
==========================================================================

Same scale showcase as before (24-bit patterns, 480-excitatory substrate = 40
native slots, 48 patterns), updated for the fixed library:

  * imports snn_lib_changing (the unified library);
  * every read (first-exposure probe, capacity-curve retest, never-trained
    probes) is a read-only, from-rest probe -- the capacity curve no longer
    carries state from one retest into the next, so "interference as a function
    of training order" is not confounded with probe order;
  * several seeds, each with its own pattern set; summary numbers are
    mean [95% bootstrap CI over seeds]; the capacity curve is the seed-mean
    with a CI band;
  * 8 never-trained probes instead of 1, summarized as an AUROC
    (novel vs trained) in addition to the mean mismatch;
  * build time is reported per configuration.

Compares 4 configurations: Full / No DPP (random allocation) / No attention /
Nothing. At this scale the Memory substrate's recurrent E/I connectivity is
nearly inert (see regime_check.py), so these results characterize the
slot/attention machinery, not recurrent dynamics.

Usage: python demo_scaled.py [--seeds 3] [--out results]






probe_contamination_demo.py --

Takes one trained network and compares the OLD probe behaviour (state_mode="legacy")
with the fixed one ("rest") on three tests that a read-only probe must pass:
  1. REPEATABILITY: probing the same pattern 5 times in a row gives one value.
  2. ORDER INDEPENDENCE: a pattern's reading must not depend on what was probed
     just before it.
  3. NO SIDE EFFECTS: probing must leave the full system state unchanged.
Legacy probes use the original 120-tick window; rest probes use 400 warm-up + 200 measured.

Usage: python probe_contamination_demo.py [--seeds 3]






regime_check.py -- is the mean-field-normalized recurrent regime actually healthy?

internal_weights() scales per-connection recurrent weights by ref_n/n (reference
8 excitatory / 2 inhibitory), so aggregate recurrent drive is constant BY
CONSTRUCTION. This script MEASURES what the substrate does at several sizes
instead of assuming it: firing rates, fraction of neurons active, and the
currents each excitatory neuron receives (external Input->Memory vs recurrent
E->E vs recurrent I->E), plus the Evaluator's un-inhibited firing.

All reads are probe_only() reads from rest (400 warm-up + 300 measured ticks).
Units: rates in spikes/neuron/ms; currents in the model's units (a neuron needs
a steady current of v_th - v_rest = 20 to reach threshold when R=1).
Flags: DEAD (<0.002/ms), SATURATED (>0.4/ms; hard limit is 0.5/ms from the 2 ms refractory period).

Usage: python regime_check.py [--out results]






stats_utils.py -- paired / factorial statistics for the ablation.

Design rules (declared BEFORE looking at any new results):
  * Primary metric: retest mismatch (mean pre-commit probe mismatch on the
    second exposure of each pattern; lower = better recognition).
    Everything else (first, after, novel, AUROC, interference, allocation
    diagnostics) is secondary/exploratory and is labelled as such.
  * Seeds are the unit of replication. Conditions use MATCHED seeds (same
    patterns, same initial weights), so every comparison is paired.
  * Factorial effects are computed PER SEED, then summarized across seeds:
    mean, percentile-bootstrap 95% CI, standardized effect size d_z
    (mean / sd of the per-seed effects), and a paired sign-flip permutation
    p-value. The 7 effects of the primary metric are Holm-adjusted together.
  * Percentile bootstrap CIs are anti-conservative for very few seeds (<8);
    prefer more seeds over a smaller p-value.




















# Results and change notes

All numbers come from the logs in `results/` (ablation.log, benchmark.log, demo_scaled.log,
regime_check.log, probe_contamination.log, tests.log). Single CPU core; 10 seeds for the ablation,
6 for the benchmark, 3 for the scaled demo. Every seed regenerates patterns, novel probes and weights.

## What changed in the library (snn_lib_changing.py)
- snapshot_state / restore_state / state_digest; probe_only(state_mode="rest") = snapshot -> reset to
  rest -> 400 discarded ticks -> measure -> restore. run_schedule(state_mode="rest") starts each exposure from rest,
  probes from rest, returns to the same rest state, then learns for steps_per_pattern ticks (the probe is extra,
  read-only time; legacy carved the probe out of the 500 ticks, so learning time is 500 vs 380).
  state_mode="legacy" is kept only to measure the old behaviour. run_schedule clears the plasticity mask at the end.
- one shared tick loop (_run_window) used by training, probing and the figure code; run_schedule can record
  warm-up / probe / main traces. The figure is built from it.
- use_dpp = greedy MAP selection only; use_reuse_damping and use_overlap_penalty are separate flags
  (default: follow use_dpp, so old bundled behaviour is unchanged).
- Also: input validation, fallback-allocation warning, vectorized routing (tests check equality with the loop
  versions), faster well_separated_patterns (tests check identical output to the original), per-tick hook.

## Evidence
- Probe contamination (probe_contamination.log): legacy probes always left state changed, readings of 44% of probes
  changed when probe order was reversed, repeated identical probes varied by 0.025. Fixed probes: 0 / 0 / 0.
- Ablation (primary metric = retest mismatch, declared before running; lower is better):
  attention -0.0172 (about 66% lower), STDP -0.0061 (about 30%), DPP bundle -0.0017 (about 9%). All 7 factorial
  effects reach the minimum possible p for 10 seeds (Holm 0.0137), but the metric is quantized (steps of 0.005, one
  rebound = 0.025) and nearly deterministic across seeds, so p-values and d_z are not informative; use absolute sizes.
  The novel-probe reading is 0.025 in every condition (uninformative).
- DPP decomposition: greedy selection alone lowers retest 0.0063 -> 0.0006 (paired -0.0058, CI [-0.0072,-0.0042]);
  reuse damping + overlap penalty add nothing detectable (-0.0003, CI [-0.0007,+0.0001]). alloc_quality rises by
  construction and is not evidence of better recognition.
- Legacy vs fixed pipeline (confounded with probe length): attention -0.0259 vs -0.0172, STDP -0.0036 vs -0.0061,
  DPP -0.0006 vs -0.0017; ranking unchanged.
- Regime check: the Memory inhibitory pool never fires at any size; slot neurons fire ~0.1 spikes/ms at all
  sizes (nothing dead or saturated); E->E recurrent current for slot neurons falls from 0.11 of external drive (12
  neurons) to 0.003 (480) because only the 12 slot neurons are active while weights scale as 1/n. The recurrent E/I
  circuit is functionally inert, so results reflect slot/attention/STDP machinery, not recurrent dynamics.
  Probe caveat: lightly trained slots show sparse rebound spikes every ~600-900 ticks.
- Benchmark, primary AUROC (mean of clean/swap1/swap2/drop2): Template matcher 0.996, ART-1 0.975/0.971,
  SNN full 0.768, SNN no-attention 0.544, SNN nothing 0.540, SNN random-alloc 0.767. The SNN ties the template
  matcher on clean and drop2 probes but is far worse on swap noise (swap2 AUROC 0.40, below chance). Hypothesis,
  untested: inhibition is built from exact stored templates. SNN full vs random-alloc: +0.001, p=1.0. The gap
  to the template matcher shrinks with density (-0.36, -0.23, -0.06 for 3/5/8 bits) and barely moves with
  overlap, capacity or order. With 6 seeds p=0.031 is the floor.
- Scaled demo (480 excitatory, 48 patterns, 3 seeds): Full retest 0.0000 flat across the capacity curve, AUROC 1.0;
  No-attention and Nothing are identical (retest 0.031, AUROC 0.458); random allocation is indistinguishable from
  Full. Not investigated: why No-attention and Nothing match to 4 decimals.

## Limits
Synthetic binary patterns; the SNN scores come from a coarse spike-count readout; legacy-vs-fixed comparison is
confounded with probe length; 3-6 seeds in the benchmark and demo.
