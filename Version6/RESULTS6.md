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

Added in this edition (see RESULTS.md for the evidence behind each)
  * retrieve() / retrieve_spiking(): rank slots for a PARTIAL pattern (excerpt). Symbolic containment
    matches the best achievable ranking; the spiking read-out is measurably worse.
  * use_learned_keys now min-max-normalizes learned receptive fields (they used to be raw weights up to 45,
    inconsistent with the Jaccard score and the inhibition scale); matching and inhibition use the same keys.
    Learned keys did NOT improve swap-noise detection (noise_fix_study.py): templates remain the default.
  * build_art_network(memory_recurrence=..., input_gain=...): see its docstring. The Memory E/I recurrent
    projection is a small, not zero, effect (removing it shifted retest by <= 0.008) and is the O(n^2) cost
    per tick; input_gain ~ 5/active_bits is required for patterns much denser than ~5 bits.
  * Read-out resolution: post-training probes should measure >= 600 ticks after the 400-tick warm-up; the
    old 200-tick window aliased noisy-known and novel readings (noise_fix_study.py).

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

Read-out: post-training probes use a 600-tick measured window (CFG_B); the earlier 200-tick results are kept in
results/benchmark_200tick.log. Usage: python benchmark_baselines.py [--seeds 12] [--out results]
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

Usage: python demo_scaled.py [--seeds 8] [--out results]






diagnose_identical.py -- why did "No attention" and "Nothing" give identical numbers in demo_scaled?

Hypothesis: with uniform (unrouted) inhibition over S slots, the cancellation signal on any bit is
~ strength * activity / S (only the matching slot is active), far below the ~35 units needed to silence an
evaluator channel (drive 55 vs threshold 20). Then the evaluator just sees its raw drive and the reading is
the same whatever STDP/DPP did. Checks, at demo scale (480 excitatory, 48 patterns, seed 0):
  1. the per-pattern read-only curves of the two configurations are exactly equal;
  2. in the no-attention network the peak inhibition on a trained pattern's bits vs the silencing level (35);
  3. that curve equals the reading of a network with NO slots at all (raw un-inhibited drive);
  4. with attention ON the inhibition does exceed the silencing level.
Usage: python diagnose_identical.py [--demo-json results/demo_scaled_raw.json]






excerpt_retrieval.py -- "which learned pattern does this fragment come from?"

Train the detector on the toy patterns (2 exposures each, read-only from-rest probes unaffected),
then give it FRAGMENTS of trained patterns and ask for the source pattern's slot.
Fragments: k = 1..4 random active bits of the pattern (everything else unobserved), and a "column"
fragment: a contiguous block of 6 observed positions (values incl. zeros), requiring >= 1 active bit.
Methods
  containment      SlottedARTMemory.retrieve (masked, symbolic; fraction of observed active bits inside a template)
  naive Jaccard    existing match_scores with the fragment treated as a full pattern (unobserved = off)
  spiking          retrieve_spiking: rank slots by firing of their neurons when the fragment drives the network
  ceiling          best achievable top-1 given ambiguity: expected accuracy of a uniform guess among every
                   template that contains the fragment (1.0 only if the fragment is unique to one template)
Top-1 / top-3 accuracy, mean [95% bootstrap CI over seeds]. Several fragments per pattern per seed.

Usage: python excerpt_retrieval.py [--seeds 8] [--out results]






noise_fix_study.py -- why did the SNN fail on swapped-bit probes, and what fixes it?

Hypothesis tested (and REJECTED): the inhibition/matching keys (exact stored templates) are the cause,
so learned, normalized keys should help. Alternative: the post-training read-out window (200 measured
ticks) is too short -- readings are quantized (a single rebound = 0.025) and a noisy known pattern and a
novel pattern often tie or alias. Variants (same tasks, matched seeds; AUROC novel vs noisy-known):

  A200..A1000  templates (current keys), read-out window of 200 / 400 / 600 / 1000 measured ticks
  B600         learned normalized keys used only at probe time (600 ticks)
  C600         learned normalized keys during training AND probing (600 ticks)

Usage: python noise_fix_study.py [--seeds 8] [--out results]






probe_contamination_demo.py -- direct evidence for todo

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






scale_check.py -- how does the library behave at larger input / substrate sizes?

For each (pattern_size, substrate_size): build, train 3 patterns, then report build time,
ms per simulated tick (with and without the Memory E/I recurrent projection), peak
resident memory, and a regime sanity check from read-only probes: the fraction of excitatory
substrate neurons that fire for a TRAINED pattern (selectivity: should be ~slot_size / n, not
~100%) and trained-vs-novel evaluator mismatch.
Each config is run with input_gain=1 (original calibration) and input_gain = 5/active_bits.

Usage: python scale_check.py [--out results]






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






# Results and change notes (noise, speed, scale, retrieval, further loose ends)

Numbers come from the logs in `results/`. One CPU core. Seeds: ablation 10, benchmark 12, scaled demo 8,
noise-fix study 8, excerpt retrieval 8. Every seed regenerates patterns, novel probes and weights.
Read-out: probes run from rest with 400 discarded ticks; the post-training read-out window is 600 measured
ticks in the benchmark and demo (200 in the ablation, whose conclusions do not depend on swap-noise resolution).

## Library changes (snn_lib_changing.py)
- Probes are read-only and history-free (snapshot -> reset to rest -> warm-up -> measure -> restore); a
  `legacy` mode is kept only to measure the old behaviour. One shared tick loop; traces can be recorded.
- DPP flags split: use_dpp = greedy MAP only; use_reuse_damping and use_overlap_penalty are separate.
- Learned keys are now min-max normalized; matching and inhibition use the same keys (templates stay the default).
- retrieve() (masked containment) and retrieve_spiking() for excerpt retrieval.
- build_art_network(memory_recurrence=True, input_gain=1.0): see docstring.
- ~2x faster per run (7.9 s -> 3.8 s) with results identical to 1e-12 on the cached ablation runs checked.
- Validation, fallback-allocation warning, vectorized routing, faster pattern generator (output unchanged).
- 24 tests (tests/test_snn_lib.py) pass.
- `steps_per_pattern` now counts only learning time (500 vs 380 ticks before); run_schedule clears the plasticity mask.

## Results
**Probe contamination.** Legacy probes always left state changed; 44% of readings changed when probe order was
reversed; repeated identical probes varied by 0.025. Fixed probes: none of these.

**Ablation (10 seeds; primary metric = retest mismatch, declared before running; 200-tick read-out).**
Attention -0.0172 (~66% lower), STDP -0.0061 (~30%), DPP bundle -0.0017 (~9%). The metric is quantized and almost
deterministic across seeds, so p-values / d_z are uninformative; use absolute sizes. DPP decomposition: greedy
selection alone lowers retest 0.0063 -> 0.0006; reuse damping + overlap penalty add nothing detectable.
Legacy vs fixed pipeline (confounded with probe length): attention -0.0259 vs -0.0172, STDP -0.0036 vs -0.0061,
DPP -0.0006 vs -0.0017; ranking unchanged.

**Noise weakness.** The earlier benchmark showed the SNN failing on swapped-bit probes. Cause: NOT the keys but the
short post-training read-out window (readings are quantized; noisy-known and novel readings aliased).
AUROC novel-vs-noisy-known, templates, 8 seeds: 200 ticks swap1 0.672 / swap2 0.395; 400 ticks 0.976 / 0.776;
600 ticks 0.999 / 0.867; 1000 ticks 1.000 / 0.933. Learned normalized keys were worse than templates at the same
window (0.971 / 0.78 at 600). The 600-tick window was chosen after seeing the 200-tick failure, and the benchmark
seeds overlap the study seeds: mild tuning-on-test risk. Longer windows make each probe slower.

**Benchmark (12 seeds, 600-tick read-out; primary = mean AUROC over clean / swap1 / swap2 / drop2).**
Template matcher 0.998; ART-1 (rho 0.8) 0.971; SNN full 0.969; ART-1 (rho 0.6) 0.965; SNN no-attention 0.796;
SNN nothing 0.626. SNN full vs template matcher -0.030 (all of it swap2: 0.875 vs 0.993; clean, swap1, drop2 tie);
vs ART-1 statistically tied (+0.004, p=0.87; -0.003, p=0.61). Attention is worth +0.173; random allocation vs
full DPP +0.003 (p=0.28). Sweeps: SNN full is below the template matcher at every level, mostly by 0.005-0.034;
by density the gap is -0.147 (3 bits), -0.030 (5), -0.079 (8), not monotone; at 8 bits SNN full beats ART-1
rho=0.6 (0.919 vs 0.809). The earlier 200-tick results (SNN 0.768) are in results/benchmark_200tick.log and
were largely a read-out artifact. p floor with 12 seeds is ~0.0005.

**Scaled demo (480 excitatory, 48 patterns, 8 seeds).** Full: retest 0.0001, AUROC 1.000. No DPP: retest 0.0006
(+0.0006 vs Full, p=0.008), AUROC 1.000. No attention 0.0412 and Nothing 0.0413: AUROC 0.458 for both.

**Identical no-attention / Nothing result (loose end 7).** With the 200-tick read-out the curves were exactly equal;
with 600 ticks they differ slightly (max per-pattern diff 0.005, mean 0.0412 vs 0.0413). Without attention the peak
inhibition on a trained pattern's bits is 4.8-10.8 vs ~35 needed to silence a channel (36-52 with attention), so
STDP/DPP differences barely reach the output and quantization hid what remained. Consistent with, not proven by,
the diagnosis (proxy network: STDP off, DPP on).

**Scale (up to 256-bit patterns, 1920 excitatory neurons; scale_check.log).** With input_gain=1, 98-100% of
substrate neurons fire for a trained pattern once patterns have >= 13 active bits (selectivity lost);
input_gain = 5/active_bits restores it (2.5% / 1.2% / 0.6% at 480 / 960 / 1920). Tick cost (recurrence on vs
off): 0.25 vs 0.11 ms at 480; 4.4 vs 0.32 ms at 1920 (timings vary run to run). Memory 227 MB at 1920.

**Recurrent E/I circuit.** The inhibitory pool never fires and E->E recurrent drive to slot neurons is
<= 11% of external drive, shrinking ~1/n (regime_check.log). It is NOT strictly inert: removing it changed 10 of 12
checked runs (e.g. Full retest 0.000 -> 0.003-0.008; one AUROC moved by 0.0625). Decision: default stays on;
memory_recurrence=False is an opt-in speed option; documented.

**Excerpt retrieval (8 seeds, excerpt_retrieval.log).** Top-1 for 1/2/3/4-bit fragments: containment 0.20/0.63/0.97/1.00
(equal to the best achievable ranking given ambiguity); contiguous 6-position block: containment 0.88 vs plain
Jaccard 0.61. Spiking retrieval is worse than containment: -0.133 (3 bits), -0.273 (4 bits), -0.281 (block), p=0.008
(floor). The block "ceiling" ignores zero-valued evidence, so it is not a true ceiling there.

## Limits
Synthetic binary patterns; coarse spike-count read-out; legacy-vs-fixed confounded with probe length; the
ablation uses the 200-tick read-out; the spiking system does not beat a plain template matcher on any tested
condition; real data and temporal data are untested.
