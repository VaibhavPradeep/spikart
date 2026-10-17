snn_lib_merged.py -- general-purpose spiking network library (merged edition)
========================================================================

This merges two parallel generations of the same project:

  Generation 1 (flat substrate): LIFPopulation / Projection / Network,
  a "hand-rolled" greedy DPP (Gram-Schmidt on a quality-weighted feature
  matrix) wrapped in an AllocationController with PERSISTENT cross-call
  orthogonal-direction tracking, and an AttentionRoutedProjection that
  recomputed full dot-product attention every tick from the *currently
  learned* Input->Memory weights.

  Generation 2 (slotted ART): the same base layer, but allocation and
  routing are lifted out of the Projection machinery into an explicit
  SlottedARTMemory -- symbolic (Jaccard) vigilance test against frozen
  slot templates, a properly-cited greedy DPP MAP algorithm (Chen, Zhang
  & Zhou, 2018), and per-slot diagnostics (quality/redundancy) that are
  measured independent of the network's spiking dynamics.

Design decision: rather than keep two parallel allocation architectures
side by side, this file adopts Generation 2's SlottedARTMemory as the
base -- it is the cleaner abstraction (explicit categories, honest
diagnostics, symbolic vigilance that's O(1) in the network's spike
dynamics rather than requiring a live probe) -- and folds in the parts of
Generation 1 that were genuinely better:

  1. Persistent cross-slot orthogonal-direction projection (hard
     diversity), layered ON TOP OF Gen 2's reuse-count damping (soft
     diversity), instead of either alone.
  2. active_mask-gated STDP (Projection.set_active_mask), so learning on
     any given presentation is restricted to the slot that "owns" the
     pattern, instead of leaving every substrate neuron plastic and
     open to incidental drift from unrelated patterns.
  3. An optional use_learned_keys mode so slot matching can track the
     *current* learned receptive field of a slot's neurons rather than
     only the frozen template captured at allocation time (Gen 2's
     stated "no template drift" simplification, made configurable
     rather than fixed).
  4. force_overlap(): Gen 1's habit of deliberately engineering pattern
     overlap for interference stress-testing, ported into a reusable
     helper that composes with Gen 2's well_separated_patterns().

  5. A genuine pre-commit "probe" measurement window in run_schedule(),
     fixing the same bug Gen 1's own commentary flagged and repaired
     once already ("the '1st exposure' numbers are measured after the
     fast-commit already fired") -- which Gen 2's per-tick allocate-then-
     measure loop had quietly reintroduced.

Not merged: Gen 1's build_multi_column_population() (independent,
block-diagonal cortical columns) is dropped. To be precise about what
that costs: a slot's member neurons are a logical/bookkeeping grouping
on top of ONE still-flat, still-shared recurrent E/I matrix -- the
substrate's internal_weights() is computed once over the whole
population, so this is not the same as genuine dynamical isolation
between categories that block-diagonal columns would have given.
The narrower, honest claim: mean-field-normalized recurrent weights
were sufficient to keep the shared substrate well-behaved at the scales
actually tested here (up to ~600 neurons, 48 slots) -- not that the
block-diagonal machinery is redundant in general, just that it wasn't
necessary to demonstrate this merge. A setup that needed hard dynamical
separation between categories (rather than the softer slot-level
bookkeeping SlottedARTMemory provides) would still want it back.






ablation_study_merged.py -- merged edition
====================================

Evolved ablation script for the ART-style, DPP-allocated, attention-routed
spiking novelty detector (snn_lib.py). Direct descendant of the original
two-column "LTP ON vs OFF" ablation: same core idea -- run matched
conditions off the same seeds and measure the difference each mechanism
actually makes -- generalized to a full 2x2x2 factorial over three
switches, each run across several seeds so effects are real means +/- std
rather than single-seed anecdotes:

    STDP      : real pair-based spike-timing plasticity
    DPP       : diverse, quality-weighted neuron selection at allocation
                time, vs. uniform random selection
    Attention : softmax-routed inhibition to the evaluator, vs. naive
                uniform averaging across every committed slot

What changed relative to the version this was forked from:

  * "1st exposure" and "retest" are now measured via the genuine
    pre-commit PROBE window run_schedule() adds (see snn_lib.py) --
    average mismatch over the whole exposure window, sampled starting at
    tick 0, is contaminated by the very fast-commit boost that a novel
    pattern triggers at tick 0; the probe reads the network strictly
    BEFORE that commit happens. This is the same fix an earlier version
    of this project made once already (measuring after fast-commit had
    already fired, silently deflating "novelty" numbers) -- the per-tick
    allocate-then-measure loop this script used to run had quietly
    reintroduced the same problem, so it's fixed at the library level
    now instead of per-script.

  * "after all learned" and "never trained" readings now use
    snn_lib.probe_only() -- a genuinely zero-side-effect readout (hard-
    disables learning, no allocation) -- instead of folding those checks
    into the training schedule itself.

  * The pattern set combines BOTH interference-stress strategies the two
    prior versions used separately: capacity-forced reuse (more patterns
    than native slots, so late allocations are forced to share neurons)
    AND deliberately-engineered content overlap (a few patterns forced to
    share half of pattern 0's active bits) via force_overlap(). Purely
    well-separated random patterns rarely overlap enough to stress
    interference at all; this makes sure the ablation actually has
    something to detect.

  * A per-pattern breakdown table for the "full" condition (all three
    switches on) reports every pattern's 1st / retest / after-all /
    interference individually, not just the factorial's aggregate means
    -- useful for spotting whether interference is concentrated on the
    patterns that were deliberately given overlap, or spread evenly.

Honest calibration note (read before citing numbers from this script):
attention is close to *necessary* once more than a handful of slots are
committed -- without it, the correct slot's cancellation signal is
diluted by every other slot regardless of relevance, so its effect on
retest mismatch is usually the largest and most reproducible of the
three. DPP's effect on these AGGREGATE behavioural metrics is often much
smaller and noisier -- the additive resonance boost + attention-routed
exact-template matching already does most of the "solving" once a slot
exists. DPP's real, measurable benefit shows up in allocation_diagnostics
(mean quality / mean redundancy of WHICH neurons got picked), reported
separately below -- that is reported as a finding, not hidden.






demo_scaled_merged.py -- merged edition
==================================

Scalability showcase for the snn_lib ART-style slotted novelty detector.
Where ablation_study.py runs a small, statistically-powered factorial to
isolate each mechanism's marginal effect, this script runs the SAME
underlying library at a much larger scale and reports a CAPACITY CURVE:
after training every pattern once, each pattern is re-tested in its
original training order, so degradation from later-learned patterns
crowding the substrate (interference on EARLIER patterns specifically) is
visible directly, rather than averaged away.

Scale, for reference:
    original two-column script     : ~20 neurons, 2 categories
    ablation study (this project)  : ~180 neurons, up to 16 categories
    this demo                      : ~600 neurons, up to 48 categories

Compares 4 representative configurations at one seed each:
    Full          : STDP + DPP allocation + attention-routed inhibition
    No DPP        : STDP + random allocation + attention-routed inhibition
    No attention  : STDP + DPP allocation + uniform (unrouted) inhibition
    Nothing       : STDP off, random allocation, uniform inhibition

What changed relative to the version this was forked from:

  * A handful of patterns are deliberately given overlapping active bits
    with pattern 0 via force_overlap(), on top of well_separated_patterns()'
    default (which alone tends to produce patterns so mutually dissimilar
    that content-driven interference rarely occurs by construction --
    only capacity-driven reuse would show up). This borrows the earlier,
    smaller two-column demo's habit of hand-engineering some overlap, now
    applied at this script's much larger scale.

  * Both the capacity-curve retest and the never-trained sensitivity probe
    now go through snn_lib.probe_only(), which hard-disables learning for
    the read. The version this was forked from prevented new allocation
    during these reads but left ordinary STDP free to keep adjusting
    weights while "just measuring" -- a genuinely clean read shouldn't
    change anything at all.

  * Build time is reported per configuration (a cheap, informative
    number an earlier, smaller version of this demo used to print but
    the scaled-up version had dropped).

  * Each configuration's first-exposure PROBE mismatch (genuine pre-
    commit novelty, from run_schedule -- see snn_lib.py) is recorded
    during training and plotted alongside the retest curve, so you can
    see whether the deliberately-overlapping patterns look *less* novel
    on first contact than their well-separated neighbors (they should --
    that's the point of giving them shared bits with an already-learned
    pattern).