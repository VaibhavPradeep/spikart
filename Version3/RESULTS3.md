snn_lib_3.py -- general-purpose spiking network library.

Base layer (unchanged from the previous generalization pass):
    LIFPopulation, Projection, Network
      - vectorized LIF neurons, generic weighted projections between any two
        populations, pluggable plasticity (None / 'hebbian' legacy / 'stdp'
        real pair-based trace STDP), current-based synapse filtering.

New layer (this version) -- turns the two-column novelty detector into a
generic, scalable, ART-style multi-category ("slotted") novelty detector:

    greedy_dpp_map     - fast greedy MAP inference for a k-DPP (Chen,
                          Zhang & Zhou, 2018). Used to pick a *diverse* set
                          of substrate neurons when a new category needs to
                          be allocated, instead of grabbing them uniformly
                          at random.
    AttentionRouter     - softmax attention over per-slot match scores; used
                          to route inhibitory ("this is already explained")
                          signal from memory slots into the mismatch
                          evaluator selectively, rather than dumping every
                          slot's inhibition in indiscriminately.
    SlottedARTMemory    - the allocation/vigilance/routing logic: decides
                          when an input needs a *new* memory category
                          (ART vigilance test), grows that category out of
                          the shared neuron substrate (DPP or random), and
                          produces the routed inhibitory current for the
                          evaluator population each tick.

build_art_network() / run_schedule() wire all of the above into a runnable
experiment so that both the ablation study script and the scaled-up demo
can share the exact same underlying mechanics and just vary configuration
(scale, which factors are switched on, how many seeds).

Modeling simplifications, stated plainly:
  - Slot "templates" are the pattern that triggered allocation, fixed
    afterwards (no template drift) -- a simplified ("fast-learning") ART.
  - The vigilance/attention match score is computed directly against the
    input pattern rather than emerging purely from network dynamics; this
    stands in for the fast top-down category-search pass ART models run
    before committing to (or forming) a category.
  - DPP quality/diversity features are the neurons' current Input->Memory
    weight rows -- a proxy for "preferred input direction", not a learned
    embedding.
These are simplifications made for a tractable, runnable demonstration of
the *mechanisms* (diversified allocation, attention-gated routing, ART
vigilance), not a claim of full biological fidelity.






ablation_dpp_attention_3.py
==========================
Evolved ablation-study script for the ART-style, DPP-allocated,
attention-routed spiking novelty detector (built on snn_lib.py).

This is the direct descendant of the very first two-column ablation script
(LTP ON vs OFF): it keeps that script's core idea -- run matched conditions
off the same seeds and measure the difference plasticity/mechanisms
actually make -- but now the "column" is a scalable, many-slot ART memory,
and the factorial covers THREE switches instead of one:

    STDP        : real pair-based spike-timing plasticity within a slot
    DPP         : diverse, quality-weighted neuron selection at allocation
                  time, vs. uniform random selection
    Attention   : softmax-routed inhibition to the evaluator, vs. naive
                  uniform averaging across every committed slot

That's a 2x2x2 = 8-condition factorial, run over several seeds each, so
main effects are actual means +/- std rather than single-seed anecdotes.

Metrics per condition:
    1st exposure mismatch      - baseline novelty response
    retest mismatch            - habituation (recognition after 1 exposure)
    "after all" mismatch       - SPECIFICITY: re-test pattern 0 after every
                                  other pattern has since been learned
                                  (this is the interference probe that the
                                  very first multi-pattern demo in this
                                  project surfaced as its key failure mode)
    novel-probe mismatch       - SENSITIVITY: a pattern the network has
                                  NEVER seen, tested at the very end
    allocation quality/redundancy - a direct, deterministic measure of
                                  what DPP vs. random selection actually
                                  bought at allocation time, independent
                                  of how the rest of the network's spiking
                                  dynamics happen to play out

Honest calibration note (read before citing numbers from this script):
at this scale, attention and STDP produce large, robust, easily
reproduced effects -- attention in particular is close to *necessary*
once more than a handful of slots are committed, since without it the
correct slot's cancellation signal is diluted by every other slot
regardless of relevance. DPP's effect on these AGGREGATE behavioural
metrics is small and often within noise here: the additive resonance
boost + attention-routed exact-template matching already does most of
the "solving" once a slot exists, so DPP's real, measurable benefit
(see allocation_diagnostics) mostly shows up at the level of *which*
neurons got selected, not in whether the aggregate mismatch curve looks
different. That is reported as a finding, not hidden -- forcing a
bigger aggregate gap by further hand-tuning constants would defeat the
point of running an ablation at all.






demo_scaled_dpp_attention_3.py
=============================
Scalability showcase for the snn_lib ART-style slotted novelty detector.

Where ablation_dpp_attention.py runs a small, statistically-powered (multi-
seed) factorial to isolate each mechanism's marginal effect, this script
runs the SAME underlying library at a much larger scale to demonstrate that
the architecture actually scales -- more neurons, many more simultaneously
learnable categories, and a proper capacity curve across the whole trained
set, not just one or two hand-picked patterns.

Scale, for reference:
    original two-column script  :  ~20 neurons,  2 categories
    ablation study (this project): ~180 neurons, up to 17 categories
    this demo                    : ~750 neurons, up to 56 categories

Rather than a full factorial (expensive at this scale x many seeds), this
compares 4 representative configurations at one seed each:
    Full            : STDP + DPP allocation + attention-routed inhibition
    No DPP          : STDP + random allocation + attention-routed inhibition
    No attention    : STDP + DPP allocation + uniform (unrouted) inhibition
    Nothing         : STDP off, random allocation, uniform inhibition

and reports a CAPACITY CURVE: after training every pattern once, each
pattern is re-tested in its original training order, so degradation from
later-learned patterns crowding the substrate (interference on EARLIER
patterns specifically) is visible directly, rather than averaged away.