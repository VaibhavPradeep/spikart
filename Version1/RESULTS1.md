snn_lib.py — a small, general-purpose spiking network library, generalized
out of the two-column novelty-detection prototype.

Core pieces:
  - LIFPopulation : vectorized leaky integrate-and-fire neurons (any size,
                     any exc/inh split), each carrying an STDP eligibility trace.
  - Projection    : weighted connection between two populations (or a
                     population and itself, for recurrent E/I connectivity).
                     plasticity is pluggable: None / 'hebbian' (legacy,
                     matches the original scripts) / 'stdp' (real,
                     pair-based, trace-driven).
  - Network       : holds any number of populations + projections and steps
                     them all together each tick. No hardcoded column count.






weights shape: (n_target, n_source).

    Synaptic current is filtered through a decaying trace (tau_syn) rather
    than delivered as a single-timestep impulse — this is the standard
    current-based synapse model used in real SNN simulators (NEST, Brian2),
    and it matters a lot here: without it, a source population that only
    spikes occasionally (a real rate-coded input, instead of the original
    scripts' "inject the raw pattern as continuous current every step"
    shortcut) delivers almost no net charge and nothing downstream ever
    reaches threshold.






demo_multi_pattern.py — two-column novelty circuit with:
  - a real spiking Input population (patterns are actual pre-synaptic
    spike trains, not injected analog current)
  - real pair-based STDP between Input -> Memory (instead of the
    rate-Hebbian outer product)
  - a test of MULTIPLE stored patterns, including re-testing an early
    pattern after later ones have been learned (interference check)

