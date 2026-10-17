snn_lib_2.py — general-purpose spiking network library.

v1: vectorized LIF populations, generic Projections (static / legacy
    rate-Hebbian / real trace-based STDP), a Network graph.
v2 (this version) adds three things needed to get past a 10-neuron,
    4-pattern toy demo:
  - LIFPopulation now takes an explicit `is_inhibitory` mask, so a
    population no longer has to be "n_exc then n_inh" — it can be built
    out of many independent columns (block-diagonal recurrent weights).
  - AllocationController: greedy DPP (determinantal point process) MAP
    selection, used to pick WHICH neurons in a large memory pool get to
    learn a newly novel pattern, biased toward neurons that are both
    responsive to it (quality) and dissimilar to neurons already
    claimed by earlier patterns (diversity). This is what "novelty-gated
    LTP" was missing: it let ANY co-active neuron potentiate, which is
    exactly what caused the cross-pattern interference in the earlier
    4-pattern demo.
  - AttentionRoutedProjection: replaces the old identity-matrix
    Memory->Evaluator inhibition (which only worked because the two
    populations were hand-sized to match 1:1) with scaled dot-product
    attention over each memory neuron's *learned* receptive field, so
    inhibition still routes correctly even when Memory is a large,
    sparsely-allocated pool with no fixed index correspondence to the
    input features.






ablation_experiment_2.py —  a 2x2 FACTORIAL ablation over with mechanisms:

  DPP-diversified allocation  (which memory neurons get to learn a
                                new pattern, chosen for quality + diversity
                                from a large candidate pool, with ART-style
                                vigilance gating and fast-commit)
  Attention-routed inhibition (Memory->Evaluator routing computed via
                                scaled dot-product attention over LEARNED
                                receptive fields, instead of a fixed map)

Conditions:            DPP   Attention
  baseline               off   off   (closest analogue to the original
                                       two-column script, just scaled up)
  dpp_only                on   off
  attention_only          off   on
  full                    on   on

Scale:
  pattern_size   = 24
  memory pool    = 6 columns x 10 excitatory (+2 inhibitory) = 72 neurons
  evaluator      = 1 column, 24 excitatory (+6 inhibitory) = 30 neurons
  patterns       = 6, sparse (~35% active), some deliberately overlapping

Metric: for each pattern, mismatch is measured via the vigilance PROBE
(a short no-learning window at the very start of each present_pattern
call) — i.e. always BEFORE that call gets to change anything. This keeps
"1st exposure", "retest", and "after-all-others-learned" all measuring
the state as it stood at that moment, uncontaminated by the fast-commit
that then happens later in that same call. (c) minus (b) is the
INTERFERENCE gap — how much an already-learned pattern degrades purely
from other patterns being learned afterward. That's the number the
earlier 4-pattern demo couldn't speak to.






demo_dpp_attention_2.py — showcase run of the full DPP-allocation +
attention-routed spiking novelty detector, at a scale meant to actually
say something about "does this hold up beyond a handful of neurons":

  pattern size   = 48 bits
  memory pool    = 16 columns x 12 excitatory (+3 inhibitory) = 192 exc
                   neurons, 240 neurons total
  evaluator      = 1 column, 48 excitatory (+8 inhibitory) = 56 neurons
  patterns       = 10, sparse (~35% active), several with deliberate
                   overlap with pattern 0 (to stress-test interference)

This is the SAME library (snn_lib.py) and mechanisms as
ablation_experiment.py's "full" condition — this script just runs one
condition at ~8x the memory-pool size and 10 patterns instead of 6.