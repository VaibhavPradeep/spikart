"""
snn_lib_2.py
============
Version 2 SNN library incorporating initial Determinantal Point Process (DPP) 
slot allocation and softmax attention-routed inhibition mechanisms.
"""

import numpy as np


# ----------------------------------------------------------------------
# Neurons
# ----------------------------------------------------------------------
class LIFPopulation:
    def __init__(self, n_neurons, n_inhibitory=0, is_inhibitory=None,
                 v_rest=-70.0, v_th=-50.0, v_reset=-75.0,
                 tau_m=20.0, R=1.0, t_refractory=2.0, tau_trace=20.0,
                 name="pop"):
        self.name = name
        self.n = n_neurons
        if is_inhibitory is not None:
            self.is_inhibitory = np.array(is_inhibitory, dtype=bool)
        else:
            self.is_inhibitory = np.zeros(n_neurons, dtype=bool)
            if n_inhibitory:
                self.is_inhibitory[n_neurons - n_inhibitory:] = True
        self.exc_idx = np.where(~self.is_inhibitory)[0]
        self.inh_idx = np.where(self.is_inhibitory)[0]
        self.n_exc, self.n_inh = len(self.exc_idx), len(self.inh_idx)

        self.v_rest, self.v_th, self.v_reset = v_rest, v_th, v_reset
        self.tau_m, self.R, self.t_refractory = tau_m, R, t_refractory
        self.tau_trace = tau_trace

        self.v = np.full(n_neurons, v_rest, dtype=float)
        self.refractory = np.zeros(n_neurons)
        self.spiked = np.zeros(n_neurons, dtype=bool)
        self.trace = np.zeros(n_neurons)   # STDP eligibility trace

    def step(self, I, dt=0.1):
        active = self.refractory <= 0
        dv = np.where(active,
                       (-(self.v - self.v_rest) + self.R * I) / self.tau_m * dt,
                       0.0)
        self.v = np.where(active, self.v + dv, self.v_reset)
        self.refractory = np.where(active, self.refractory, self.refractory - dt)

        fired = active & (self.v >= self.v_th)
        self.v = np.where(fired, self.v_reset, self.v)
        self.refractory = np.where(fired, self.t_refractory, self.refractory)
        self.spiked = fired

        self.trace *= np.exp(-dt / self.tau_trace)
        self.trace[fired] += 1.0
        return fired.astype(float)

    def reset(self):
        self.v[:] = self.v_rest
        self.refractory[:] = 0.0
        self.spiked[:] = False
        self.trace[:] = 0.0


def build_multi_column_population(n_columns, n_exc_per_col, n_inh_per_col,
                                    name="Memory", rng=None):
    """
    Build ONE big LIFPopulation representing `n_columns` independent
    cortical columns, each with local E/I recurrent connectivity
    (block-diagonal — no direct lateral wiring between columns). This is
    the standard trick for scaling: many columns as one vectorized
    population instead of many small Python objects in a loop.

    Returns (population, recurrent_weights, exc_mask) where exc_mask is a
    boolean array (True at every excitatory neuron, across all columns) —
    the natural "candidate pool" for allocation.
    """
    rng = rng or np.random
    per_col = n_exc_per_col + n_inh_per_col
    total = n_columns * per_col
    is_inh = np.zeros(total, dtype=bool)
    W = np.zeros((total, total))

    for c in range(n_columns):
        s = c * per_col
        e = s + n_exc_per_col
        i_e = s + per_col
        is_inh[e:i_e] = True

        W[s:e, s:e] = rng.uniform(1.5, 3.5, (n_exc_per_col, n_exc_per_col))
        np.fill_diagonal(W[s:e, s:e], 0.0)     # E-E recurrent excitation
        W[s:e, e:i_e] = -6.0                    # I -> E inhibition
        W[e:i_e, s:e] = 4.0                     # E -> I excitation

    pop = LIFPopulation(total, is_inhibitory=is_inh, name=name)
    return pop, W, ~is_inh


# ----------------------------------------------------------------------
# Synapses
# ----------------------------------------------------------------------
class Projection:
    """
    weights shape: (n_target, n_source).

    Synaptic current is filtered through a decaying trace (tau_syn) rather
    than delivered as a single-timestep impulse — the standard
    current-based synapse model (NEST/Brian2-style). Without it, a source
    population that only spikes occasionally delivers almost no net charge
    downstream.
    """
    def __init__(self, source, target, weights, plasticity=None,
                 learning_rate=1.0, A_plus=0.06, A_minus=0.072,
                 w_min=0.0, w_max=45.0, tau_syn=5.0, mask=None,
                 gate=None, name=""):
        self.source, self.target = source, target
        self.weights = np.array(weights, dtype=float)
        self.plasticity = plasticity
        self.lr = learning_rate
        self.A_plus, self.A_minus = A_plus, A_minus
        self.w_min, self.w_max = w_min, w_max
        self.tau_syn = tau_syn
        self.psc = np.zeros(source.n)
        self.gate = gate
        self.name = name or f"{source.name}->{target.name}"
        # connectivity mask: which (target, source) pairs are real synapses.
        # Without this, STDP can "grow" brand-new connections wherever a
        # target neuron fires — e.g. onto inhibitory neurons that were
        # never supposed to receive this input.
        self.mask = (self.weights != 0) if mask is None else np.array(mask, dtype=bool)
        # active_mask: which TARGET rows are currently plastic-eligible.
        # None = all mask-connected rows eligible (old behavior). Set by
        # an AllocationController to gate learning to only the
        # DPP-selected neurons for the pattern currently being presented.
        self.active_mask = None

    def set_active_mask(self, target_row_mask):
        self.active_mask = None if target_row_mask is None else np.array(target_row_mask, dtype=bool)

    def pulse_and_decay(self, dt=0.1):
        """Call once per network tick, before current(). Advances the
        synaptic filter using the source's spikes from the previous tick."""
        self.psc *= np.exp(-dt / self.tau_syn)
        self.psc += self.source.spiked.astype(float)

    def current(self):
        return self.weights @ self.psc

    def update(self, pattern=None, dt=0.1):
        if self.plasticity is None:
            return

        if self.plasticity == "hebbian":
            # legacy rule from the original scripts: gated bulk update on
            # the raw pattern vector, no spike timing involved at all.
            if pattern is None:
                return
            if self.gate is not None and not self.gate():
                return
            self.weights += self.lr * np.outer(pattern, pattern)

        elif self.plasticity == "stdp":
            # real pair-based STDP: uses actual pre/post spikes this step
            # against the *other* side's decayed trace, gated to only the
            # currently-allocated rows if an AllocationController set one.
            post_fired, pre_fired = self.target.spiked, self.source.spiked
            pre_trace, post_trace = self.source.trace, self.target.trace
            row_gate = self.active_mask if self.active_mask is not None \
                else np.ones(self.target.n, dtype=bool)

            ltp_rows = post_fired & row_gate
            if ltp_rows.any():
                self.weights[ltp_rows, :] += (self.A_plus * pre_trace[np.newaxis, :]
                                               * self.mask[ltp_rows, :])
            if pre_fired.any():
                ltd = (self.A_minus * post_trace[:, np.newaxis] * self.mask[:, pre_fired]
                       * row_gate[:, np.newaxis])
                self.weights[:, pre_fired] -= ltd

        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)


class AttentionRoutedProjection(Projection):
    """
    Memory -> Evaluator inhibition computed each tick via scaled
    dot-product attention, instead of a fixed identity matrix.

    Keys   = each Memory neuron's current receptive field (the row of a
             *reference* plastic projection, typically Input->Memory) —
             i.e. "what pattern-feature has this neuron learned to
             represent".
    Queries = one-hot per input feature (each Evaluator neuron j asks
              "which memory neurons are tuned to feature j?").
    This lets inhibition route correctly to a large, sparsely-allocated
    memory pool with no fixed 1:1 index correspondence to the input —
    the old identity-matrix approach only worked because Memory and
    Evaluator happened to be hand-sized to match exactly.
    """
    def __init__(self, source, target, key_projection, n_features,
                 inhibition_strength=60.0, temperature=None,
                 tau_syn=5.0, name=""):
        weights = np.zeros((target.n, source.n))
        super().__init__(source, target, weights, plasticity=None,
                          tau_syn=tau_syn, name=name)
        self.key_projection = key_projection
        self.n_features = n_features
        self.inhibition_strength = inhibition_strength
        self.temperature = temperature or np.sqrt(n_features)

    def _recompute_weights(self):
        K = self.key_projection.weights[:, :self.n_features]   # (n_mem, n_features)
        scores = K / self.temperature
        scores = scores - scores.max(axis=0, keepdims=True)
        expv = np.exp(scores)
        attn = expv / (expv.sum(axis=0, keepdims=True) + 1e-12)  # softmax over memory axis
        self.weights[:] = 0.0
        self.weights[:self.n_features, :] = -(self.inhibition_strength * attn.T)

    def pulse_and_decay(self, dt=0.1):
        self._recompute_weights()
        super().pulse_and_decay(dt)


# ----------------------------------------------------------------------
# DPP-diversified allocation
# ----------------------------------------------------------------------
def _greedy_gram_schmidt_select(B, k, eps=1e-8):
    """
    Greedy MAP inference for a DPP with kernel L = B B^T (quality-diversity
    kernel: each row of B is quality_i * (feature_i / ||feature_i||)).
    Equivalent to greedy log-det maximization; implemented as iterative
    Gram-Schmidt residual-norm maximization, which is the standard
    efficient formulation (avoids ever forming the NxN kernel L).
    """
    N = B.shape[0]
    residual = B.copy()
    selected_mask = np.zeros(N, dtype=bool)
    for _ in range(min(k, N)):
        scores = np.sum(residual ** 2, axis=1)
        scores[selected_mask] = -np.inf
        i = int(np.argmax(scores))
        if scores[i] <= eps:
            break
        selected_mask[i] = True
        v = residual[i] / (np.linalg.norm(residual[i]) + eps)
        residual = residual - np.outer(residual @ v, v)
    return selected_mask


class AllocationController:
    """
    Decides which neurons in a large candidate pool get to represent a
    newly novel pattern, using greedy DPP MAP selection over a
    quality-diversity kernel:
      - quality  = how strongly each candidate currently responds to this
                   pattern (raw dot product with its input weights)
      - diversity = how different each candidate's receptive field is
                   from neurons already claimed by EARLIER patterns
                   (tracked persistently across calls) and from each
                   other within this allocation.
    This directly targets the interference problem: without it, STDP
    potentiates whichever neurons happen to already be co-active, which
    is exactly what causes different patterns to blur into the same
    neurons when they share input features.
    """
    def __init__(self, n_candidates, feature_dim):
        self.n_candidates = n_candidates
        self.feature_dim = feature_dim
        self.used_directions = []   # persistent orthonormal basis of claimed directions

    def _project_out_used(self, B):
        residual = B.copy()
        for u in self.used_directions:
            residual = residual - np.outer(residual @ u, u)
        return residual

    def allocate(self, features, quality, k):
        """
        features: (n_candidates, feature_dim) receptive-field vectors.
        quality:  (n_candidates,) nonnegative responsiveness to the new pattern.
        k: number of neurons to allocate.
        Returns a boolean mask over candidates.
        """
        norms = np.linalg.norm(features, axis=1) + 1e-8
        B = (np.clip(quality, 0, None)[:, None] * features) / norms[:, None]
        residual = self._project_out_used(B)
        mask = _greedy_gram_schmidt_select(residual, k)

        for i in np.where(mask)[0]:
            v = self._project_out_used(B[i:i + 1])[0]
            n = np.linalg.norm(v)
            if n > 1e-6 and len(self.used_directions) < self.feature_dim:
                self.used_directions.append(v / n)
        return mask


# ----------------------------------------------------------------------
# Factory: build a full (possibly large, multi-column) novelty-detector
# network, with DPP allocation and attention-routed inhibition as
# switchable options — this is what the ablation experiment and the demo
# both build on, so the two stay consistent.
# ----------------------------------------------------------------------
def _internal_ei_weights(n_exc, n_inh):
    w = np.zeros((n_exc + n_inh, n_exc + n_inh))
    w[:n_exc, :n_exc] = np.random.uniform(1.5, 3.5, (n_exc, n_exc))
    np.fill_diagonal(w[:n_exc, :n_exc], 0.0)
    w[:n_exc, n_exc:] = -6.0
    w[n_exc:, :n_exc] = 4.0
    return w


def build_novelty_network(pattern_size, n_columns, exc_per_col, inh_per_col,
                           use_dpp=True, use_attention=True, ev_inh=None,
                           inhibition_strength=70.0, in_ev_strength=45.0,
                           A_plus=0.15, A_minus=0.12, w_max=60.0, tau_syn=5.0,
                           mem_init_range=(0.5, 1.5)):
    """
    Returns a dict bundling the network and everything present_pattern()
    needs. Set use_dpp=False / use_attention=False to fall back to the
    old behaviors, for ablation:
      - use_dpp=False: STDP is unrestricted (any co-active memory neuron
        potentiates — the original interference-prone behavior).
      - use_attention=False: Memory->Evaluator inhibition is a fixed,
        one-time "hard" nearest-feature routing computed from each
        memory neuron's INITIAL random weights, instead of adaptively
        computed attention over its LEARNED weights.
    """
    ev_inh = ev_inh or max(2, pattern_size // 4)

    inp = LIFPopulation(pattern_size, n_inhibitory=0, name="Input")
    mem, mem_W, mem_exc_mask = build_multi_column_population(
        n_columns, exc_per_col, inh_per_col, name="Memory")
    ev, ev_W, _ = build_multi_column_population(1, pattern_size, ev_inh, name="Evaluator")
    ev.rectify_input = True

    net = Network()
    net.add_population(inp)
    net.add_population(mem)
    net.add_population(ev)
    net.add_projection(Projection(mem, mem, mem_W, plasticity=None))
    net.add_projection(Projection(ev, ev, ev_W, plasticity=None))

    w_in_mem = np.zeros((mem.n, pattern_size))
    w_in_mem[mem_exc_mask, :] = np.random.uniform(mem_init_range[0], mem_init_range[1],
                                                    (mem_exc_mask.sum(), pattern_size))
    proj_in_mem = net.add_projection(Projection(
        inp, mem, w_in_mem, plasticity="stdp", A_plus=A_plus, A_minus=A_minus,
        w_min=0.0, w_max=w_max, tau_syn=tau_syn, name="Input->Memory"))

    w_in_ev = np.zeros((ev.n, pattern_size))
    w_in_ev[:pattern_size, :pattern_size] = np.eye(pattern_size) * in_ev_strength
    net.add_projection(Projection(inp, ev, w_in_ev, plasticity=None, tau_syn=tau_syn))

    if use_attention:
        mem_ev = AttentionRoutedProjection(
            mem, ev, key_projection=proj_in_mem, n_features=pattern_size,
            inhibition_strength=inhibition_strength, tau_syn=tau_syn)
    else:
        # "hard" baseline: each memory neuron permanently wired to whichever
        # single input feature its INITIAL random weights happen to favor —
        # fixed at build time, never adapts as Input->Memory learns.
        w_mem_ev = np.zeros((ev.n, mem.n))
        peak_feature = np.argmax(w_in_mem[mem_exc_mask, :], axis=1)
        mem_exc_idx = np.where(mem_exc_mask)[0]
        for local_i, global_i in enumerate(mem_exc_idx):
            w_mem_ev[peak_feature[local_i], global_i] = -inhibition_strength
        mem_ev = Projection(mem, ev, w_mem_ev, plasticity=None, tau_syn=tau_syn)
    net.add_projection(mem_ev)

    allocator = AllocationController(n_candidates=int(mem_exc_mask.sum()),
                                      feature_dim=pattern_size) if use_dpp else None

    return dict(net=net, inp=inp, mem=mem, ev=ev, mem_exc_mask=mem_exc_mask,
                proj_in_mem=proj_in_mem, allocator=allocator,
                pattern_size=pattern_size)


def present_pattern(bundle, pattern, steps, dt=0.1, drive=180.0, k_alloc=12,
                     probe_steps=150, vigilance=0.02, primer_strength=3.0):
    """
    Run one pattern for `steps` ticks. Returns (probe_rate, mismatches):
      probe_rate  = mean mismatch during the vigilance PROBE, i.e. BEFORE
                    any allocation/fast-commit for this call — this is the
                    genuine "how novel does this look right now" signal.
      mismatches  = per-step mismatch array for the main window AFTER
                    any allocation/priming has already been applied (so on
                    a truly new pattern this will typically already be
                    lower than probe_rate — that's the fast-commit
                    working, not a bug).

    If the bundle has a DPP allocator, this follows real ART-style
    vigilance-gated allocation instead of reallocating on every call:
      1. PROBE: present the pattern briefly with no learning, measure the
         current mismatch rate.
      2. If mismatch is above `vigilance` (existing memory doesn't
         recognize this well), it's genuinely novel -> DPP-select k fresh
         neurons and "fast-commit" them toward this pattern (ART's usual
         fast-learning category commit), then let real STDP refine them
         over the rest of the presentation.
      3. If mismatch is already low, this pattern is already represented
         -> no new allocation; whatever's already responding keeps
         adjusting via ordinary unrestricted STDP.
    Without this, every call would re-trigger allocation (even for a
    pattern already learned), and the DPP's permanent used-directions
    bookkeeping would incorrectly force it onto NEW neurons instead of
    reinforcing the ones that already represent it.
    """
    inp, mem, ev = bundle["inp"], bundle["mem"], bundle["ev"]
    proj_in_mem, allocator = bundle["proj_in_mem"], bundle["allocator"]
    mem_exc_mask = bundle["mem_exc_mask"]
    pattern = np.asarray(pattern, dtype=float)

    probe_rate = None
    if allocator is not None:
        proj_in_mem.set_active_mask(None)
        probe_total = 0.0
        for _ in range(probe_steps):
            bundle["net"].step({"Input": pattern * drive}, dt=dt)
            probe_total += ev.spiked[:bundle["pattern_size"]].sum()
        probe_rate = probe_total / probe_steps

        if probe_rate > vigilance:
            cand_idx = np.where(mem_exc_mask)[0]
            cand_features = proj_in_mem.weights[cand_idx, :]
            cand_quality = cand_features @ pattern
            chosen_local = allocator.allocate(cand_features, cand_quality, k=k_alloc)
            active_idx = cand_idx[chosen_local]
            active_mask = np.zeros(mem.n, dtype=bool)
            active_mask[active_idx] = True

            proj_in_mem.weights[active_idx, :] = np.clip(
                proj_in_mem.weights[active_idx, :]
                + primer_strength * pattern[np.newaxis, :] * proj_in_mem.mask[active_idx, :],
                proj_in_mem.w_min, proj_in_mem.w_max)
            proj_in_mem.set_active_mask(active_mask)
        else:
            proj_in_mem.set_active_mask(None)
    else:
        proj_in_mem.set_active_mask(None)

    mismatches = np.zeros(steps)
    for t in range(steps):
        bundle["net"].step({"Input": pattern * drive}, dt=dt)
        mismatches[t] = ev.spiked[:bundle["pattern_size"]].sum()

    if probe_rate is None:
        probe_rate = mismatches[:probe_steps].mean() if steps >= probe_steps else mismatches.mean()
    return probe_rate, mismatches


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
class Network:
    def __init__(self):
        self.populations = {}
        self.projections = []

    def add_population(self, pop):
        self.populations[pop.name] = pop
        return pop

    def add_projection(self, proj):
        self.projections.append(proj)
        return proj

    def step(self, external_currents, dt=0.1, pattern=None):
        """external_currents: dict {pop_name: array} of extra input current."""
        totals = {name: np.array(external_currents.get(name, np.zeros(p.n)), dtype=float)
                  for name, p in self.populations.items()}

        for proj in self.projections:
            proj.pulse_and_decay(dt)
            totals[proj.target.name][:proj.weights.shape[0]] += proj.current()

        for name, pop in self.populations.items():
            t = totals[name]
            if getattr(pop, "rectify_input", False):
                t = np.maximum(t, 0.0)
            pop.step(t, dt)

        for proj in self.projections:
            proj.update(pattern=pattern, dt=dt)

    def reset(self):
        for p in self.populations.values():
            p.reset()
        for proj in self.projections:
            proj.psc[:] = 0.0
