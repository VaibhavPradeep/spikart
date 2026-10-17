"""
snn_lib_3.py
============
Version 3 SNN core library featuring expanded Adaptive Resonance Theory (ART) 
memory substrate and integrated STDP learning rules.
"""

import numpy as np


# =============================================================================
# 1. LIF POPULATION
# =============================================================================
class LIFPopulation:
    def __init__(self, n_neurons, n_inhibitory=0, v_rest=-70.0, v_th=-50.0,
                 v_reset=-75.0, tau_m=20.0, R=1.0, t_refractory=2.0,
                 tau_trace=20.0, name="pop"):
        self.name = name
        self.n = n_neurons
        self.n_inh = n_inhibitory
        self.n_exc = n_neurons - n_inhibitory
        self.v_rest, self.v_th, self.v_reset = v_rest, v_th, v_reset
        self.tau_m, self.R, self.t_refractory = tau_m, R, t_refractory
        self.tau_trace = tau_trace

        self.v = np.full(n_neurons, v_rest, dtype=float)
        self.refractory = np.zeros(n_neurons)
        self.spiked = np.zeros(n_neurons, dtype=bool)
        self.trace = np.zeros(n_neurons)   # STDP / activity eligibility trace
        self.rectify_input = False

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


# =============================================================================
# 2. PROJECTION
# =============================================================================
class Projection:
    """weights shape: (n_target, n_source). Current-based synapse: spikes are
    filtered through a decaying trace (tau_syn) instead of delivered as a
    single-timestep impulse, matching NEST/Brian2-style simulators."""

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
        self.mask = (self.weights != 0) if mask is None else np.array(mask, dtype=bool)

    def pulse_and_decay(self, dt=0.1):
        self.psc *= np.exp(-dt / self.tau_syn)
        self.psc += self.source.spiked.astype(float)

    def current(self):
        return self.weights @ self.psc

    def update(self, pattern=None, dt=0.1):
        if self.plasticity is None:
            return

        if self.plasticity == "hebbian":
            if pattern is None:
                return
            if self.gate is not None and not self.gate():
                return
            self.weights += self.lr * np.outer(pattern, pattern)

        elif self.plasticity == "stdp":
            post_fired, pre_fired = self.target.spiked, self.source.spiked
            pre_trace, post_trace = self.source.trace, self.target.trace
            if post_fired.any():
                self.weights[post_fired, :] += (self.A_plus * pre_trace[np.newaxis, :]
                                                 * self.mask[post_fired, :])
            if pre_fired.any():
                self.weights[:, pre_fired] -= (self.A_minus * post_trace[:, np.newaxis]
                                                * self.mask[:, pre_fired])

        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)


# =============================================================================
# 3. NETWORK
# =============================================================================
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


# =============================================================================
# 4. GREEDY DPP MAP INFERENCE  (Chen, Zhang & Zhou, 2018)
# =============================================================================
def greedy_dpp_map(kernel, k):
    """Greedy MAP inference for a k-DPP given a PSD kernel L (n x n).
    Returns up to k indices approximately maximizing det(L_S) -- i.e. a set
    that is simultaneously high "quality" (large diagonal entries) and
    mutually diverse (low off-diagonal similarity). O(k^2 n) via the
    incremental-Cholesky greedy algorithm; no explicit determinant ever
    needs to be computed."""
    n = kernel.shape[0]
    d2 = np.diag(kernel).copy().astype(float)
    selected = []
    c = []  # incremental Cholesky columns, one length-n vector per selection
    for _ in range(min(k, n)):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        selected.append(j)
        Lj = kernel[j, :].astype(float)
        if c:
            C = np.vstack(c)                 # (t, n)
            proj = C[:, j] @ C                # (n,)
            e = (Lj - proj) / np.sqrt(d2[j])
        else:
            e = Lj / np.sqrt(d2[j])
        c.append(e)
        d2 = d2 - e ** 2
        d2[j] = -np.inf
    return selected


# =============================================================================
# 5. ATTENTION ROUTER
# =============================================================================
class AttentionRouter:
    """Softmax attention over per-slot match/relevance scores. Used to decide
    how much of each memory category's inhibitory ("already explained")
    signal is allowed to reach the mismatch evaluator -- a stand-in for
    top-down attentional gating / precision-weighting of prediction error."""

    def __init__(self, temperature=0.15):
        self.temperature = temperature
        self.weights = None

    def compute(self, match_scores):
        m = np.asarray(match_scores, dtype=float)
        m = m - m.max()
        w = np.exp(m / self.temperature)
        w /= w.sum() + 1e-12
        self.weights = w
        return w


# =============================================================================
# 6. SLOTTED ART MEMORY  (vigilance test + DPP allocation + routed inhibition)
# =============================================================================
class SlottedARTMemory:
    """
    Manages a shared pool ("substrate") of memory neurons as a set of
    ART-style category slots:

      - route(pattern)   : ART vigilance test against committed slot
                            templates. Returns whether a NEW slot is needed,
                            plus (if not) the attention weights over
                            existing slots.
      - allocate(...)     : grows a new slot out of the substrate, choosing
                             member neurons either via DPP (diverse +
                             quality-weighted) or uniformly at random
                             (the ablation baseline), and gives those
                             neurons a one-shot "resonance boost" toward the
                             new pattern (fast-learning ART commit).
      - inhibition_for(...): the routed inhibitory current delivered to the
                             evaluator population this tick.

    use_dpp and use_attention are independent toggles so a caller can run
    the full with/without factorial ablation.
    """

    def __init__(self, substrate_size, slot_size, pattern_size, vigilance=0.75,
                 use_dpp=True, use_attention=True, attn_temperature=0.15,
                 boost=6.0, seed=0):
        self.substrate_size = substrate_size
        self.slot_size = slot_size
        self.pattern_size = pattern_size
        self.vigilance = vigilance
        self.use_dpp = use_dpp
        self.use_attention = use_attention
        self.boost = boost
        self.rng = np.random.default_rng(seed)
        self.reuse_count = np.zeros(substrate_size, dtype=int)
        self.slots = []  # each: {"neurons": idx array, "template": pattern vec}
        self.router = AttentionRouter(temperature=attn_temperature)

    def match_scores(self, pattern):
        """Jaccard similarity |p AND t| / |p OR t| between the (binary)
        input pattern and each committed slot's template. Symmetric, unlike
        classic ART's directional subset-match ratio -- with dense random
        binary patterns a one-directional criterion produces frequent
        chance "matches" whenever a template happens to be a superset of a
        sparser pattern's active bits, which would falsely recognize
        patterns the network has never actually seen."""
        if not self.slots:
            return np.array([])
        scores = []
        for s in self.slots:
            t = s["template"]
            inter = np.minimum(pattern, t).sum()
            union = np.maximum(pattern, t).sum()
            scores.append(inter / (union + 1e-9))
        return np.array(scores)

    def route(self, pattern):
        scores = self.match_scores(pattern)
        needs_alloc = (scores.size == 0) or (scores.max() < self.vigilance)
        if scores.size == 0:
            attn = scores
        elif self.use_attention:
            attn = self.router.compute(scores)
        else:
            attn = np.full(scores.shape, 1.0 / len(scores))
        return needs_alloc, attn

    def allocate(self, pattern, W_in):
        """W_in: Input->Memory weight matrix (substrate_size x pattern_size),
        modified in place with a small additive resonance boost for chosen
        neurons. Also records, for every allocation regardless of use_dpp,
        the mean "quality" (pre-existing cosine alignment with the pattern)
        and mean pairwise similarity ("redundancy", lower = more diverse)
        of the chosen neurons -- a direct, deterministic measure of what
        the selection method actually bought, independent of how the rest
        of the spiking network's dynamics happen to play out."""
        unallocated = np.where(self.reuse_count == 0)[0]
        pool = unallocated if len(unallocated) >= self.slot_size else np.arange(self.substrate_size)
        k = min(self.slot_size, len(pool))

        feats_all = W_in[pool, :]
        norms_all = np.linalg.norm(feats_all, axis=1, keepdims=True) + 1e-9
        featsn_all = feats_all / norms_all
        S_all = featsn_all @ featsn_all.T
        pat_n = pattern / (np.linalg.norm(pattern) + 1e-9)
        q_all = np.clip(featsn_all @ pat_n, 1e-3, None)

        if self.use_dpp:
            q = q_all / (1.0 + self.reuse_count[pool])
            L = (q[:, None] * S_all) * q[None, :]
            chosen_local = greedy_dpp_map(L, k)
            if len(chosen_local) < k:
                remaining = [i for i in range(len(pool)) if i not in chosen_local]
                chosen_local += list(self.rng.choice(remaining, k - len(chosen_local), replace=False))
        else:
            chosen_local = list(self.rng.choice(len(pool), size=k, replace=False))
        chosen = pool[chosen_local]

        mean_quality = float(q_all[chosen_local].mean())
        sub_S = S_all[np.ix_(chosen_local, chosen_local)]
        off_diag = sub_S[~np.eye(k, dtype=bool)]
        mean_redundancy = float(off_diag.mean()) if k > 1 else 0.0

        self.reuse_count[chosen] += 1
        W_in[chosen, :] += self.boost * pattern
        np.clip(W_in, 0.0, 45.0, out=W_in)
        self.slots.append({"neurons": chosen, "template": pattern.copy(),
                            "alloc_quality": mean_quality, "alloc_redundancy": mean_redundancy})
        return chosen

    def allocation_diagnostics(self):
        """Mean quality / mean redundancy across every slot ever allocated
        -- a construction-level report of what the allocation method (DPP
        vs random) actually selected, regardless of downstream dynamics."""
        if not self.slots:
            return {"mean_quality": float("nan"), "mean_redundancy": float("nan")}
        q = np.mean([s["alloc_quality"] for s in self.slots])
        r = np.mean([s["alloc_redundancy"] for s in self.slots])
        return {"mean_quality": float(q), "mean_redundancy": float(r)}

    def inhibition_for(self, pattern, substrate_trace, inhibition_strength):
        needs_alloc, attn = self.route(pattern)
        inhib = np.zeros(self.pattern_size)
        for w, slot in zip(attn, self.slots):
            activity = substrate_trace[slot["neurons"]].mean()
            inhib += w * activity * slot["template"]
        return needs_alloc, inhib * inhibition_strength


# =============================================================================
# 7. NETWORK-BUILDING / EXPERIMENT-RUNNING HELPERS (shared by all scripts)
# =============================================================================
def internal_weights(n_exc, n_inh, rng, ref_n_exc=8, ref_n_inh=2):
    """Recurrent E/I connectivity for a competitive population: sparse-ish
    random excitatory recurrence + a shared inhibitory pool giving soft
    competition (same qualitative pattern as the original two-column
    scripts). Per-connection strengths are mean-field normalized against a
    reference population size (the original 8-exc/2-inh column) so that the
    *aggregate* recurrent current a neuron receives stays in the same
    regime regardless of how large the substrate is scaled up -- without
    this, per-connection weights calibrated for ~10 neurons make a
    several-hundred-neuron substrate's shared inhibitory pool massively
    over- or under-powered, and the E/I balance breaks down."""
    n = n_exc + n_inh
    w = np.zeros((n, n))
    w[:n_exc, :n_exc] = rng.uniform(1.0, 2.5, (n_exc, n_exc)) * (ref_n_exc / n_exc)
    np.fill_diagonal(w[:n_exc, :n_exc], 0.0)
    w[:n_exc, n_exc:] = -6.0 * (ref_n_inh / n_inh)
    w[n_exc:, :n_exc] = 4.0 * (ref_n_exc / n_exc)
    return w


def well_separated_patterns(pattern_size, n_patterns, active_bits=None, seed=0,
                             pool_size=6000):
    """Generates n_patterns distinct fixed-density binary vectors, chosen
    greedily to minimize pairwise Jaccard overlap. Plain i.i.d. random bit
    vectors can collide (two "different" patterns can look almost identical
    just by chance, especially at small pattern_size) which would confuse
    the ART vigilance test; this keeps the generated pattern set honestly
    separable so that mismatches reflect the network's behaviour rather
    than an accidentally ambiguous benchmark."""
    rng = np.random.default_rng(seed)
    k = active_bits or max(1, pattern_size // 3)
    seen = set()
    pool = []
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 20:
        attempts += 1
        idx = rng.choice(pattern_size, size=k, replace=False)
        key = tuple(sorted(idx.tolist()))
        if key in seen:
            continue
        seen.add(key)
        v = np.zeros(pattern_size)
        v[list(key)] = 1.0
        pool.append(v)

    chosen = [pool.pop(0)]
    while len(chosen) < n_patterns and pool:
        def worst_overlap(v):
            return max(np.minimum(v, c).sum() / (np.maximum(v, c).sum() + 1e-9)
                        for c in chosen)
        pool.sort(key=worst_overlap)
        chosen.append(pool.pop(0))
    return chosen


def build_art_network(pattern_size, substrate_size, slot_size, vigilance,
                       use_dpp, use_attention, use_stdp, seed=0,
                       n_inh_frac=0.25, attn_temperature=0.15, boost=8.0,
                       tau_syn_in_mem=10.0):
    """Assembles Input / Memory-substrate / Evaluator populations, wires the
    (STDP-plastic or static) Input->Memory projection and the recurrent E/I
    projections, and creates the SlottedARTMemory that manages allocation
    and routed inhibition. Returns a dict with everything a caller needs to
    drive the network and read out its state."""
    rng = np.random.default_rng(seed)

    n_inh_sub = max(2, int(substrate_size * n_inh_frac))
    n_inh_ev = 2

    net = Network()
    inp = net.add_population(LIFPopulation(pattern_size, n_inhibitory=0, name="Input"))
    mem = net.add_population(LIFPopulation(substrate_size + n_inh_sub, n_inhibitory=n_inh_sub, name="Memory"))
    ev = net.add_population(LIFPopulation(pattern_size + n_inh_ev, n_inhibitory=n_inh_ev, name="Evaluator"))
    ev.rectify_input = True

    net.add_projection(Projection(mem, mem, internal_weights(substrate_size, n_inh_sub, rng),
                                   plasticity=None, tau_syn=5.0, name="Memory-recurrent"))
    net.add_projection(Projection(ev, ev, internal_weights(pattern_size, n_inh_ev, rng),
                                   plasticity=None, tau_syn=5.0, name="Evaluator-recurrent"))

    w_in_mem = np.zeros((substrate_size + n_inh_sub, pattern_size))
    w_in_mem[:substrate_size, :] = rng.uniform(0.5, 2.5, (substrate_size, pattern_size))
    proj_in_mem = net.add_projection(Projection(
        inp, mem, w_in_mem, plasticity=("stdp" if use_stdp else None),
        A_plus=0.18, A_minus=0.15, w_min=0.0, w_max=45.0, tau_syn=tau_syn_in_mem,
        name="Input->Memory"))

    art = SlottedARTMemory(substrate_size, slot_size, pattern_size, vigilance=vigilance,
                            use_dpp=use_dpp, use_attention=use_attention,
                            attn_temperature=attn_temperature, boost=boost, seed=seed)

    return {"net": net, "art": art, "input": inp, "memory": mem, "evaluator": ev,
            "proj_in_mem": proj_in_mem, "substrate_size": substrate_size,
            "pattern_size": pattern_size}


def run_schedule(system, schedule, steps_per_pattern, dt=0.1,
                  input_drive=180.0, sensory_drive=55.0, inhibition_strength=20.0):
    """Runs a list of (label, pattern) exposures through an already-built
    system (as returned by build_art_network). Returns a list of per-exposure
    result dicts: label, mean/total mismatch, whether a new slot was
    allocated during this exposure, and slot count afterwards."""
    net, art = system["net"], system["art"]
    inp, mem, ev = system["input"], system["memory"], system["evaluator"]
    proj_in_mem = system["proj_in_mem"]
    substrate_size = system["substrate_size"]

    results = []
    for label, pattern in schedule:
        pattern = np.asarray(pattern, dtype=float)
        mismatches = np.zeros(steps_per_pattern)
        allocated_this_exposure = False

        for t in range(steps_per_pattern):
            needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:substrate_size],
                                                     inhibition_strength)
            if needs_alloc and not allocated_this_exposure:
                art.allocate(pattern, proj_in_mem.weights)
                allocated_this_exposure = True
                needs_alloc, inhib = art.inhibition_for(pattern, mem.trace[:substrate_size],
                                                         inhibition_strength)

            ev_drive = np.maximum(pattern * sensory_drive - inhib, 0.0)
            ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)

            net.step({"Input": pattern * input_drive, "Evaluator": ext_ev}, dt=dt)
            mismatches[t] = ev.spiked[:system["pattern_size"]].sum()

        results.append({
            "label": label,
            "mean_mismatch": float(mismatches.mean()),
            "total_mismatch": float(mismatches.sum()),
            "allocated": allocated_this_exposure,
            "n_slots": len(art.slots),
        })
    return results
