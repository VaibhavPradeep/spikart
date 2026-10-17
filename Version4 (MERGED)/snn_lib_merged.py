"""
snn_lib_merged.py
=================
Unified Version 4 library combining ART substrate allocation, STDP 
learning rules, and attention-routed inhibition loops into a single module.
"""
import numpy as np


# =========================================================================
# 1. LIF POPULATION
# =========================================================================
class LIFPopulation:
    """Vectorized leaky integrate-and-fire population. Supports either an
    explicit boolean inhibitory mask (any neuron can be excitatory or
    inhibitory, in any order) or the simpler "last n_inhibitory neurons are
    inhibitory" convention -- whichever the caller finds more convenient."""

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
        self.trace = np.zeros(n_neurons)      # STDP / activity eligibility trace
        self.rectify_input = False            # explicit default (Gen 2's cleanliness;
                                               # Gen 1 set this ad hoc post-construction)

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


# =========================================================================
# 2. PROJECTION
# =========================================================================
class Projection:
    """weights shape: (n_target, n_source). Current-based synapse: spikes
    are filtered through a decaying trace (tau_syn) instead of delivered as
    a single-timestep impulse (NEST/Brian2-style).

    active_mask (merged from Gen 1): which TARGET rows are currently
    plastic-eligible under STDP. None = every mask-connected row is
    eligible (both generations' original behavior). An external allocator
    (SlottedARTMemory, via run_schedule) can set this to just the neurons
    belonging to the slot that "owns" the pattern being presented right
    now, so unrelated committed slots don't drift from incidental
    co-activation with other patterns.
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
        self.mask = (self.weights != 0) if mask is None else np.array(mask, dtype=bool)
        self.active_mask = None

    def set_active_mask(self, target_row_mask):
        self.active_mask = None if target_row_mask is None else np.array(target_row_mask, dtype=bool)

    def pulse_and_decay(self, dt=0.1):
        self.psc *= np.exp(-dt / self.tau_syn)
        self.psc += self.source.spiked.astype(float)

    def current(self):
        return self.weights @ self.psc

    def update(self, pattern=None, dt=0.1):
        if self.plasticity is None:
            return
        if self.plasticity == "hebbian":
            # legacy bulk-update rule (kept for backward compatibility with
            # simple two-population scripts); not used by the ART network.
            if pattern is None:
                return
            if self.gate is not None and not self.gate():
                return
            self.weights += self.lr * np.outer(pattern, pattern)
        elif self.plasticity == "stdp":
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


# =========================================================================
# 3. NETWORK
# =========================================================================
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


# =========================================================================
# 4. GREEDY DPP MAP SELECTION
# =========================================================================
def greedy_dpp_map(kernel, k):
    """Greedy MAP inference for a k-DPP given an explicit PSD kernel L
    (n x n), following Chen, Zhang & Zhou (2018): O(k^2 n) via incremental
    Cholesky, no determinant ever computed explicitly. Reference
    implementation for when a kernel matrix is already what you have."""
    n = kernel.shape[0]
    d2 = np.diag(kernel).copy().astype(float)
    selected = []
    c = []
    for _ in range(min(k, n)):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        selected.append(j)
        Lj = kernel[j, :].astype(float)
        if c:
            C = np.vstack(c)
            proj = C[:, j] @ C
            e = (Lj - proj) / np.sqrt(d2[j])
        else:
            e = Lj / np.sqrt(d2[j])
        c.append(e)
        d2 = d2 - e ** 2
        d2[j] = -np.inf
    return selected


def greedy_dpp_map_from_features(B, k, eps=1e-8):
    """Same greedy MAP algorithm as greedy_dpp_map(), but operating
    directly on a low-rank factor B where the implied kernel is L = B B^T
    (row i of B = quality_i * unit_direction_i -- a quality-diversity
    kernel). This is the version SlottedARTMemory actually uses: at a
    substrate of hundreds of neurons, forming the full N x N kernel just
    to throw it away is wasted memory for no benefit, so selection is done
    by incremental Gram-Schmidt residual-norm maximization on B directly
    -- mathematically the same greedy log-det objective as greedy_dpp_map,
    just without ever materializing L."""
    N = B.shape[0]
    residual = B.copy()
    selected_mask = np.zeros(N, dtype=bool)
    selected = []
    for _ in range(min(k, N)):
        scores = np.sum(residual ** 2, axis=1)
        scores[selected_mask] = -np.inf
        i = int(np.argmax(scores))
        if scores[i] <= eps:
            break
        selected_mask[i] = True
        selected.append(i)
        v = residual[i] / (np.linalg.norm(residual[i]) + eps)
        residual = residual - np.outer(residual @ v, v)
    return selected


# =========================================================================
# 5. ATTENTION ROUTER
# =========================================================================
class AttentionRouter:
    """Softmax attention over per-slot match/relevance scores. Decides how
    much of each memory category's inhibitory ("already explained") signal
    is allowed to reach the mismatch evaluator -- a stand-in for top-down
    attentional gating / precision-weighting of prediction error."""

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


# =========================================================================
# 6. SLOTTED ART MEMORY (vigilance test + DPP allocation + routed inhibition)
# =========================================================================
class SlottedARTMemory:
    """
    Manages a shared substrate as a set of ART-style category slots.

    route(pattern)      -- symbolic vigilance test: Jaccard match against
                            each committed slot's key (frozen template, or
                            the slot's current learned receptive field if
                            use_learned_keys=True). Purely symbolic -- no
                            network dynamics involved, so it's O(1) in
                            simulation cost and doesn't suffer the
                            probe-vs-fast-commit ordering problem that
                            spike-based vigilance tests have to guard
                            against explicitly (see run_schedule()).

    allocate(...)        -- grows a new slot out of the substrate. Diversity
                            against every slot allocated so far is enforced
                            two ways at once:
                              - reuse_count damping (soft): a candidate's
                                DPP "quality" is discounted by how many
                                times it has already been claimed.
                              - persistent orthogonal projection (hard):
                                candidate receptive-field directions are
                                projected out of the span of every
                                previously-claimed direction before DPP
                                selection runs, so a new slot's members
                                are geometrically distinct from every slot
                                committed so far, not just individually
                                discounted.

    inhibition_for(...)  -- the routed inhibitory current delivered to the
                            evaluator population this tick.

    use_dpp, use_attention, and use_learned_keys are independent toggles
    so a caller can run a full ablation over any combination.
    """

    def __init__(self, substrate_size, slot_size, pattern_size, vigilance=0.75,
                 use_dpp=True, use_attention=True, attn_temperature=0.15,
                 boost=6.0, seed=0, use_learned_keys=False):
        self.substrate_size = substrate_size
        self.slot_size = slot_size
        self.pattern_size = pattern_size
        self.vigilance = vigilance
        self.use_dpp = use_dpp
        self.use_attention = use_attention
        self.use_learned_keys = use_learned_keys
        self.boost = boost
        self.rng = np.random.default_rng(seed)
        self.reuse_count = np.zeros(substrate_size, dtype=int)
        self.slots = []  # each: {"neurons", "template", "alloc_quality", "alloc_redundancy"}
        self.router = AttentionRouter(temperature=attn_temperature)
        self._key_source = None            # optional Input->Memory Projection
        self._used_directions = []         # one unit vector per slot ever allocated
                                            # (the slot's own boosted receptive-field
                                            # direction), used to softly discourage new
                                            # slots from re-treading old ones

    def attach_key_source(self, projection):
        """Point at the Input->Memory Projection so match scores can
        (optionally) reflect each slot's *current* learned receptive field
        rather than only the pattern frozen in at allocation time."""
        self._key_source = projection

    def _slot_key(self, slot):
        if self.use_learned_keys and self._key_source is not None:
            rows = self._key_source.weights[slot["neurons"], :]
            key = rows.mean(axis=0)
            if key.sum() > 1e-9:
                return key
        return slot["template"]

    def match_scores(self, pattern):
        """Jaccard similarity |p AND key| / |p OR key| between the input
        pattern and each committed slot's key. Symmetric, unlike classic
        ART's directional subset-match ratio -- with dense random binary
        patterns a one-directional criterion produces frequent chance
        "matches" whenever a key happens to be a superset of a sparser
        pattern's active bits, which would falsely recognize patterns the
        network has never actually seen."""
        if not self.slots:
            return np.array([])
        scores = []
        for s in self.slots:
            t = self._slot_key(s)
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

    def _overlap_penalty(self, featsn):
        """Soft discount in (0, 1]: how much each (unit-normalized)
        candidate feature vector already overlaps with a direction some
        EARLIER slot claimed, summed across every prior slot. Deliberately
        soft rather than a hard Gram-Schmidt projection: a hard projection
        against a growing basis saturates once enough slots have been
        claimed to span the whole pattern space (pattern_size dimensions
        exhausted after pattern_size / slot_size or fewer allocations),
        at which point it would zero out EVERY candidate's signal instead
        of merely discouraging overlap -- this stays graceful past that
        point, asymptoting toward a floor instead of collapsing."""
        if not self._used_directions:
            return np.ones(featsn.shape[0])
        U = np.array(self._used_directions)                    # (m, pattern_size)
        proj_energy = np.sum((featsn @ U.T) ** 2, axis=1)       # >= 0, grows with overlap
        return 1.0 / (1.0 + proj_energy)

    def allocate(self, pattern, W_in):
        """W_in: Input->Memory weight matrix (substrate_size x pattern_size
        for the excitatory rows), modified in place with a small additive
        resonance boost for chosen neurons (fast-learning ART commit).
        Also records, for every allocation regardless of use_dpp, the mean
        pre-existing quality and mean pairwise redundancy of the chosen
        neurons -- a direct, deterministic measure of what the selection
        method actually bought, independent of how the rest of the
        network's spiking dynamics happen to play out."""
        unallocated = np.where(self.reuse_count == 0)[0]
        pool = unallocated if len(unallocated) >= self.slot_size else np.arange(self.substrate_size)
        k = min(self.slot_size, len(pool))

        feats_all = W_in[pool, :]
        norms_all = np.linalg.norm(feats_all, axis=1, keepdims=True) + 1e-9
        featsn_all = feats_all / norms_all
        pat_n = pattern / (np.linalg.norm(pattern) + 1e-9)
        q_all = np.clip(featsn_all @ pat_n, 1e-3, None)

        if self.use_dpp:
            q_reuse = q_all / (1.0 + self.reuse_count[pool])            # soft anti-reuse
            q_cross = self._overlap_penalty(featsn_all)                  # soft anti-overlap
            q = q_reuse * q_cross
            B = q[:, None] * featsn_all
            chosen_local = greedy_dpp_map_from_features(B, k)
            if len(chosen_local) < k:
                remaining = [i for i in range(len(pool)) if i not in chosen_local]
                extra = self.rng.choice(remaining, k - len(chosen_local), replace=False)
                chosen_local = chosen_local + list(extra)
        else:
            chosen_local = list(self.rng.choice(len(pool), size=k, replace=False))

        chosen = pool[chosen_local]
        mean_quality = float(q_all[chosen_local].mean())
        S_all = featsn_all @ featsn_all.T
        sub_S = S_all[np.ix_(chosen_local, chosen_local)]
        mean_redundancy = float(sub_S[~np.eye(k, dtype=bool)].mean()) if k > 1 else 0.0

        self.reuse_count[chosen] += 1
        W_in[chosen, :] += self.boost * pattern
        np.clip(W_in, 0.0, 45.0, out=W_in)

        if self.use_dpp:
            # record this slot's own average (post-boost) receptive-field
            # direction so future allocations softly discount overlap with it
            new_dir = W_in[chosen, :].mean(axis=0)
            n = np.linalg.norm(new_dir)
            if n > 1e-9:
                self._used_directions.append(new_dir / n)

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

    def owner_mask(self, mem_n, slot_index):
        mask = np.zeros(mem_n, dtype=bool)
        mask[self.slots[slot_index]["neurons"]] = True
        return mask

    def inhibition_for(self, pattern, substrate_trace, inhibition_strength):
        needs_alloc, attn = self.route(pattern)
        inhib = np.zeros(self.pattern_size)
        for w, slot in zip(attn, self.slots):
            activity = substrate_trace[slot["neurons"]].mean()
            inhib += w * activity * slot["template"]
        return needs_alloc, inhib * inhibition_strength


# =========================================================================
# 7. NETWORK-BUILDING / EXPERIMENT-RUNNING HELPERS
# =========================================================================
def internal_weights(n_exc, n_inh, rng, ref_n_exc=8, ref_n_inh=2):
    """Recurrent E/I connectivity for a competitive population, per-
    connection strengths mean-field normalized against a reference
    population size so the *aggregate* recurrent current a neuron receives
    stays in the same regime regardless of how large the substrate is
    scaled up -- without this, weights calibrated for ~10 neurons make a
    several-hundred-neuron substrate's shared inhibitory pool massively
    over- or under-powered."""
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
    vectors can collide by chance (especially at small pattern_size), which
    would confuse the ART vigilance test; this keeps the generated pattern
    set honestly separable so mismatches reflect network behaviour rather
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


def force_overlap(patterns, source_idx, target_indices, frac=0.5):
    """Deliberately make `target_indices` patterns share a `frac` fraction
    of `source_idx`'s active bits, on top of an otherwise well-separated
    set. well_separated_patterns() alone tends to produce patterns so
    mutually dissimilar that interference basically never occurs by
    construction -- a poor testbed for an ablation whose whole point is
    measuring interference under overlap. Returns a mutated copy; safe to
    call repeatedly with different indices."""
    patterns = [p.copy() for p in patterns]
    src = patterns[source_idx]
    active = np.where(src == 1)[0]
    n_share = max(1, int(len(active) * frac))
    shared = active[:n_share]
    for idx in target_indices:
        patterns[idx][shared] = 1.0
    return patterns


def build_art_network(pattern_size, substrate_size, slot_size, vigilance,
                       use_dpp, use_attention, use_stdp, seed=0,
                       n_inh_frac=0.25, attn_temperature=0.15, boost=8.0,
                       tau_syn_in_mem=10.0, use_learned_keys=False):
    """Assembles Input / Memory-substrate / Evaluator populations, wires the
    (STDP-plastic or static) Input->Memory projection and the recurrent E/I
    projections, and creates the SlottedARTMemory that manages allocation
    and routed inhibition."""
    rng = np.random.default_rng(seed)
    n_inh_sub = max(2, int(substrate_size * n_inh_frac))
    n_inh_ev = 2

    net = Network()
    inp = net.add_population(LIFPopulation(pattern_size, n_inhibitory=0, name="Input"))
    mem = net.add_population(LIFPopulation(substrate_size + n_inh_sub,
                                            n_inhibitory=n_inh_sub, name="Memory"))
    ev = net.add_population(LIFPopulation(pattern_size + n_inh_ev, n_inhibitory=n_inh_ev,
                                           name="Evaluator"))
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
                            attn_temperature=attn_temperature, boost=boost, seed=seed,
                            use_learned_keys=use_learned_keys)
    art.attach_key_source(proj_in_mem)

    return {"net": net, "art": art, "input": inp, "memory": mem, "evaluator": ev,
            "proj_in_mem": proj_in_mem, "substrate_size": substrate_size,
            "pattern_size": pattern_size}


def run_schedule(system, schedule, steps_per_pattern, dt=0.1,
                  input_drive=180.0, sensory_drive=55.0, inhibition_strength=20.0,
                  probe_steps=40, gate_learning_to_owner=True):
    """Runs a list of (label, pattern) exposures through an already-built
    system. Each exposure now has TWO phases, fixing a measurement-order
    bug present in the naive version of this loop: allocating and fast-
    committing a new slot on tick 0 of an exposure, then averaging mismatch
    over the WHOLE exposure (including tick 0 onward), silently deflates
    the reported "how novel did this look" number by the very commit that
    was triggered *because* it looked novel.

      1. PROBE (`probe_steps` ticks, active_mask cleared -> no learning):
         measures probe_mismatch, the genuine pre-commit novelty signal,
         with the slot state exactly as it stood before this exposure.
      2. The (symbolic, instant) vigilance decision is made once, using
         the same pre-exposure slot state the probe just measured against.
         If novel, a new slot is allocated/fast-committed now, once --
         not re-decided on every remaining tick.
      3. MAIN WINDOW (remaining ticks): learning proceeds, gated (if
         gate_learning_to_owner) to only the slot now recognized as
         "owning" this pattern -- the newly allocated slot, or the
         highest-attention existing slot -- so committed-but-unrelated
         slots don't drift from incidental co-activation.

    Returns a list of per-exposure dicts: label, probe_mismatch (new),
    mean_mismatch / total_mismatch (main window only), allocated, n_slots.
    """
    net, art = system["net"], system["art"]
    inp, mem, ev = system["input"], system["memory"], system["evaluator"]
    proj_in_mem = system["proj_in_mem"]
    substrate_size = system["substrate_size"]
    results = []

    for label, pattern in schedule:
        pattern = np.asarray(pattern, dtype=float)

        needs_alloc, _ = art.route(pattern)  # only the decision is used here; see below
                                              # for why ownership is re-derived separately
                                              # rather than reusing this call's attn

        # --- phase 1: genuine pre-commit probe, no learning ---
        # active_mask=None means "unrestricted" (see Projection.update), the
        # OPPOSITE of what a clean probe needs -- an explicit all-False mask
        # is what actually silences STDP for this phase.
        proj_in_mem.set_active_mask(np.zeros(mem.n, dtype=bool))
        p_steps = min(probe_steps, steps_per_pattern)
        probe_total = 0.0
        for _ in range(p_steps):
            _, inhib = art.inhibition_for(pattern, mem.trace[:substrate_size], inhibition_strength)
            ev_drive = np.maximum(pattern * sensory_drive - inhib, 0.0)
            ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
            net.step({"Input": pattern * input_drive, "Evaluator": ext_ev}, dt=dt)
            probe_total += ev.spiked[:system["pattern_size"]].sum()
        probe_mismatch = probe_total / max(p_steps, 1)

        # --- phase 2: commit decision (uses pre-exposure state only) ---
        allocated_this_exposure = False
        if needs_alloc:
            art.allocate(pattern, proj_in_mem.weights)
            allocated_this_exposure = True

        owner_mask = None
        if gate_learning_to_owner and art.slots:
            if allocated_this_exposure:
                owner_idx = len(art.slots) - 1
            else:
                # Deliberately re-derive ownership from raw match_scores, NOT
                # from attn0. attn0 is the ATTENTION-WEIGHTED distribution --
                # exactly the thing use_attention=False ablates by flattening
                # it to uniform. Using attn0 here means an ablated attention
                # mechanism would silently corrupt an unrelated measurement
                # (which slot's neurons get to learn) via np.argmax's
                # first-index tie-break: every recognized pattern's STDP
                # would get redirected to slot 0 whenever use_attention is
                # False, regardless of which slot it actually matches.
                # Ownership is a question of genuine match, independent of
                # how inhibition happens to be routed.
                raw_scores = art.match_scores(pattern)
                owner_idx = int(np.argmax(raw_scores)) if raw_scores.size else None
            if owner_idx is not None:
                owner_mask = art.owner_mask(mem.n, owner_idx)
        proj_in_mem.set_active_mask(owner_mask)

        # --- phase 3: main window ---
        remaining = max(steps_per_pattern - p_steps, 0)
        mismatches = np.zeros(remaining)
        for t in range(remaining):
            _, inhib = art.inhibition_for(pattern, mem.trace[:substrate_size], inhibition_strength)
            ev_drive = np.maximum(pattern * sensory_drive - inhib, 0.0)
            ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
            net.step({"Input": pattern * input_drive, "Evaluator": ext_ev}, dt=dt)
            mismatches[t] = ev.spiked[:system["pattern_size"]].sum()

        results.append({
            "label": label,
            "probe_mismatch": float(probe_mismatch),
            "mean_mismatch": float(mismatches.mean()) if remaining else float(probe_mismatch),
            "total_mismatch": float(mismatches.sum()),
            "allocated": allocated_this_exposure,
            "n_slots": len(art.slots),
        })

    return results


def probe_only(system, pattern, steps=100, dt=0.1, input_drive=180.0,
                sensory_drive=55.0, inhibition_strength=20.0):
    """Read-only measurement: presents `pattern` for `steps` ticks with
    learning hard-disabled and no allocation, and returns the mean
    mismatch (evaluator spike rate). This is the same "clean read" phase
    run_schedule() uses internally, factored out for callers who want to
    check how something currently looks WITHOUT changing anything -- e.g.
    testing for interference on an already-learned pattern after other
    patterns have since been learned, or checking sensitivity to a
    pattern that was deliberately never trained at all."""
    net, art = system["net"], system["art"]
    mem, ev = system["memory"], system["evaluator"]
    proj_in_mem = system["proj_in_mem"]
    substrate_size = system["substrate_size"]
    pattern = np.asarray(pattern, dtype=float)

    proj_in_mem.set_active_mask(np.zeros(mem.n, dtype=bool))
    total = 0.0
    for _ in range(steps):
        _, inhib = art.inhibition_for(pattern, mem.trace[:substrate_size], inhibition_strength)
        ev_drive = np.maximum(pattern * sensory_drive - inhib, 0.0)
        ext_ev = np.pad(ev_drive, (0, ev.n_inh), constant_values=0.0)
        net.step({"Input": pattern * input_drive, "Evaluator": ext_ev}, dt=dt)
        total += ev.spiked[:system["pattern_size"]].sum()
    proj_in_mem.set_active_mask(None)
    return total / max(steps, 1)