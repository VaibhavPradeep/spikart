"""
snn_lib.py
==========
Production release of the Spiking Neural Network (SNN) library featuring 
min-max normalized learned keys, vectorized routing, state-restoration probing,
and noise-tolerant ART substrate dynamics.
"""

import hashlib
import math
import warnings
import numpy as np

# Discarded settling ticks (test pattern on, learning off) run before every
# probe measurement when state_mode="rest". 0 would measure the cold-start
# transient; see RESULTS.md: from rest, the evaluator transient (rebound spikes) is over by ~400 ticks (40 ms).
DEFAULT_PROBE_WARMUP = 400


# =========================================================================
# 1. LIF POPULATION
# =========================================================================
class LIFPopulation:
    """Vectorized leaky integrate-and-fire population. Supports either an
    explicit boolean inhibitory mask or the simpler "last n_inhibitory
    neurons are inhibitory" convention."""

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
        self.rectify_input = False

    def step(self, I, dt=0.1):
        active = self.refractory <= 0
        # values for refractory neurons are computed but discarded by the where below,
        # so the result is identical to masking dv first
        dv = (-(self.v - self.v_rest) + self.R * I) / self.tau_m * dt
        v = np.where(active, self.v + dv, self.v_reset)
        refractory = np.where(active, self.refractory, self.refractory - dt)
        fired = active & (v >= self.v_th)
        if fired.any():
            v[fired] = self.v_reset
            refractory[fired] = self.t_refractory
        self.v, self.refractory, self.spiked = v, refractory, fired
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
    """weights shape: (n_target, n_source). Current-based synapse: spikes are
    filtered through a decaying trace (tau_syn).

    active_mask: which TARGET rows are plastic-eligible under STDP.
    None = every mask-connected row is eligible; an all-False mask silences
    STDP completely (used by probes).
    """

    def __init__(self, source, target, weights, plasticity=None,
                 learning_rate=1.0, A_plus=0.06, A_minus=0.072,
                 w_min=0.0, w_max=45.0, tau_syn=5.0, mask=None,
                 gate=None, name=""):
        self.source, self.target = source, target
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape != (target.n, source.n):
            raise ValueError(f"weights shape {self.weights.shape} != (target.n, source.n) = "
                             f"{(target.n, source.n)}")
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
            changed = False
            if ltp_rows.any():
                self.weights[ltp_rows, :] += (self.A_plus * pre_trace[np.newaxis, :]
                                              * self.mask[ltp_rows, :])
                changed = True
            if pre_fired.any() and row_gate.any():
                ltd = (self.A_minus * post_trace[:, np.newaxis] * self.mask[:, pre_fired]
                       * row_gate[:, np.newaxis])
                self.weights[:, pre_fired] -= ltd
                changed = True
            if not changed:
                return
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
        # shape is validated in Projection.__init__ (survives python -O)
        self.projections.append(proj)
        return proj

    def step(self, external_currents, dt=0.1, pattern=None):
        totals = {name: np.array(external_currents.get(name, np.zeros(p.n)), dtype=float)
                  for name, p in self.populations.items()}
        for proj in self.projections:
            proj.pulse_and_decay(dt)
            totals[proj.target.name] += proj.current()
        for name, pop in self.populations.items():
            t = totals[name]
            if getattr(pop, "rectify_input", False):
                t = np.maximum(t, 0.0)
            pop.step(t, dt)
        for proj in self.projections:
            proj.update(pattern=pattern, dt=dt)

    def reset(self):
        """Return all DYNAMICAL state to rest (voltages, refractory timers,
        spike flags, traces, psc). Weights and plasticity masks are untouched."""
        for p in self.populations.values():
            p.reset()
        for proj in self.projections:
            proj.psc[:] = 0.0


# =========================================================================
# 4. GREEDY DPP MAP SELECTION
# =========================================================================
def greedy_dpp_map(kernel, k):
    """Greedy MAP inference for a k-DPP given an explicit PSD kernel L
    (n x n), following Chen, Zhang & Zhou (2018): incremental Cholesky, no
    determinant computed explicitly."""
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
        d2 = np.maximum(d2, 0.0)
        d2[j] = -np.inf
    return selected


def greedy_dpp_map_from_features(B, k, eps=1e-8):
    """Same greedy MAP objective as greedy_dpp_map(), operating on a low-rank
    factor B (kernel L = B B^T, row i = quality_i * unit_direction_i), via
    incremental Gram-Schmidt residual-norm maximization. Never forms L.
    Stops early once the residual energy is <= eps (rank(B) is exhausted)."""
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
    """Softmax attention over per-slot match scores; decides how much of each
    category's inhibitory ("already explained") signal reaches the mismatch
    evaluator."""

    def __init__(self, temperature=0.15):
        self.temperature = temperature
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.weights = None

    def compute(self, match_scores):
        m = np.asarray(match_scores, dtype=float)
        m = m - m.max()
        w = np.exp(m / self.temperature)
        w /= w.sum() + 1e-12
        self.weights = w
        return w


# =========================================================================
# 6. SLOTTED ART MEMORY
# =========================================================================
class SlottedARTMemory:
    """
    Manages a shared substrate as a set of ART-style category slots.

    route(pattern)       -- symbolic vigilance test (Jaccard vs each slot key).
    allocate(...)        -- grows a new slot. Selection (when use_dpp=True) is
                            greedy quality-diversity MAP over a quality-weighted
                            feature matrix, with two optional SOFT quality
                            modifiers that are our own additions:
                              use_reuse_damping:   q / (1 + times claimed)
                              use_overlap_penalty: q / (1 + overlap energy with
                                                   earlier slots' directions)
                            Nothing is projected out. use_dpp=False replaces
                            ALL selection logic with uniform random choice
                            (and disables both modifiers); the resonance boost
                            is applied either way.
    inhibition_for(...)  -- routed inhibitory current for the evaluator.

    use_reuse_damping / use_overlap_penalty default to "follow use_dpp", which
    reproduces the original bundled behaviour; set them explicitly to ablate
    the components one at a time. They require use_dpp=True.
    """

    def __init__(self, substrate_size, slot_size, pattern_size, vigilance=0.75,
                 use_dpp=True, use_attention=True, attn_temperature=0.15,
                 boost=6.0, seed=0, use_learned_keys=False,
                 use_reuse_damping=None, use_overlap_penalty=None):
        if pattern_size < 1 or substrate_size < 1 or slot_size < 1:
            raise ValueError("pattern_size, substrate_size and slot_size must be >= 1")
        if not (0.0 < vigilance <= 1.0):
            raise ValueError(f"vigilance must be in (0, 1], got {vigilance}")
        if boost < 0:
            raise ValueError(f"boost must be >= 0, got {boost}")
        if slot_size > substrate_size:
            raise ValueError(f"slot_size ({slot_size}) cannot exceed substrate_size ({substrate_size})")
        if not use_dpp and (use_reuse_damping or use_overlap_penalty):
            raise ValueError("use_reuse_damping / use_overlap_penalty modify DPP quality "
                             "and require use_dpp=True")
        self.substrate_size = substrate_size
        self.slot_size = slot_size
        self.pattern_size = pattern_size
        self.vigilance = vigilance
        self.use_dpp = bool(use_dpp)
        self.use_reuse_damping = self.use_dpp if use_reuse_damping is None else bool(use_reuse_damping)
        self.use_overlap_penalty = self.use_dpp if use_overlap_penalty is None else bool(use_overlap_penalty)
        self.use_attention = use_attention
        self.use_learned_keys = use_learned_keys
        self.boost = boost
        self.rng = np.random.default_rng(seed)
        self.reuse_count = np.zeros(substrate_size, dtype=int)
        self.slots = []  # each: {"neurons", "template", "alloc_quality", "alloc_redundancy"}
        self.router = AttentionRouter(temperature=attn_temperature)
        self.n_fallback_allocations = 0
        self._key_source = None
        self._used_directions = []
        # stacked caches (kept in sync by allocate) for vectorized routing
        self._tmpl = np.zeros((0, pattern_size))
        self._nidx = np.zeros((0, slot_size), dtype=int)

    def attach_key_source(self, projection):
        self._key_source = projection

    def _check_pattern(self, pattern):
        p = np.asarray(pattern, dtype=float)
        if p.shape != (self.pattern_size,):
            raise ValueError(f"pattern must have shape ({self.pattern_size},), got {p.shape}")
        return p

    @staticmethod
    def _normalize_keys(raw):
        """Rescale each learned receptive field to [0, 1] (min-max per slot) so it is
        commensurable with the binary patterns the Jaccard score and the inhibition
        (which scales as strength * activity * key) were calibrated for. Raw mean
        weights can reach 45, which previously made learned-key matching and
        inhibition scale with weight magnitude rather than with the pattern."""
        lo = raw.min(axis=-1, keepdims=True)
        hi = raw.max(axis=-1, keepdims=True)
        return (raw - lo) / (hi - lo + 1e-9)

    def _keys(self):
        """(n_slots, pattern_size) keys, used by BOTH matching and inhibition: frozen
        templates, or (use_learned_keys) the min-max-normalized mean learned
        receptive field of each slot's neurons (template if the field is flat)."""
        if self.use_learned_keys and self._key_source is not None and len(self.slots):
            raw = self._key_source.weights[self._nidx].mean(axis=1)
            learned = self._normalize_keys(raw)
            ok = (raw.max(axis=1) - raw.min(axis=1)) > 1e-9
            return np.where(ok[:, None], learned, self._tmpl)
        return self._tmpl

    def _slot_key(self, slot):
        if self.use_learned_keys and self._key_source is not None:
            raw = self._key_source.weights[slot["neurons"], :].mean(axis=0)
            if raw.max() - raw.min() > 1e-9:
                return self._normalize_keys(raw)
        return slot["template"]

    @staticmethod
    def _jaccard(pattern, keys):
        inter = np.minimum(pattern[None, :], keys).sum(axis=1)
        union = np.maximum(pattern[None, :], keys).sum(axis=1)
        return inter / (union + 1e-9)

    def match_scores(self, pattern):
        """Jaccard similarity between the pattern and each committed slot key."""
        pattern = self._check_pattern(pattern)
        if not self.slots:
            return np.array([])
        return self._jaccard(pattern, self._keys())

    def _route_from_scores(self, scores):
        needs_alloc = (scores.size == 0) or (scores.max() < self.vigilance)
        if scores.size == 0:
            attn = scores
        elif self.use_attention:
            attn = self.router.compute(scores)
        else:
            attn = np.full(scores.shape, 1.0 / len(scores))
        return needs_alloc, attn

    def route(self, pattern):
        return self._route_from_scores(self.match_scores(pattern))

    def _overlap_penalty(self, featsn):
        """Soft discount in (0, 1]: overlap of each unit-normalized candidate
        with directions earlier slots claimed. Soft on purpose (a hard
        projection would saturate once the span covers the pattern space)."""
        if not self._used_directions:
            return np.ones(featsn.shape[0])
        U = np.array(self._used_directions)
        proj_energy = np.sum((featsn @ U.T) ** 2, axis=1)
        return 1.0 / (1.0 + proj_energy)

    def allocate(self, pattern, W_in):
        """W_in: Input->Memory weights (substrate rows first), modified in place
        by a small additive resonance boost for the chosen neurons.

        Topology note: the boost adds to EVERY input dimension of the chosen
        rows, so it may create nonzero weights where none existed. This is
        deliberate: allocation is allowed to change topology, STDP is not
        (Projection.mask is fixed at construction). It is harmless with
        build_art_network (excitatory rows start dense) but a sparse initial
        W_in would have boosted synapses STDP can never adjust.

        Records mean pre-existing quality and mean pairwise redundancy of the
        chosen neurons for EVERY allocation, whatever the selection method."""
        pattern = self._check_pattern(pattern)
        if W_in.shape[1] != self.pattern_size or W_in.shape[0] < self.substrate_size:
            raise ValueError(f"W_in shape {W_in.shape} incompatible with "
                             f"substrate_size={self.substrate_size}, pattern_size={self.pattern_size}")
        unallocated = np.where(self.reuse_count == 0)[0]
        if len(unallocated) >= self.slot_size:
            pool = unallocated
        else:
            warnings.warn(
                f"Only {len(unallocated)} unallocated neurons remain for slot_size="
                f"{self.slot_size}; falling back to the whole substrate, so this "
                f"slot will share neurons with earlier slots.", RuntimeWarning)
            self.n_fallback_allocations += 1
            pool = np.arange(self.substrate_size)
        k = min(self.slot_size, len(pool))

        feats_all = W_in[pool, :]
        norms_all = np.linalg.norm(feats_all, axis=1, keepdims=True) + 1e-9
        featsn_all = feats_all / norms_all
        pat_n = pattern / (np.linalg.norm(pattern) + 1e-9)
        q_all = np.clip(featsn_all @ pat_n, 1e-3, None)

        if self.use_dpp:
            q = q_all.copy()
            if self.use_reuse_damping:
                q = q / (1.0 + self.reuse_count[pool])
            if self.use_overlap_penalty:
                q = q * self._overlap_penalty(featsn_all)
            B = q[:, None] * featsn_all
            chosen_local = greedy_dpp_map_from_features(B, k)
            if len(chosen_local) < k:       # rank exhausted: random top-up
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

        if self.use_overlap_penalty:
            new_dir = W_in[chosen, :].mean(axis=0)
            n = np.linalg.norm(new_dir)
            if n > 1e-9:
                self._used_directions.append(new_dir / n)

        self.slots.append({"neurons": chosen, "template": pattern.copy(),
                           "alloc_quality": mean_quality, "alloc_redundancy": mean_redundancy})
        self._tmpl = np.vstack([self._tmpl, pattern[None, :]])
        nrow = np.full(self.slot_size, -1, dtype=int)
        nrow[:len(chosen)] = chosen
        self._nidx = np.vstack([self._nidx, nrow[None, :]])
        return chosen

    def retrieve(self, excerpt, known=None, top_k=3):
        """Rank committed slots by how well they explain a PARTIAL pattern.

        excerpt: length-pattern_size 0/1 vector. known: optional boolean mask of positions that
        were actually observed (default: all positions). Unobserved positions never count as
        evidence. Primary score = containment of the observed active bits: the fraction of the
        excerpt's active bits that lie inside the slot's template (so a small fragment is not
        punished by the template's size, unlike Jaccard). Ties are broken by Jaccard restricted
        to the observed positions, then by slot index. Returns [(slot, containment,
        restricted_jaccard), ...] best first. An excerpt contained in several templates yields
        tied candidates -- the honest answer there is a ranked list, not one label."""
        v = self._check_pattern(excerpt)
        kn = np.ones(self.pattern_size, bool) if known is None else np.asarray(known, bool)
        if kn.shape != (self.pattern_size,):
            raise ValueError(f"known must have shape ({self.pattern_size},)")
        if not self.slots:
            return []
        ve = v * kn
        T = self._tmpl
        inter = np.minimum(ve[None, :], T * kn[None, :]).sum(axis=1)
        cont = inter / max(ve.sum(), 1e-9)
        rj = inter / (np.maximum(ve[None, :], T * kn[None, :]).sum(axis=1) + 1e-9)
        order = np.lexsort((np.arange(len(T)), -rj, -cont))
        return [(int(i), float(cont[i]), float(rj[i])) for i in order[:top_k]]

    def allocation_diagnostics(self):
        """Construction-level diagnostics, independent of spiking dynamics.
        mean_quality is what greedy selection maximizes (so DPP beating random on
        it is nearly guaranteed -- it is NOT evidence of better recognition);
        mean_redundancy is within-slot pairwise cosine; n_shared_neurons and
        cross_slot_cosine (mean off-diagonal cosine between slots' current mean
        receptive fields; needs attach_key_source) describe cross-slot geometry."""
        if not self.slots:
            nan = float("nan")
            return {"mean_quality": nan, "mean_redundancy": nan,
                    "n_shared_neurons": 0, "cross_slot_cosine": nan,
                    "n_fallback_allocations": self.n_fallback_allocations}
        q = np.mean([s["alloc_quality"] for s in self.slots])
        r = np.mean([s["alloc_redundancy"] for s in self.slots])
        cross = float("nan")
        if self._key_source is not None and len(self.slots) > 1:
            D = self._key_source.weights[self._nidx].mean(axis=1)
            D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            S = D @ D.T
            cross = float(S[~np.eye(len(D), dtype=bool)].mean())
        return {"mean_quality": float(q), "mean_redundancy": float(r),
                "n_shared_neurons": int((self.reuse_count > 1).sum()),
                "cross_slot_cosine": cross,
                "n_fallback_allocations": self.n_fallback_allocations}

    def owner_mask(self, mem_n, slot_index, exc_idx=None):
        mask = np.zeros(mem_n, dtype=bool)
        local = self.slots[slot_index]["neurons"]
        if exc_idx is None:
            mask[local] = True
        else:
            mask[exc_idx[local]] = True
        return mask

    def inhibition_for(self, pattern, substrate_trace, inhibition_strength):
        pattern = self._check_pattern(pattern)
        if not self.slots:
            return True, np.zeros(self.pattern_size)
        keys = self._keys()
        needs_alloc, attn = self._route_from_scores(self._jaccard(pattern, keys))
        activity = substrate_trace[self._nidx].mean(axis=1)
        inhib = (attn * activity) @ keys
        return needs_alloc, inhib * inhibition_strength


# =========================================================================
# 7. NETWORK-BUILDING HELPERS
# =========================================================================
def internal_weights(n_exc, n_inh, rng, ref_n_exc=8, ref_n_inh=2):
    """Recurrent E/I connectivity, per-connection strengths mean-field
    normalized against a small reference population so the AGGREGATE recurrent
    current stays in the same regime as the substrate grows.

    NOTE: per-connection weights scale as ref/n, so at n_exc=480 an E->E weight
    is ~0.017x its small-network value. The aggregate drive is constant BY
    CONSTRUCTION; whether the resulting regime is healthy (firing rates, E/I
    balance) is measured in regime_check.py, not assumed."""
    if n_exc < 1 or n_inh < 1:
        raise ValueError(f"n_exc and n_inh must be >= 1 (got {n_exc}, {n_inh})")
    if ref_n_exc < 1 or ref_n_inh < 1:
        raise ValueError("reference sizes must be >= 1")
    n = n_exc + n_inh
    w = np.zeros((n, n))
    w[:n_exc, :n_exc] = rng.uniform(1.0, 2.5, (n_exc, n_exc)) * (ref_n_exc / n_exc)
    np.fill_diagonal(w[:n_exc, :n_exc], 0.0)
    w[:n_exc, n_exc:] = -6.0 * (ref_n_inh / n_inh)
    w[n_exc:, :n_exc] = 4.0 * (ref_n_exc / n_exc)
    return w


def well_separated_patterns(pattern_size, n_patterns, active_bits=None, seed=0,
                            pool_size=6000):
    """n_patterns distinct fixed-density binary vectors chosen greedily to
    minimize the worst pairwise Jaccard overlap with those already chosen.
    Output is identical to the original loop-based implementation for a given
    seed; this version is vectorized and stops sampling once every possible
    pattern has been found (the original kept drawing duplicates)."""
    rng = np.random.default_rng(seed)
    k = active_bits or max(1, pattern_size // 3)
    target = min(pool_size, math.comb(pattern_size, k))
    seen, keys, attempts = set(), [], 0
    while len(keys) < target and attempts < pool_size * 20:
        attempts += 1
        idx = rng.choice(pattern_size, size=k, replace=False)
        key = tuple(sorted(idx.tolist()))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    M = np.zeros((len(keys), pattern_size))
    for i, key in enumerate(keys):
        M[i, list(key)] = 1.0
    sizes = M.sum(axis=1)

    def jac(c):
        inter = M @ M[c]
        return inter / (sizes + sizes[c] - inter + 1e-9)

    order = np.arange(len(keys))
    chosen = [int(order[0])]
    order = order[1:]
    worst = jac(chosen[0])
    while len(chosen) < n_patterns and len(order):
        order = order[np.argsort(worst[order], kind="stable")]
        c = int(order[0])
        order = order[1:]
        chosen.append(c)
        worst = np.maximum(worst, jac(c))
    return [M[c].copy() for c in chosen]


def force_overlap(patterns, source_idx, target_indices, frac=0.5):
    """Make `target_indices` patterns share a `frac` fraction of `source_idx`'s
    active bits. Returns a mutated copy."""
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
                      tau_syn_in_mem=10.0, use_learned_keys=False,
                      use_reuse_damping=None, use_overlap_penalty=None,
                      memory_recurrence=True, input_gain=1.0):
    """memory_recurrence=False omits the Memory E/I recurrent projection. regime_check.py
    shows it is functionally inert at the tested scales (the inhibitory pool never
    fires; E->E drive is <=11% of external drive and shrinks ~1/n), and it is the
    O(n^2) cost per tick, so dropping it is a large speedup at big substrates.
    The default stays True so published numbers are unchanged.

    input_gain scales the initial Input->Memory weights AND the resonance boost.
    The model is calibrated for ~5 active bits per pattern: a neuron's input
    current grows with the number of active bits, so for denser/larger patterns
    use input_gain ~ 5 / active_bits or every substrate neuron saturates and
    selectivity is lost (see scale_check.py).

    Assembles Input / Memory-substrate / Evaluator populations, the
    (STDP-plastic or static) Input->Memory projection, the recurrent E/I
    projections, and the SlottedARTMemory."""
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

    w_mem = internal_weights(substrate_size, n_inh_sub, rng)      # always drawn: keeps the rng stream
    if memory_recurrence:
        net.add_projection(Projection(mem, mem, w_mem, plasticity=None, tau_syn=5.0,
                                      name="Memory-recurrent"))
    net.add_projection(Projection(ev, ev, internal_weights(pattern_size, n_inh_ev, rng),
                                  plasticity=None, tau_syn=5.0, name="Evaluator-recurrent"))

    w_in_mem = np.zeros((substrate_size + n_inh_sub, pattern_size))
    w_in_mem[:substrate_size, :] = rng.uniform(0.5, 2.5, (substrate_size, pattern_size)) * input_gain
    proj_in_mem = net.add_projection(Projection(
        inp, mem, w_in_mem, plasticity=("stdp" if use_stdp else None),
        A_plus=0.18, A_minus=0.15, w_min=0.0, w_max=45.0, tau_syn=tau_syn_in_mem,
        name="Input->Memory"))

    # allocate() indexes W_in by LOCAL excitatory index and internal_weights()
    # lays excitatory neurons first: both require the "last n are inhibitory" layout.
    assert np.array_equal(mem.exc_idx, np.arange(substrate_size)), \
        "ART path requires excitatory neurons to occupy the first substrate_size slots"

    art = SlottedARTMemory(substrate_size, slot_size, pattern_size, vigilance=vigilance,
                           use_dpp=use_dpp, use_attention=use_attention,
                           attn_temperature=attn_temperature, boost=boost * input_gain, seed=seed,
                           use_learned_keys=use_learned_keys,
                           use_reuse_damping=use_reuse_damping,
                           use_overlap_penalty=use_overlap_penalty)
    art.attach_key_source(proj_in_mem)

    return {"net": net, "art": art, "input": inp, "memory": mem, "evaluator": ev,
            "proj_in_mem": proj_in_mem, "substrate_size": substrate_size,
            "pattern_size": pattern_size}


# =========================================================================
# 8. STATE SNAPSHOT / RESTORE / DIGEST   (todo #5 / #8)
# =========================================================================
def snapshot_state(system):
    """Copy of all DYNAMICAL state (see module docstring)."""
    net, art = system["net"], system["art"]
    return {
        "pops": {n: (p.v.copy(), p.refractory.copy(), p.spiked.copy(), p.trace.copy())
                 for n, p in net.populations.items()},
        "psc": [pr.psc.copy() for pr in net.projections],
        "masks": [None if pr.active_mask is None else pr.active_mask.copy()
                  for pr in net.projections],
        "router": None if art.router.weights is None else art.router.weights.copy(),
    }


def restore_state(system, snap):
    net, art = system["net"], system["art"]
    for n, p in net.populations.items():
        v, r, s, t = snap["pops"][n]
        p.v, p.refractory, p.spiked, p.trace = v.copy(), r.copy(), s.copy(), t.copy()
    for pr, psc, m in zip(net.projections, snap["psc"], snap["masks"]):
        pr.psc = psc.copy()
        pr.active_mask = None if m is None else m.copy()
    art.router.weights = None if snap["router"] is None else snap["router"].copy()


def state_digest(system):
    """SHA-256 over EVERYTHING (dynamical + learned state, slots, RNG state).
    Two systems/readouts with equal digests are in identical states."""
    net, art = system["net"], system["art"]
    h = hashlib.sha256()

    def add(a):
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    for n in sorted(net.populations):
        p = net.populations[n]
        for a in (p.v, p.refractory, p.spiked, p.trace):
            add(a)
    for pr in net.projections:
        add(pr.weights)
        add(pr.psc)
        h.update(b"none" if pr.active_mask is None else pr.active_mask.tobytes())
    add(np.array([]) if art.router.weights is None else art.router.weights)
    add(art.reuse_count)
    for s in art.slots:
        add(s["neurons"])
        add(s["template"])
    for d in art._used_directions:
        add(d)
    h.update(repr(art.rng.bit_generator.state).encode())
    return h.hexdigest()


# =========================================================================
# 9. SHARED PRESENTATION LOOP, run_schedule, probe_only
# =========================================================================
def _run_window(system, pattern, n_steps, dt, input_drive, sensory_drive, inhibition_strength,
                hook=None):
    """The ONE per-tick presentation loop used by training, probing and the
    figure code (so there is no second, subtly different copy to drift).
    Returns per-tick mismatch (# evaluator spikes on the pattern's channels).
    hook(t), if given, is called after every tick (diagnostics only)."""
    net, art = system["net"], system["art"]
    mem, ev = system["memory"], system["evaluator"]
    P = system["pattern_size"]
    exc = mem.exc_idx
    out = np.zeros(n_steps)
    in_ext = pattern * input_drive
    ext_ev = np.zeros(ev.n)
    sens = pattern * sensory_drive
    # Keys (templates) and match scores cannot change inside a window unless
    # learned keys are in use, so routing is computed once; only slot activity
    # changes per tick. Identical numbers to recomputing every tick.
    static = not art.use_learned_keys and len(art.slots) > 0
    if static:
        keys = art._keys()
        _, attn = art._route_from_scores(art._jaccard(pattern, keys))
        nidx = art._nidx
    for t in range(n_steps):
        if static:
            inhib = ((attn * mem.trace[exc][nidx].mean(axis=1)) @ keys) * inhibition_strength
        else:
            _, inhib = art.inhibition_for(pattern, mem.trace[exc], inhibition_strength)
        ext_ev[:P] = np.maximum(sens - inhib, 0.0)
        net.step({"Input": in_ext, "Evaluator": ext_ev}, dt=dt)
        out[t] = ev.spiked[:P].sum()
        if hook is not None:
            hook(t)
    return out


def _check_state_mode(state_mode):
    if state_mode not in ("rest", "legacy"):
        raise ValueError("state_mode must be 'rest' or 'legacy'")


def run_schedule(system, schedule, steps_per_pattern, dt=0.1,
                 input_drive=180.0, sensory_drive=55.0, inhibition_strength=20.0,
                 probe_steps=40, probe_warmup=None, gate_learning_to_owner=True,
                 state_mode="rest", record_traces=False):
    """Run (label, pattern) exposures. Each exposure has a read-only PROBE
    (genuine pre-commit novelty reading) followed by commit + learning.

    state_mode="rest" (default): every exposure starts from rest; the probe
      (probe_warmup discarded ticks + probe_steps measured ticks, learning off)
      starts from rest, and the network is returned to the SAME rest state
      before the commit and main window. The main window runs
      `steps_per_pattern` ticks of learning; the probe is extra, read-only time.
    state_mode="legacy": previous behaviour (no resets; the probe is carved out
      of steps_per_pattern and the main window continues from probed state).

    Returns one dict per exposure: label, probe_mismatch, mean_mismatch /
    total_mismatch (main window), allocated, n_slots, owner_idx, max_match
    (symbolic Jaccard to best existing slot before the decision, nan if none);
    with record_traces also probe_trace, warmup_trace, main_trace (per-tick)."""
    _check_state_mode(state_mode)
    if probe_warmup is None:
        probe_warmup = DEFAULT_PROBE_WARMUP
    net, art = system["net"], system["art"]
    mem = system["memory"]
    proj = system["proj_in_mem"]
    results = []
    drive = dict(dt=dt, input_drive=input_drive, sensory_drive=sensory_drive,
                 inhibition_strength=inhibition_strength)

    for label, pattern in schedule:
        pattern = np.asarray(pattern, dtype=float)
        scores = art.match_scores(pattern)
        needs_alloc = (scores.size == 0) or (scores.max() < art.vigilance)
        max_match = float(scores.max()) if scores.size else float("nan")

        # --- probe (learning off: an explicit all-False mask; None means UNRESTRICTED) ---
        proj.set_active_mask(np.zeros(mem.n, dtype=bool))
        if state_mode == "rest":
            net.reset()
            warm = _run_window(system, pattern, probe_warmup, **drive)
            p_steps = probe_steps
            probe = _run_window(system, pattern, p_steps, **drive)
            net.reset()                      # main window starts from the probe's start state
        else:
            warm = np.zeros(0)
            p_steps = min(probe_steps, steps_per_pattern)
            probe = _run_window(system, pattern, p_steps, **drive)
        probe_mismatch = float(probe.mean()) if p_steps else 0.0

        # --- commit decision (pre-exposure slot state) ---
        allocated = False
        if needs_alloc:
            art.allocate(pattern, proj.weights)
            allocated = True

        owner_idx, owner_mask = None, None
        if gate_learning_to_owner and art.slots:
            if allocated:
                owner_idx = len(art.slots) - 1
            else:
                # ownership = genuine symbolic match, NOT the attention weights
                # (those are flattened by use_attention=False; argmax ties would
                # then hand every recognized pattern to slot 0).
                owner_idx = int(np.argmax(scores)) if scores.size else None
            if owner_idx is not None:
                owner_mask = art.owner_mask(mem.n, owner_idx, exc_idx=mem.exc_idx)
        proj.set_active_mask(owner_mask)

        # --- main window ---
        remaining = steps_per_pattern if state_mode == "rest" else max(steps_per_pattern - p_steps, 0)
        main = _run_window(system, pattern, remaining, **drive)

        rec = {"label": label, "probe_mismatch": probe_mismatch,
               "mean_mismatch": float(main.mean()) if remaining else probe_mismatch,
               "total_mismatch": float(main.sum()), "allocated": allocated,
               "n_slots": len(art.slots), "owner_idx": owner_idx, "max_match": max_match}
        if record_traces:
            rec.update(probe_trace=probe, warmup_trace=warm, main_trace=main)
        results.append(rec)

    proj.set_active_mask(None)   # training finished: no learning gate left behind
    return results


def probe_only(system, pattern, steps=100, dt=0.1, input_drive=180.0,
               sensory_drive=55.0, inhibition_strength=20.0, warmup=None,
               state_mode="rest", return_trace=False, hook=None):
    """Read-only measurement of mean mismatch for `pattern`.

    state_mode="rest": snapshot -> reset to rest -> `warmup` discarded ticks ->
      `steps` measured ticks (learning off, no allocation) -> restore snapshot.
      Leaves the system exactly as it found it (check with state_digest) and
      does not depend on anything presented earlier.
    state_mode="legacy": old behaviour -- continues from whatever state the
      network is in, leaves the advanced state behind, and ends with
      active_mask=None."""
    _check_state_mode(state_mode)
    if warmup is None:
        warmup = DEFAULT_PROBE_WARMUP
    mem = system["memory"]
    proj = system["proj_in_mem"]
    pattern = np.asarray(pattern, dtype=float)
    drive = dict(dt=dt, input_drive=input_drive, sensory_drive=sensory_drive,
                 inhibition_strength=inhibition_strength)
    if state_mode == "rest":
        snap = snapshot_state(system)
        system["net"].reset()
        proj.set_active_mask(np.zeros(mem.n, dtype=bool))
        _run_window(system, pattern, warmup, **drive)
        trace = _run_window(system, pattern, steps, hook=hook, **drive)
        restore_state(system, snap)
    else:
        proj.set_active_mask(np.zeros(mem.n, dtype=bool))
        trace = _run_window(system, pattern, steps, hook=hook, **drive)
        proj.set_active_mask(None)
    value = float(trace.sum() / max(steps, 1))
    return (value, trace) if return_trace else value


def retrieve_spiking(system, excerpt, top_k=3, steps=300, warmup=300, dt=0.1,
                     input_drive=180.0, sensory_drive=55.0, inhibition_strength=20.0):
    """Spiking pattern completion: present the excerpt (unobserved positions off) from rest,
    read-only, and rank slots by the mean firing rate of THEIR substrate neurons -- i.e. which
    learned receptive fields the fragment drives hardest. Uses the learned weights, not the
    stored templates. Leaves the system unchanged (state_digest-checkable).
    Returns [(slot, rate_spikes_per_neuron_per_tick), ...] best first (ties -> lower index)."""
    art, mem = system["art"], system["memory"]
    if not art.slots:
        return []
    x = art._check_pattern(excerpt)
    snap = snapshot_state(system)
    system["net"].reset()
    system["proj_in_mem"].set_active_mask(np.zeros(mem.n, dtype=bool))
    drive = dict(dt=dt, input_drive=input_drive, sensory_drive=sensory_drive,
                 inhibition_strength=inhibition_strength)
    _run_window(system, x, warmup, **drive)
    counts = np.zeros(mem.n)

    def hook(t):
        counts[:] += mem.spiked

    _run_window(system, x, steps, hook=hook, **drive)
    restore_state(system, snap)
    rate = counts[mem.exc_idx][art._nidx].mean(axis=1) / steps
    order = np.lexsort((np.arange(len(rate)), -rate))
    return [(int(i), float(rate[i])) for i in order[:top_k]]
