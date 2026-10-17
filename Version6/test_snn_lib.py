"""Tests for snn_lib_changing.py. Runs under pytest or directly:  python tests/test_snn_lib.py"""
import os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snn_lib_changing import (retrieve_spiking, greedy_dpp_map, greedy_dpp_map_from_features, SlottedARTMemory,
                              AttentionRouter, Projection, LIFPopulation, build_art_network,
                              run_schedule, probe_only, snapshot_state, restore_state,
                              state_digest, well_separated_patterns, force_overlap, internal_weights)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def small_system(seed=0, **kw):
    args = dict(pattern_size=12, substrate_size=48, slot_size=6, vigilance=0.6, use_dpp=True,
                use_attention=True, use_stdp=True, seed=seed, boost=4.0)
    args.update(kw)
    return build_art_network(**args)


def small_patterns(n=6):
    return well_separated_patterns(12, n, active_bits=4, seed=3)


# ---------------- DPP -----------------
def test_kernel_and_feature_versions_agree():
    B = np.random.default_rng(0).normal(size=(30, 8))
    assert greedy_dpp_map(B @ B.T, 6) == greedy_dpp_map_from_features(B, 6)


def test_selection_distinct_and_sized():
    B = np.random.default_rng(1).normal(size=(40, 12))
    sel = greedy_dpp_map_from_features(B, 10)
    assert len(sel) == 10 and len(set(sel)) == 10


def test_duplicate_rows_not_both_selected():
    B = np.random.default_rng(2).normal(size=(10, 6))
    B[1] = B[0] * 2.0
    sel = greedy_dpp_map_from_features(B, 5)
    assert not (0 in sel and 1 in sel)


def test_stops_at_rank_limit():
    B = np.random.default_rng(3).normal(size=(20, 4))
    assert len(greedy_dpp_map_from_features(B, 10)) == 4
    assert len(greedy_dpp_map(B @ B.T, 10)) == 4


# ---------------- validation ----------------
def test_constructor_validation():
    assert raises(ValueError, SlottedARTMemory, 10, 20, 8)
    assert raises(ValueError, SlottedARTMemory, 20, 5, 8, vigilance=1.5)
    assert raises(ValueError, AttentionRouter, 0)
    assert raises(ValueError, SlottedARTMemory, 20, 5, 8, use_dpp=False, use_overlap_penalty=True)
    assert raises(ValueError, internal_weights, 0, 2, np.random.default_rng(0))


def test_pattern_and_weight_shape_checked():
    art = SlottedARTMemory(20, 5, 8)
    assert raises(ValueError, art.match_scores, np.ones(7))
    assert raises(ValueError, art.allocate, np.ones(8), np.zeros((20, 7)))
    a, b = LIFPopulation(3, name="a"), LIFPopulation(4, name="b")
    assert raises(ValueError, Projection, a, b, np.zeros((3, 4)))   # should be (4, 3)


def test_fallback_warning_when_reuse_forced():
    sysm = small_system()
    art, W = sysm["art"], sysm["proj_in_mem"].weights
    pats = small_patterns(9)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for p in pats[:8]:            # 48 neurons / 6 per slot = 8 native slots
            art.allocate(p, W)
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)
        art.allocate(pats[8], W)
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
    assert art.n_fallback_allocations == 1


# ---------------- split DPP flags ----------------
def test_flag_defaults_follow_use_dpp():
    a = SlottedARTMemory(20, 5, 8, use_dpp=True)
    assert a.use_reuse_damping and a.use_overlap_penalty
    b = SlottedARTMemory(20, 5, 8, use_dpp=False)
    assert not b.use_reuse_damping and not b.use_overlap_penalty
    c = SlottedARTMemory(20, 5, 8, use_dpp=True, use_reuse_damping=False)
    assert not c.use_reuse_damping and c.use_overlap_penalty


def test_flags_change_selection_only_through_their_term():
    """greedy-only and bundled differ in tracked directions: only the overlap flag records them."""
    out = {}
    for name, kw in {"greedy": dict(use_reuse_damping=False, use_overlap_penalty=False),
                     "bundle": {}}.items():
        sysm = small_system(**{"use_dpp": True}, **kw)
        for p in small_patterns(5):
            sysm["art"].allocate(p, sysm["proj_in_mem"].weights)
        out[name] = len(sysm["art"]._used_directions)
    assert out["greedy"] == 0 and out["bundle"] == 5


def test_bundled_default_equals_explicit_all_on():
    ch = []
    for kw in ({}, dict(use_reuse_damping=True, use_overlap_penalty=True)):
        sysm = small_system(**kw)
        for p in small_patterns(5):
            sysm["art"].allocate(p, sysm["proj_in_mem"].weights)
        ch.append([s["neurons"].tolist() for s in sysm["art"].slots])
    assert ch[0] == ch[1]


# ---------------- vectorized routing == reference loops ----------------
def _loop_scores(art, pattern):
    out = []
    for s in art.slots:
        t = art._slot_key(s)
        out.append(np.minimum(pattern, t).sum() / (np.maximum(pattern, t).sum() + 1e-9))
    return np.array(out)


def _loop_inhib(art, pattern, trace, strength):
    _, attn = art.route(pattern)
    inhib = np.zeros(art.pattern_size)
    for w, s in zip(attn, art.slots):
        inhib += w * trace[s["neurons"]].mean() * art._slot_key(s)
    return inhib * strength


def test_vectorized_match_and_inhibition_match_reference():
    for learned in (False, True):
        sysm = small_system(use_learned_keys=learned)
        art, W = sysm["art"], sysm["proj_in_mem"].weights
        pats = small_patterns(6)
        for p in pats[:5]:
            art.allocate(p, W)
        trace = np.random.default_rng(5).uniform(0, 3, 48)
        for p in pats:
            assert np.allclose(art.match_scores(p), _loop_scores(art, p))
            assert np.allclose(art.inhibition_for(p, trace, 20.0)[1], _loop_inhib(art, p, trace, 20.0))


# ---------------- #5/#8: probes are read-only and history-free ----------------
def trained_system(seed=0, **kw):
    sysm = small_system(seed=seed, **kw)
    pats = small_patterns(6)
    sched = [(f"p{i}", p) for i, p in enumerate(pats[:4])]
    run_schedule(sysm, sched, steps_per_pattern=120, probe_steps=40, probe_warmup=60)
    return sysm, pats


def test_snapshot_restore_roundtrip():
    sysm, pats = trained_system()
    d0 = state_digest(sysm)
    snap = snapshot_state(sysm)
    sysm["net"].step({"Input": pats[0] * 180.0}, dt=0.1)
    assert state_digest(sysm) != d0
    restore_state(sysm, snap)
    assert state_digest(sysm) == d0


def test_probe_only_is_side_effect_free_and_repeatable():
    sysm, pats = trained_system()
    d0 = state_digest(sysm)
    vals = [probe_only(sysm, pats[0], steps=60, warmup=60) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]
    for p in pats:
        probe_only(sysm, p, steps=60, warmup=60)
    assert state_digest(sysm) == d0, "probe changed system state"


def test_probe_only_is_independent_of_probe_order_and_history():
    s1, pats = trained_system()
    s2, _ = trained_system()
    a = [probe_only(s1, p, steps=60, warmup=60) for p in pats]
    b = [probe_only(s2, p, steps=60, warmup=60) for p in reversed(pats)][::-1]
    assert a == b


def test_legacy_probe_mode_does_leave_state_behind():
    """Documents the old defect: legacy probes advance state and reset the mask to None."""
    sysm, pats = trained_system()
    d0 = state_digest(sysm)
    probe_only(sysm, pats[0], steps=60, state_mode="legacy")
    assert state_digest(sysm) != d0


def test_pre_commit_probe_in_schedule_equals_probe_only():
    sysm, pats = trained_system()
    v = probe_only(sysm, pats[5], steps=50, warmup=70)      # novel pattern, before any commit
    ex = run_schedule(sysm, [("novel", pats[5])], steps_per_pattern=30, probe_steps=50, probe_warmup=70)
    assert ex[0]["allocated"] and ex[0]["probe_mismatch"] == v


def test_run_schedule_rest_mode_is_deterministic():
    r = []
    for _ in range(2):
        sysm = small_system()
        sched = [(f"p{i}", p) for i, p in enumerate(small_patterns(4))]
        ex = run_schedule(sysm, sched, steps_per_pattern=100, probe_steps=40, probe_warmup=40)
        r.append(([e["probe_mismatch"] for e in ex], state_digest(sysm)))
    assert r[0] == r[1]


def test_record_traces_lengths_and_commit_ordering():
    sysm = small_system()
    ex = run_schedule(sysm, [("a", small_patterns(2)[0])], steps_per_pattern=70,
                      probe_steps=30, probe_warmup=20, record_traces=True)
    e = ex[0]
    assert len(e["warmup_trace"]) == 20 and len(e["probe_trace"]) == 30 and len(e["main_trace"]) == 70
    assert e["allocated"] and abs(e["probe_mismatch"] - e["probe_trace"].mean()) < 1e-12


# ---------------- pattern generator unchanged vs original implementation ----------------
def _original_well_separated(pattern_size, n_patterns, active_bits=None, seed=0, pool_size=6000):
    rng = np.random.default_rng(seed)
    k = active_bits or max(1, pattern_size // 3)
    seen, pool, attempts = set(), [], 0
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
        def worst(v):
            return max(np.minimum(v, c).sum() / (np.maximum(v, c).sum() + 1e-9) for c in chosen)
        pool.sort(key=worst)
        chosen.append(pool.pop(0))
    return chosen


def test_pattern_generator_matches_original():
    for (P, n, k, seed) in [(10, 8, 3, 1), (12, 7, 4, 2)]:
        a = _original_well_separated(P, n, k, seed, pool_size=6000)
        b = well_separated_patterns(P, n, k, seed, pool_size=6000)
        assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ---------------- retrieval, recurrence flag, input gain ----------------
def test_retrieve_ranks_containing_slot_first_and_respects_mask():
    art = SlottedARTMemory(20, 5, 8)
    T = [np.array([1, 1, 1, 1, 0, 0, 0, 0.]), np.array([0, 0, 0, 0, 1, 1, 1, 1.]),
         np.array([1, 0, 0, 0, 1, 0, 0, 1.])]
    for i, t in enumerate(T):
        art.slots.append({"neurons": np.arange(5), "template": t, "alloc_quality": 0, "alloc_redundancy": 0})
        art._tmpl = np.vstack([art._tmpl, t[None, :]])
        art._nidx = np.vstack([art._nidx, np.arange(5)[None, :]])
    frag = np.array([0, 1, 1, 0, 0, 0, 0, 0.])                 # bits 1,2 -> only slot 0 contains both
    assert art.retrieve(frag, top_k=1)[0][0] == 0
    frag2 = np.array([1, 0, 0, 0, 0, 0, 0, 0.])                # bit 0 is in slots 0 and 2: tie on containment
    top = art.retrieve(frag2, top_k=2)
    assert {top[0][0], top[1][0]} == {0, 2} and top[0][1] == top[1][1] == 1.0
    known = np.array([1, 0, 0, 0, 0, 0, 0, 1], bool)           # only positions 0 and 7 observed
    assert art.retrieve(np.array([1, 0, 0, 0, 0, 0, 0, 1.]), known=known, top_k=1)[0][0] == 2
    assert raises(ValueError, art.retrieve, np.ones(7))


def test_retrieve_spiking_is_read_only_and_finds_trained_slot():
    sysm, pats = trained_system()
    d0 = state_digest(sysm)
    frag = pats[1].copy()
    on = np.where(frag == 1)[0]
    frag[on[2:]] = 0.0                                        # keep 2 of 4 active bits
    out = retrieve_spiking(sysm, frag, top_k=2, steps=80, warmup=80)
    assert state_digest(sysm) == d0
    assert len(out) == 2 and out[0][1] >= out[1][1]


def test_memory_recurrence_flag_removes_projection_and_keeps_rng_stream():
    a = small_system(memory_recurrence=True)
    b = small_system(memory_recurrence=False)
    assert len(a["net"].projections) == len(b["net"].projections) + 1
    assert np.array_equal(a["proj_in_mem"].weights, b["proj_in_mem"].weights)   # same init draws


def test_input_gain_scales_init_weights_and_boost():
    a = small_system(input_gain=1.0)
    b = small_system(input_gain=0.5)
    assert np.allclose(b["proj_in_mem"].weights, 0.5 * a["proj_in_mem"].weights)
    assert b["art"].boost == 0.5 * a["art"].boost


def test_learned_keys_are_normalized_to_unit_range():
    sysm = small_system(use_learned_keys=True)
    art, W = sysm["art"], sysm["proj_in_mem"].weights
    for p in small_patterns(3):
        art.allocate(p, W)
    K = art._keys()
    assert K.min() >= -1e-9 and K.max() <= 1.0 + 1e-9
    # same key object feeds matching and inhibition
    p = small_patterns(3)[0]
    trace = np.full(48, 1.0)
    _, inh = art.inhibition_for(p, trace, 1.0)
    _, attn = art.route(p)
    assert np.allclose(inh, (attn * 1.0) @ K)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
