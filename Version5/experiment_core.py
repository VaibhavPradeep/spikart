"""
experiment_core.py
==================
Core experiment harness, synthetic task generation, and baseline execution pipeline
shared across ablation and benchmark evaluations.
"""

import warnings
import numpy as np
from snn_lib_changing import (build_art_network, run_schedule, probe_only,
                              well_separated_patterns, force_overlap)
from stats_utils import auroc

CFG = dict(pattern_size=16, active_bits=5, n_patterns=16, n_novel=8,
           substrate_size=144, slot_size=12, vigilance=0.6, boost=4.0,
           steps_per_pattern=500, probe_steps=200, probe_warmup=400,
           legacy_probe_steps=120, dt=0.1,
           overlap_frac=0.5, overlap_targets=(3, 7, 11), order="fixed")

NOISE_TYPES = ("swap1", "swap2", "drop2")


def _noisy(p, kind, rng):
    q = p.copy()
    on, off = np.where(q == 1)[0], np.where(q == 0)[0]
    k = int(kind[-1])
    if kind.startswith("swap"):
        k = min(k, len(on), len(off))
        q[rng.choice(on, k, replace=False)] = 0.0
        q[rng.choice(off, k, replace=False)] = 1.0
    else:                                           # drop (occlusion)
        k = min(k, max(len(on) - 1, 0))
        q[rng.choice(on, k, replace=False)] = 0.0
    return q


def make_task(seed, cfg=CFG):
    P, N = cfg["pattern_size"], cfg["n_patterns"]
    pats = well_separated_patterns(P, N + cfg["n_novel"], active_bits=cfg["active_bits"],
                                   seed=1000 + seed)
    train, novel = pats[:N], pats[N:]
    if cfg["overlap_frac"]:
        targets = [i for i in cfg["overlap_targets"] if i < N]
        train = force_overlap(train, 0, targets, frac=cfg["overlap_frac"])
    rng = np.random.default_rng(5000 + seed)
    noisy = {k: [_noisy(p, k, rng) for p in train] for k in NOISE_TYPES}
    idx = np.arange(N)
    if cfg["order"] == "reverse":
        idx = idx[::-1]
    elif cfg["order"] == "random":
        idx = np.random.default_rng(9000 + seed).permutation(N)
    schedule = []
    for i in idx:
        schedule.append((f"p{i} 1st", train[i]))
        schedule.append((f"p{i} retest", train[i]))
    return {"train": train, "novel": novel, "noisy": noisy, "order": idx.tolist(),
            "schedule": schedule}


def run_snn(seed, use_stdp, use_dpp, use_attn, cfg=CFG, state_mode="rest",
            use_reuse_damping=None, use_overlap_penalty=None, with_noise=False,
            record_traces=False):
    """One full training + probing run. Returns a JSON-friendly dict."""
    task = make_task(seed, cfg)
    N = cfg["n_patterns"]
    legacy = state_mode == "legacy"
    p_steps = cfg["legacy_probe_steps"] if legacy else cfg["probe_steps"]
    system = build_art_network(
        pattern_size=cfg["pattern_size"], substrate_size=cfg["substrate_size"],
        slot_size=cfg["slot_size"], vigilance=cfg["vigilance"], use_dpp=use_dpp,
        use_attention=use_attn, use_stdp=use_stdp, seed=seed, boost=cfg["boost"],
        use_reuse_damping=use_reuse_damping, use_overlap_penalty=use_overlap_penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # reuse fallback is expected here
        ex = run_schedule(system, task["schedule"], steps_per_pattern=cfg["steps_per_pattern"],
                          dt=cfg["dt"], probe_steps=p_steps, probe_warmup=cfg["probe_warmup"],
                          state_mode=state_mode, record_traces=record_traces)
    first, retest = np.zeros(N), np.zeros(N)
    for e in ex:
        i = int(e["label"].split()[0][1:])
        (first if "1st" in e["label"] else retest)[i] = e["probe_mismatch"]

    def probe(p):
        return probe_only(system, p, steps=p_steps, dt=cfg["dt"], warmup=cfg["probe_warmup"],
                          state_mode=state_mode)

    after = np.array([probe(p) for p in task["train"]])
    novel = np.array([probe(p) for p in task["novel"]])
    diag = system["art"].allocation_diagnostics()
    out = {
        "first": float(first.mean()), "retest": float(retest.mean()),
        "after": float(after.mean()), "novel": float(novel.mean()),
        "auroc": auroc(novel, after),
        "interference": float((after - retest).mean()),
        "abs_interference": float(np.abs(after - retest).mean()),
        "n_slots": len(system["art"].slots),
        "alloc_quality": diag["mean_quality"], "alloc_redundancy": diag["mean_redundancy"],
        "cross_slot_cosine": diag["cross_slot_cosine"],
        "n_shared_neurons": diag["n_shared_neurons"],
        "n_fallback": diag["n_fallback_allocations"],
        "per_pattern": {"first": first.tolist(), "retest": retest.tolist(),
                        "after": after.tolist(), "novel": novel.tolist()},
        "exposure_probe": [e["probe_mismatch"] for e in ex],
        "exposure_main": [e["mean_mismatch"] for e in ex],
        "exposure_alloc": [bool(e["allocated"]) for e in ex],
    }
    if with_noise:
        for k in NOISE_TYPES:
            sc = np.array([probe(p) for p in task["noisy"][k]])
            out[f"auroc_{k}"] = auroc(novel, sc)
            out[f"known_{k}"] = float(sc.mean())
    if record_traces:
        out["_exposures"] = ex
    return out


# ---------------------------------------------------------------------------
# spike-free baselines (same tasks, same training order, same test probes)
# ---------------------------------------------------------------------------
def _jac(p, T):
    inter = np.minimum(p[None, :], T).sum(1)
    return inter / (np.maximum(p[None, :], T).sum(1) + 1e-9)


def _scores_baseline(templates, probes, score_fn):
    T = np.array(templates)
    return np.array([score_fn(p, T) for p in probes])


def run_template_baseline(seed, cfg=CFG):
    """Symbolic template matcher: stores a new template whenever the best
    Jaccard match is below vigilance (exactly the architecture's own routing
    rule, with no spiking dynamics); novelty score = 1 - best Jaccard."""
    task = make_task(seed, cfg)
    T = []
    for _, p in task["schedule"]:
        if not T or _jac(p, np.array(T)).max() < cfg["vigilance"]:
            T.append(p.copy())
    return _baseline_eval(task, T, lambda p, T: 1.0 - _jac(p, T).max(), len(T))


def run_art1_baseline(seed, rho, cfg=CFG, alpha=0.001):
    """Classic ART-1 (fast learning): choice function |I^w|/(alpha+|w|),
    vigilance |I^w|/|I| >= rho, resonance sets w <- I^w; novelty score =
    1 - best vigilance match."""
    task = make_task(seed, cfg)
    W = []
    for _, I in task["schedule"]:
        placed = False
        if W:
            Wm = np.array(W)
            inter = np.minimum(I[None, :], Wm).sum(1)
            for j in np.argsort(-inter / (alpha + Wm.sum(1)), kind="stable"):
                if inter[j] / max(I.sum(), 1e-9) >= rho:
                    W[j] = np.minimum(W[j], I)
                    placed = True
                    break
        if not placed:
            W.append(I.copy())

    def score(p, T):
        return 1.0 - (np.minimum(p[None, :], T).sum(1) / max(p.sum(), 1e-9)).max()

    return _baseline_eval(task, W, score, len(W))


def _baseline_eval(task, templates, score_fn, n_cat):
    nov = _scores_baseline(templates, task["novel"], score_fn)
    known = _scores_baseline(templates, task["train"], score_fn)
    out = {"auroc": auroc(nov, known), "n_slots": n_cat, "novel": float(nov.mean()),
           "after": float(known.mean())}
    for k in NOISE_TYPES:
        s = _scores_baseline(templates, task["noisy"][k], score_fn)
        out[f"auroc_{k}"] = auroc(nov, s)
        out[f"known_{k}"] = float(s.mean())
    return out
