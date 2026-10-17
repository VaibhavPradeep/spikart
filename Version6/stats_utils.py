"""
stats_utils.py
==============
Statistical analysis toolkit for computing paired bootstrap CIs, Cohen's d_z, 
sign-flip permutation p-values, and AUROC metrics.
"""

import itertools
import numpy as np

PRIMARY_METRIC = "retest"


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def signflip_p(d, n_mc=20000, seed=0):
    """Two-sided paired sign-flip permutation test of mean(d) == 0
    (exact when n <= 14, Monte Carlo otherwise)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if n <= 14:
        signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
        stat = np.abs((signs * d).mean(axis=1))
        return float((stat >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_mc, n))
    stat = np.abs((signs * d).mean(axis=1))
    return float((np.sum(stat >= obs - 1e-12) + 1) / (n_mc + 1))


def cohens_dz(d):
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return float(d.mean() / sd) if sd and sd > 1e-12 else (0.0 if np.allclose(d, 0) else float("inf"))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(d, seed=0):
    """Summary of a vector of per-seed (paired) differences/effects."""
    d = np.asarray(d, dtype=float)
    lo, hi = bootstrap_ci(d, seed=seed)
    return {"mean": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "dz": cohens_dz(d),
            "p": signflip_p(d, seed=seed), "n": int(len(d))}


def mean_ci(x, seed=0):
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(x, seed=seed)
    return float(x.mean()), lo, hi


def paired_compare(a, b, seed=0):
    """a - b per matched seed."""
    return summarize(np.asarray(a, float) - np.asarray(b, float), seed=seed)


FACTORS = ("STDP", "DPP", "Attention")


def factorial_effects(Y, names=FACTORS):
    """Y: array (n_seeds, 2, 2, 2) indexed [seed, f0, f1, f2], index 1 = ON.
    Returns {effect_name: per-seed vector}. Main effect of A = mean(A on) -
    mean(A off) over the other factors; the interaction effects are the
    standard (1/4) * sum(prod of +-1 signs * y) contrasts, so every effect is
    on the same scale as the metric."""
    Y = np.asarray(Y, dtype=float)
    s = np.array([-1.0, 1.0])
    out = {}
    for r in (1, 2, 3):
        for S in itertools.combinations(range(3), r):
            coef = np.ones((2, 2, 2))
            for f in S:
                shape = [1, 1, 1]
                shape[f] = 2
                coef = coef * s.reshape(shape)
            out["x".join(names[f] for f in S)] = (Y * coef[None]).sum(axis=(1, 2, 3)) / 4.0
    return out


def effects_table(Y, names=FACTORS, seed=0):
    """Rows for all 7 effects with CI, d_z, raw and Holm-adjusted p."""
    eff = factorial_effects(Y, names)
    rows = []
    for k, v in eff.items():
        r = summarize(v, seed=seed)
        r["effect"] = k
        rows.append(r)
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    return rows


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie): 1.0 = every novel item scores
    above every known item."""
    pos = np.asarray(pos, float)[:, None]
    neg = np.asarray(neg, float)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())
