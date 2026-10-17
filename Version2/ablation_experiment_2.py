"""
ablation_experiment_2.py
========================
Initial factorial ablation study measuring the individual contributions 
of DPP slot selection and top-down inhibition.
"""

import numpy as np
from snn_lib_2 import build_novelty_network, present_pattern

PATTERN_SIZE = 24
N_COLUMNS, EXC_PER_COL, INH_PER_COL = 6, 10, 2
N_PATTERNS = 6
STEPS = 900
K_ALLOC = 6
SEED = 42

COMMON_KW = dict(
    inhibition_strength=90.0, in_ev_strength=70.0,
    mem_init_range=(0.5, 1.5), w_max=60.0, tau_syn=5.0,
)
PRESENT_KW = dict(steps=STEPS, k_alloc=K_ALLOC, primer_strength=25.0, vigilance=0.015)


def make_patterns(n, size, seed):
    rng = np.random.RandomState(seed)
    pats = []
    for i in range(n):
        p = (rng.rand(size) < 0.35).astype(float)
        pats.append(p)
    # force some deliberate overlap so interference is actually testable:
    # patterns 2 and 4 share half of pattern 0's active bits
    active0 = np.where(pats[0] == 1)[0]
    if len(active0) >= 2 and n > 4:
        shared = active0[:len(active0) // 2]
        for idx in [2, 4]:
            pats[idx][shared] = 1.0
    return pats


def run_condition(name, use_dpp, use_attention, patterns):
    np.random.seed(SEED)
    bundle = build_novelty_network(
        pattern_size=PATTERN_SIZE, n_columns=N_COLUMNS,
        exc_per_col=EXC_PER_COL, inh_per_col=INH_PER_COL,
        use_dpp=use_dpp, use_attention=use_attention, **COMMON_KW)

    first, retest = [], []
    for p in patterns:
        f, _ = present_pattern(bundle, p, **PRESENT_KW)   # probe = genuine pre-learning novelty
        r, _ = present_pattern(bundle, p, **PRESENT_KW)   # probe = true retest, before this call's own update
        first.append(f)
        retest.append(r)

    # after ALL patterns have been through first+retest, go back and
    # re-test each one again to measure interference from everything
    # learned since
    after_all = [present_pattern(bundle, p, **PRESENT_KW)[0] for p in patterns]

    interference = [a - r for a, r in zip(after_all, retest)]
    return dict(name=name, first=first, retest=retest, after_all=after_all,
                interference=interference)


if __name__ == "__main__":
    patterns = make_patterns(N_PATTERNS, PATTERN_SIZE, seed=1)
    overlaps = [int((patterns[0] * patterns[i]).sum()) for i in range(N_PATTERNS)]
    print(f"pattern size={PATTERN_SIZE}, memory pool={N_COLUMNS * EXC_PER_COL} exc neurons "
          f"across {N_COLUMNS} columns, {N_PATTERNS} patterns")
    print("active bits per pattern:", [int(p.sum()) for p in patterns])
    print("overlap with pattern 0:", overlaps)
    print()

    conditions = [
        ("baseline (no DPP, no attention)", False, False),
        ("dpp_only",                         True,  False),
        ("attention_only",                   False, True),
        ("full (DPP + attention)",           True,  True),
    ]

    results = []
    for name, use_dpp, use_attention in conditions:
        r = run_condition(name, use_dpp, use_attention, patterns)
        results.append(r)
        print(f"--- {name} ---")
        print(f"  {'pattern':8s} {'1st':>8s} {'retest':>8s} {'after-all':>10s} {'interf.':>8s}")
        for i in range(N_PATTERNS):
            print(f"  p{i:<7d} {r['first'][i]:8.4f} {r['retest'][i]:8.4f} "
                  f"{r['after_all'][i]:10.4f} {r['interference'][i]:8.4f}")
        mean_interf = np.mean(np.abs(r['interference']))
        mean_reduction = np.mean([f - re for f, re in zip(r['first'], r['retest'])])
        print(f"  mean |interference| = {mean_interf:.4f}   "
              f"mean 1st->retest reduction = {mean_reduction:.4f}")
        print()

    print("=" * 65)
    print("SUMMARY: mean |interference| across all 6 patterns, by condition")
    print("=" * 65)
    for r in results:
        print(f"  {r['name']:32s}  {np.mean(np.abs(r['interference'])):.4f}")
