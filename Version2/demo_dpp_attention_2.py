"""
demo_dpp_attention_2.py
=======================
Demonstration script evaluating pattern memory allocation under DPP slot 
selection and attention routing.
"""

import time
import numpy as np
from snn_lib_2 import build_novelty_network, present_pattern

PATTERN_SIZE = 48
N_COLUMNS, EXC_PER_COL, INH_PER_COL = 16, 12, 3
N_PATTERNS = 10
STEPS = 900
SEED = 7

NETWORK_KW = dict(
    inhibition_strength=90.0, in_ev_strength=70.0,
    mem_init_range=(0.5, 1.5), w_max=60.0, tau_syn=5.0,
)
PRESENT_KW = dict(steps=STEPS, k_alloc=8, primer_strength=25.0, vigilance=0.015)


def make_patterns(n, size, seed):
    rng = np.random.RandomState(seed)
    pats = [(rng.rand(size) < 0.35).astype(float) for _ in range(n)]
    active0 = np.where(pats[0] == 1)[0]
    shared = active0[:len(active0) // 2]
    for idx in [3, 6, 8]:          # force overlap with pattern 0
        pats[idx][shared] = 1.0
    return pats


if __name__ == "__main__":
    np.random.seed(SEED)
    t_build0 = time.time()
    bundle = build_novelty_network(
        pattern_size=PATTERN_SIZE, n_columns=N_COLUMNS,
        exc_per_col=EXC_PER_COL, inh_per_col=INH_PER_COL,
        use_dpp=True, use_attention=True, **NETWORK_KW)
    t_build = time.time() - t_build0

    n_mem = bundle["mem"].n
    n_mem_exc = int(bundle["mem_exc_mask"].sum())
    print(f"network: {PATTERN_SIZE}-bit input | memory pool {n_mem} neurons "
          f"({n_mem_exc} excitatory across {N_COLUMNS} columns) | "
          f"evaluator {bundle['ev'].n} neurons")
    print(f"build time: {t_build*1000:.1f} ms")
    print()

    patterns = make_patterns(N_PATTERNS, PATTERN_SIZE, seed=3)
    overlaps = [int((patterns[0] * patterns[i]).sum()) for i in range(N_PATTERNS)]
    print("active bits per pattern:", [int(p.sum()) for p in patterns])
    print("overlap with pattern 0: ", overlaps)
    print()

    t_run0 = time.time()
    first, retest = [], []
    for p in patterns:
        f, _ = present_pattern(bundle, p, **PRESENT_KW)
        r, _ = present_pattern(bundle, p, **PRESENT_KW)
        first.append(f)
        retest.append(r)

    # retest everything again after all 10 patterns have been learned,
    # to see how much interference accumulated over the whole run
    after_all = [present_pattern(bundle, p, **PRESENT_KW)[0] for p in patterns]
    t_run = time.time() - t_run0

    print(f"{'pattern':8s} {'active':>7s} {'overlap p0':>11s} {'1st':>8s} "
          f"{'retest':>8s} {'after-all':>10s}")
    for i in range(N_PATTERNS):
        print(f"p{i:<7d} {int(patterns[i].sum()):7d} {overlaps[i]:11d} "
              f"{first[i]:8.4f} {retest[i]:8.4f} {after_all[i]:10.4f}")

    mean_reduction = np.mean([f - r for f, r in zip(first, retest)])
    mean_interference = np.mean([a - r for a, r in zip(after_all, retest)])
    print()
    print(f"mean 1st -> retest reduction : {mean_reduction:.4f}")
    print(f"mean interference (after all patterns learned): {mean_interference:.4f}")
    print(f"total simulation time for {N_PATTERNS} patterns x 3 presentations "
          f"({N_PATTERNS*3*STEPS} network ticks over {n_mem+bundle['ev'].n+PATTERN_SIZE} "
          f"neurons): {t_run:.2f} s")
