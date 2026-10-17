"""
probe_contamination_demo.py
===========================
Side-by-side comparison confirming zero side-effects, repeatability, and order 
independence in rest-mode probing vs legacy state-advancing probes.
"""

import argparse, copy, warnings
import numpy as np
from experiment_core import CFG, make_task
from snn_lib_changing import build_art_network, run_schedule, probe_only, state_digest

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
warnings.simplefilter("ignore", RuntimeWarning)

MODES = {"legacy": dict(state_mode="legacy", steps=CFG["legacy_probe_steps"], warmup=0),
         "rest": dict(state_mode="rest", steps=CFG["probe_steps"], warmup=CFG["probe_warmup"])}
rep_range, order_changed, order_absdiff, state_changed = {m: [] for m in MODES}, {m: [] for m in MODES}, \
    {m: [] for m in MODES}, {m: [] for m in MODES}
for seed in range(args.seeds):
    task = make_task(seed)
    base = build_art_network(CFG["pattern_size"], CFG["substrate_size"], CFG["slot_size"], CFG["vigilance"],
                             True, True, True, seed=seed, boost=CFG["boost"])
    run_schedule(base, task["schedule"], steps_per_pattern=CFG["steps_per_pattern"],
                 probe_steps=CFG["probe_steps"], probe_warmup=CFG["probe_warmup"])
    probes = task["train"] + task["novel"]
    for mode, kw in MODES.items():
        # 1. repeatability: same trained pattern x5, and a novel pattern x5
        for p in (task["train"][0], task["novel"][0]):
            s = copy.deepcopy(base)
            vals = [probe_only(s, p, **kw) for _ in range(5)]
            rep_range[mode].append(max(vals) - min(vals))
        # 2. order independence over all 24 probes: forward vs reversed order
        s1, s2 = copy.deepcopy(base), copy.deepcopy(base)
        fwd = np.array([probe_only(s1, p, **kw) for p in probes])
        rev = np.array([probe_only(s2, p, **kw) for p in reversed(probes)])[::-1]
        order_changed[mode].append(np.mean(fwd != rev))
        order_absdiff[mode].append(np.mean(np.abs(fwd - rev)))
        # 3. side effects
        s3 = copy.deepcopy(base)
        d0 = state_digest(s3)
        probe_only(s3, task["train"][1], **kw)
        state_changed[mode].append(state_digest(s3) != d0)

print(f"{len(MODES) and args.seeds} seeds, one trained network each (rest-mode training), 16 trained + 8 novel probes")
print(f"{'':28s}{'legacy':>12s}{'rest (fixed)':>14s}")
print(f"{'repeat-probe range (5x same)':28s}{np.mean(rep_range['legacy']):12.4f}{np.mean(rep_range['rest']):14.4f}")
print(f"{'frac probes changed by order':28s}{np.mean(order_changed['legacy']):12.2f}{np.mean(order_changed['rest']):14.2f}")
print(f"{'mean |fwd - reversed order|':28s}{np.mean(order_absdiff['legacy']):12.4f}{np.mean(order_absdiff['rest']):14.4f}")
print(f"{'probe left state changed':28s}{np.mean(state_changed['legacy']):12.2f}{np.mean(state_changed['rest']):14.2f}")
