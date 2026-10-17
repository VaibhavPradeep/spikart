"""
diagnose_identical.py
=====================
Diagnostic script investigating unrouted inhibition dynamics and evaluator 
cancellation thresholds in non-attentional configurations.
"""
import argparse, json, warnings
import numpy as np
from snn_lib_changing import build_art_network, run_schedule, probe_only, well_separated_patterns, force_overlap

ap = argparse.ArgumentParser()
ap.add_argument("--demo-json", default="results/demo_scaled_raw.json")
args = ap.parse_args()
warnings.simplefilter("ignore", RuntimeWarning)
raw = json.load(open(args.demo_json))
a = np.array(raw["No attention (uniform inhibition)|0"]["curve"])
b = np.array(raw["Nothing (no STDP/DPP/attention)|0"]["curve"])
print(f"1. per-pattern curves equal (seed 0): {np.array_equal(a, b)}   (max abs diff {np.abs(a - b).max():.2e})")

P, BITS, N, SUB = 24, 6, 48, 480
pats = force_overlap(well_separated_patterns(P, N, active_bits=BITS, seed=9), 0, [3, 12, 25, 40], 0.5)
fresh = build_art_network(P, SUB, 12, 0.55, True, True, False, seed=3, boost=4.0)
raw_level = np.mean([probe_only(fresh, p, steps=200, warmup=400) for p in pats[:6]])
print(f"3. reading with NO slots (raw drive), first 6 patterns: {raw_level:.4f}   no-attention curve mean over the same: {a[:6].mean():.4f}")

for attn in (False, True):
    s = build_art_network(P, SUB, 12, 0.55, True, attn, False, seed=3, boost=4.0)
    run_schedule(s, [(f"p{i}", p) for i, p in enumerate(pats)], steps_per_pattern=350, probe_steps=50, probe_warmup=50)
    art, mem = s["art"], s["memory"]
    peak = []
    for i in (0, 5, 20, 40):
        got = []
        probe_only(s, pats[i], steps=100, warmup=400,
                   hook=lambda t: got.append(art.inhibition_for(pats[i], mem.trace[mem.exc_idx], 20.0)[1][pats[i] == 1].max()))
        peak.append(max(got))
    print(f"{'4' if attn else '2'}. attention={'ON ' if attn else 'OFF'}: peak inhibition on a trained pattern's active bits "
          f"(4 patterns) = {np.round(peak, 1)}  vs ~35 needed to silence an evaluator channel; slots={len(art.slots)}")
