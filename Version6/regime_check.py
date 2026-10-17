"""
regime_check.py
===============
Substrate firing rate and recurrent current diagnostic verifying network stability 
and E/I balance under scaled pattern loads.
"""

import argparse, csv, os, warnings
import numpy as np
from snn_lib_changing import (build_art_network, run_schedule, probe_only, well_separated_patterns)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)

DT, WARM, STEPS = 0.1, 400, 300
CONFIGS = [  # (n_exc, pattern_size, active_bits)
    (12, 16, 5), (48, 16, 5), (144, 16, 5), (480, 16, 5), (480, 24, 6)]


def capture(system, pattern):
    mem, net = system["memory"], system["net"]
    nE = system["substrate_size"]
    rec = next(p for p in net.projections if p.name == "Memory-recurrent")
    inp = next(p for p in net.projections if p.name == "Input->Memory")
    sp = np.zeros((STEPS, mem.n))
    c_in, c_ee, c_ie = (np.zeros((STEPS, nE)) for _ in range(3))
    ev = system["evaluator"]
    mism = np.zeros(STEPS)

    def hook(t):
        sp[t] = mem.spiked
        c_in[t] = inp.weights[:nE] @ inp.psc
        c_ee[t] = rec.weights[:nE, :nE] @ rec.psc[:nE]
        c_ie[t] = rec.weights[:nE, nE:] @ rec.psc[nE:]
        mism[t] = ev.spiked[:system["pattern_size"]].sum()
    probe_only(system, pattern, steps=STEPS, warmup=WARM, hook=hook)
    return sp, c_in, c_ee, c_ie, mism


def summarize(system, pattern, slot_idx, label):
    nE = system["substrate_size"]
    sp, c_in, c_ee, c_ie, mism = capture(system, pattern)
    rate = sp.sum(0) / (STEPS * DT)                       # spikes/neuron/ms
    E, I = rate[:nE], rate[nE:]
    in_slot = np.zeros(nE, bool)
    if slot_idx is not None:
        in_slot[slot_idx] = True
    out = {"state": label, "E_rate": E.mean(), "I_rate": I.mean(),
           "E_slot_rate": E[in_slot].mean() if in_slot.any() else np.nan,
           "E_other_rate": E[~in_slot].mean(),
           "frac_E_active": (sp[:, :nE].sum(0) > 0).mean(),
           "I_ext": c_in[:, ~in_slot].mean(), "I_ee": c_ee[:, ~in_slot].mean(), "I_ie": c_ie[:, ~in_slot].mean(),
           "I_ext_slot": c_in[:, in_slot].mean() if in_slot.any() else np.nan,
           "I_ee_slot": c_ee[:, in_slot].mean() if in_slot.any() else np.nan,
           "I_ie_slot": c_ie[:, in_slot].mean() if in_slot.any() else np.nan,
           "evaluator_mismatch_per_ms": mism.sum() / (STEPS * DT)}
    return out


rows = []
for n_exc, P, bits in CONFIGS:
    pats = well_separated_patterns(P, 4, active_bits=bits, seed=11)
    sysm = build_art_network(P, n_exc, 12, 0.6, True, True, True, seed=0, boost=4.0)
    tag = f"nE={n_exc},P={P}"
    rows.append({"config": tag, **summarize(sysm, np.zeros(P), None, "no input")})
    rows.append({"config": tag, **summarize(sysm, pats[0], None, "pattern, no slots (untrained)")})
    run_schedule(sysm, [(f"p{i}", pats[i]) for i in range(3)], steps_per_pattern=300,
                 probe_steps=50, probe_warmup=50)
    rows.append({"config": tag, **summarize(sysm, pats[0], sysm["art"].slots[0]["neurons"], "trained pattern (3 slots)")})
    rows.append({"config": tag, **summarize(sysm, pats[3], None, "novel pattern (3 slots)")})
    print(f"done {tag}", flush=True)

print("\n" + "=" * 118)
print("Memory-substrate regime (rates in spikes/neuron/ms; currents in model units; threshold current = 20)")
print("=" * 118)
print(f"{'config':14s} {'state':30s} {'E rate':>7s} {'slotE':>7s} {'othrE':>7s} {'I rate':>7s} {'%E act':>6s} "
      f"{'I_ext':>7s} {'I_ee':>7s} {'I_ie':>7s} {'flag':>14s}")
for r in rows:
    flag = []
    if r["state"] != "no input":
        if r["I_rate"] < 0.002 and r["state"].startswith("trained"):
            flag.append("I-pool silent")
        top = r["E_slot_rate"] if not np.isnan(r["E_slot_rate"]) else r["E_rate"]
        if top > 0.4:
            flag.append("SATURATED")
        if r["state"].startswith("trained") and top < 0.002:
            flag.append("DEAD slot")
    print(f"{r['config']:14s} {r['state']:30s} {r['E_rate']:7.4f} {np.nan_to_num(r['E_slot_rate']):7.4f} "
          f"{r['E_other_rate']:7.4f} {r['I_rate']:7.4f} {100 * r['frac_E_active']:6.1f} "
          f"{r['I_ext']:7.2f} {r['I_ee']:7.2f} {r['I_ie']:7.2f} {','.join(flag) or 'ok':>14s}")
print("\nFor slot neurons of the TRAINED pattern (the neurons that carry the memory):")
print(f"{'config':14s} {'I_ext_slot':>11s} {'I_ee_slot':>10s} {'I_ie_slot':>10s} {'|rec net|/ext':>14s} "
      f"{'evaluator mism/ms (trained)':>28s}")
for r in rows:
    if r["state"].startswith("trained"):
        net_rec = r["I_ee_slot"] + r["I_ie_slot"]
        print(f"{r['config']:14s} {r['I_ext_slot']:11.2f} {r['I_ee_slot']:10.3f} {r['I_ie_slot']:10.3f} "
              f"{abs(net_rec) / max(r['I_ext_slot'], 1e-9):14.4f} {r['evaluator_mismatch_per_ms']:28.4f}")
print("\nEvaluator mismatch per ms: untrained pattern (un-inhibited) vs trained vs novel")
for tag in dict.fromkeys(r["config"] for r in rows):
    g = {r["state"]: r["evaluator_mismatch_per_ms"] for r in rows if r["config"] == tag}
    print(f"  {tag:14s} untrained={g['pattern, no slots (untrained)']:.4f}  trained={g['trained pattern (3 slots)']:.4f}  "
          f"novel={g['novel pattern (3 slots)']:.4f}")

with open(os.path.join(args.out, "regime_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved -> {args.out}/regime_check.csv")
