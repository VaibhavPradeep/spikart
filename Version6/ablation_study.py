"""
ablation_study.py
=================
Factorial ablation study and DPP component decomposition (greedy MAP vs reuse damping
vs overlap penalty) under fixed state-restoration probing.
"""

import argparse
import itertools
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_core import CFG, run_snn
from stats_utils import (PRIMARY_METRIC, FACTORS, mean_ci, effects_table, paired_compare,
                         summarize)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", default="results")
ap.add_argument("--skip-legacy", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
RAW = os.path.join(args.out, "ablation_raw.json")
SEEDS = list(range(args.seeds))
CELLS = list(itertools.product([False, True], repeat=3))      # (stdp, dpp, attn), index 1 = ON
DPP_VARIANTS = {   # (use_dpp, reuse_damping, overlap_penalty) at STDP on, attention on
    "random": (False, None, None),
    "greedy only": (True, False, False),
    "greedy + reuse": (True, True, False),
    "greedy + overlap": (True, False, True),
    "greedy + both (bundle)": (True, True, True),
}

cache = json.load(open(RAW)) if os.path.exists(RAW) else {}


def get(key, **kw):
    if key not in cache:
        cache[key] = run_snn(**kw)
        json.dump(cache, open(RAW, "w"))
    return cache[key]


def key_fact(mode, s, d, a, seed):
    return f"{mode}|S{int(s)}D{int(d)}A{int(a)}|{seed}"


t0 = time.time()
total = len(CELLS) * len(SEEDS) * (1 if args.skip_legacy else 2) + 3 * len(SEEDS)
done = 0
for mode in (["rest"] if args.skip_legacy else ["rest", "legacy"]):
    for seed in SEEDS:
        for (s, d, a) in CELLS:
            get(key_fact(mode, s, d, a, seed), seed=seed, use_stdp=s, use_dpp=d, use_attn=a,
                state_mode=mode)
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{total}] {time.time() - t0:.0f}s", flush=True)
for name, (d, rd, op) in DPP_VARIANTS.items():
    for seed in SEEDS:
        k = f"dpp|{name}|{seed}"
        get(k, seed=seed, use_stdp=True, use_dpp=d, use_attn=True, use_reuse_damping=rd,
            use_overlap_penalty=op)
print(f"all runs done in {time.time() - t0:.0f}s", flush=True)


def metric_array(mode, metric):
    """(n_seeds, 2, 2, 2)"""
    Y = np.zeros((len(SEEDS), 2, 2, 2))
    for i, seed in enumerate(SEEDS):
        for (s, d, a) in CELLS:
            Y[i, int(s), int(d), int(a)] = cache[key_fact(mode, s, d, a, seed)][metric]
    return Y


def fmt(m, lo, hi, nd=4):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


W = 100
print("\n" + "=" * W)
print(f"1. FACTORIAL ABLATION  ({len(SEEDS)} matched seeds; pattern set regenerated per seed; "
      f"state_mode=rest)")
print(f"   {CFG['pattern_size']}-bit patterns, {CFG['active_bits']} active, {CFG['n_patterns']} patterns, "
      f"{CFG['substrate_size'] // CFG['slot_size']} native slots (reuse forced beyond that), "
      f"overlap on patterns 3/7/11")
print(f"   probe = {CFG['probe_warmup']} discarded + {CFG['probe_steps']} measured ticks from rest, "
      f"learning off. PRIMARY METRIC = {PRIMARY_METRIC} (declared before running).")
print("   cells: mean [95% bootstrap CI over seeds]; lower mismatch = better; AUROC higher = better")
print("=" * W)
hdr = f"{'STDP':>5s} {'DPP':>5s} {'ATTN':>5s} | {'1st':>22s} {'retest (PRIMARY)':>22s} {'novel':>22s} | {'AUROC':>6s}"
print(hdr)
print("-" * len(hdr))
for (s, d, a) in CELLS[::-1]:
    row = []
    for m in ("first", "retest", "novel"):
        v = [cache[key_fact("rest", s, d, a, sd)][m] for sd in SEEDS]
        row.append(fmt(*mean_ci(v), nd=4))
    au = np.mean([cache[key_fact("rest", s, d, a, sd)]["auroc"] for sd in SEEDS])
    print(f"{str(s):>5s} {str(d):>5s} {str(a):>5s} | {row[0]:>22s} {row[1]:>22s} {row[2]:>22s} | {au:6.3f}")
print("(AUROC = P(novel probe mismatch > trained-pattern mismatch), 8 novel vs 16 trained probes after all training.)")


def print_effects(metric, title, lower_is_better=True):
    rows = effects_table(metric_array("rest", metric))
    print(f"\n{title}  [effect = ON minus OFF (main) or contrast (interaction), per-seed, then summarized]")
    print(f"  {'effect':26s} {'mean':>9s} {'95% CI':>20s} {'d_z':>7s} {'p':>8s} {'p_Holm':>8s}")
    for r in rows:
        print(f"  {r['effect']:26s} {r['mean']:9.4f} [{r['ci_lo']:8.4f},{r['ci_hi']:8.4f}] "
              f"{r['dz']:7.2f} {r['p']:8.4f} {r['p_holm']:8.4f}")
    return rows


prim_rows = print_effects(PRIMARY_METRIC, f"PRIMARY: effects on {PRIMARY_METRIC} mismatch "
                          "(negative main effect = the mechanism lowers mismatch = helps)")
print("\nSecondary / exploratory (no multiplicity control across metrics):")
sec_rows = {m: print_effects(m, f"  effects on {m}") for m in ("novel", "auroc", "after")}

print("\nRelative main effects on the primary metric (ON vs OFF, averaged over other factors):")
Yp = metric_array("rest", PRIMARY_METRIC)
for name, ax in zip(FACTORS, (1, 2, 3)):
    on = np.take(Yp, 1, axis=ax).mean(axis=(1, 2))
    off = np.take(Yp, 0, axis=ax).mean(axis=(1, 2))
    rel = 100.0 * (off - on) / np.where(off > 1e-9, off, np.nan)
    print(f"  {name:10s}: ON={on.mean():.4f}  OFF={off.mean():.4f}  "
          f"per-seed relative reduction mean={np.nanmean(rel):.1f}%")

# ------------------------------------------------------------------ section 2
print("\n" + "=" * W)
print("2. DPP DECOMPOSITION (todo #13): STDP on, attention on; matched seeds; paired vs the comparison arm")
print("   use_dpp now means ONLY greedy MAP selection; reuse damping and the overlap penalty are separate flags.")
print("   alloc_quality is what greedy selection maximizes -> circular, NOT evidence of better recognition.")
print("=" * W)
metrics2 = [("retest", "retest(PRIMARY)"), ("after", "after"), ("novel", "novel"), ("auroc", "AUROC"),
            ("alloc_quality", "alloc_q*"), ("alloc_redundancy", "alloc_red"),
            ("cross_slot_cosine", "xslot_cos"), ("n_shared_neurons", "shared_n")]
print(f"  {'variant':26s}" + "".join(f"{lab:>17s}" for _, lab in metrics2))
dpp_data = {n: {m: np.array([cache[f'dpp|{n}|{sd}'][m] for sd in SEEDS]) for m, _ in metrics2}
            for n in DPP_VARIANTS}
for n in DPP_VARIANTS:
    print(f"  {n:26s}" + "".join(f"{dpp_data[n][m].mean():17.4f}" for m, _ in metrics2))
print("\n  Paired differences (variant minus 'random'), mean [95% CI], sign-flip p:")
for n in list(DPP_VARIANTS)[1:]:
    parts = []
    for m, lab in metrics2[:4]:
        r = paired_compare(dpp_data[n][m], dpp_data["random"][m])
        parts.append(f"{lab.split('(')[0]}:{r['mean']:+.4f}[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p']:.3f}")
    print(f"   {n:24s} " + "  ".join(parts))
print("\n  Paired differences (bundle minus 'greedy only') -- what the two soft modifiers add on top of greedy MAP:")
for m, lab in metrics2:
    r = paired_compare(dpp_data["greedy + both (bundle)"][m], dpp_data["greedy only"][m])
    print(f"   {lab:18s} {r['mean']:+9.4f} [{r['ci_lo']:+9.4f},{r['ci_hi']:+9.4f}]  d_z={r['dz']:+.2f}  p={r['p']:.3f}")
print("\n  Paired differences (greedy only minus 'random') -- the effect of greedy DPP selection ITSELF:")
for m, lab in metrics2:
    r = paired_compare(dpp_data["greedy only"][m], dpp_data["random"][m])
    print(f"   {lab:18s} {r['mean']:+9.4f} [{r['ci_lo']:+9.4f},{r['ci_hi']:+9.4f}]  d_z={r['dz']:+.2f}  p={r['p']:.3f}")

# ------------------------------------------------------------------ section 3
if not args.skip_legacy:
    print("\n" + "=" * W)
    print("3. EFFECT OF THE PRE-FIX STATE HANDLING (todo #5/#8): state_mode='legacy' vs 'rest', same seeds/patterns")
    print("   legacy = probes continue from whatever state the network is in and leave it advanced;")
    print("   legacy probes are 120 ticks (the original setting), rest probes are 400 warm-up + 200 measured.")
    print("=" * W)
    print(f"  {'metric':14s} {'legacy mean':>12s} {'rest mean':>12s}   (all 8 cells x seeds pooled)")
    for m in ("first", "retest", "after", "novel", "interference", "abs_interference", "auroc"):
        L = metric_array("legacy", m).mean()
        R = metric_array("rest", m).mean()
        print(f"  {m:14s} {L:12.4f} {R:12.4f}")
    neg = {m: 100 * np.mean([cache[key_fact(m, *c, sd)]['interference'] < -1e-9 for c in CELLS for sd in SEEDS])
           for m in ("legacy", "rest")}
    print(f"\n  'interference' (after-all minus retest) is negative in {neg['legacy']:.0f}% of legacy runs and "
          f"{neg['rest']:.0f}% of rest runs. A negative value is not by itself an artifact (later exposures can keep "
          f"improving a slot), so the DIRECT contamination evidence is probe_contamination_demo.py "
          f"(repeat-probe and probe-order tests); here compare magnitudes: mean |interference| legacy vs rest above.")
    print("  CAUTION: legacy vs rest differs in BOTH state handling and probe length (120 ticks vs 400+200), so the "
          "differences below are the size of the old pipeline's numbers vs the new pipeline's, not a pure state-handling effect.")
    print("\n  Main effects on retest, legacy vs rest (mean over seeds):")
    for mode in ("legacy", "rest"):
        rows = effects_table(metric_array(mode, PRIMARY_METRIC))
        print(f"   {mode:7s} " + "  ".join(f"{r['effect'].split('x')[0] if 'x' not in r['effect'] else r['effect']}"
                                           f"={r['mean']:+.4f}(p={r['p']:.3f})" for r in rows[:3]))

# ------------------------------------------------------------------ section 4
print("\n" + "=" * W)
print("4. PER-PATTERN BREAKDOWN, full condition (STDP+DPP+Attention), seed 0, state_mode=rest")
print("   (patterns 3, 7, 11 share bits with pattern 0; 12-15 are forced to reuse neurons)")
print("=" * W)
pp = cache[key_fact("rest", True, True, True, 0)]["per_pattern"]
print(f"  {'pattern':8s} {'1st':>8s} {'retest':>8s} {'after-all':>10s} {'after-retest':>13s}")
for i in range(CFG["n_patterns"]):
    flag = " *overlap" if i in CFG["overlap_targets"] else (" (reused neurons)" if i >= 12 else "")
    print(f"  p{i:<7d} {pp['first'][i]:8.4f} {pp['retest'][i]:8.4f} {pp['after'][i]:10.4f} "
          f"{pp['after'][i] - pp['retest'][i]:13.4f}{flag}")
print(f"  novel probes (never trained): {np.round(pp['novel'], 4).tolist()}")

# ------------------------------------------------------------------ figures
print("\nBuilding figures...")
from snn_lib_changing import DEFAULT_PROBE_WARMUP  # noqa: E402

trace_run = run_snn(seed=0, use_stdp=True, use_dpp=True, use_attn=True, record_traces=True)
ex = trace_run["_exposures"]

SER = [("Full (STDP+DPP+Attn)", (1, 1, 1), "green"), ("No attention", (1, 1, 0), "crimson"),
       ("No STDP", (0, 1, 1), "orange"), ("Nothing (all off)", (0, 0, 0), "gray")]


def per_pattern_series(cell, field, kind):
    """(n_seeds, n_patterns) from per-exposure arrays; kind '1st' or 'retest'."""
    out = []
    for sd in SEEDS:
        r = cache[key_fact("rest", *map(bool, cell), sd)][field]
        out.append(np.array(r[0::2] if kind == "1st" else r[1::2]))
    return np.array(out)


fig, ax = plt.subplots(3, 1, figsize=(13, 13))
x = np.arange(CFG["n_patterns"])
for name, cell, col in SER:
    for kind, ls in (("1st", "--"), ("retest", "-")):
        A = per_pattern_series(cell, "exposure_probe", kind)
        m = A.mean(0)
        lo, hi = np.array([mean_ci(A[:, i])[1:] for i in x]).T
        ax[0].plot(x, m, ls, color=col, lw=1.4, label=name if kind == "1st" else None)
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.12)
        ax[1].plot(x, per_pattern_series(cell, "exposure_main", kind).mean(0), ls, color=col, lw=1.4)
for a_ in ax[:2]:
    a_.axvline(CFG["slot_size"] - 0.5, color="gold", ls=":", lw=1.5)
    a_.grid(True, alpha=0.3)
ax[0].set_title("A. PRE-COMMIT probe mismatch per pattern (genuine novelty signal; read-only, from rest)\n"
                "dashed = 1st exposure, solid = retest; band = 95% bootstrap CI over seeds; "
                "gold = native-slot capacity boundary")
ax[0].set_ylabel("probe mismatch (spikes/tick)")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].set_title("B. POST-COMMIT main-window mismatch per pattern (after fast-commit; learning on, from rest)")
ax[1].set_ylabel("main-window mismatch (spikes/tick)")
ax[1].set_xlabel("pattern index")

BIN = 20
offset, ticks_x = 0, []
for i, e in enumerate(ex[:6]):
    segs = [("warm-up (discarded)", e["warmup_trace"], "lightgray"),
            ("pre-commit probe", e["probe_trace"], "tab:blue"),
            ("post-commit main window", e["main_trace"], "tab:green")]
    for lab, tr, col in segs:
        n = len(tr) // BIN * BIN
        b = tr[:n].reshape(-1, BIN).sum(1)
        xs = offset + np.arange(len(b)) * BIN + BIN / 2
        ax[2].bar(xs * 0.1, b, width=BIN * 0.1, color=col, label=lab if i == 0 else None)
        offset += len(tr)
    ax[2].axvline((offset - len(e["main_trace"])) * 0.1, color="k", lw=0.8)
ax[2].set_title("C. Per-tick evaluator spikes (seed 0, full), first 3 patterns (1st + retest each)\n"
                "warm-up | pre-commit probe | post-commit main window; black line = start of main window (commit)", fontsize=10)
ax[2].set_xlabel("simulated time (ms)")
ax[2].set_ylabel(f"spikes per {BIN} ticks")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(args.out, "ablation_timeseries.png"), dpi=140)
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
labs = [f"S{int(s)}D{int(d)}A{int(a)}" for (s, d, a) in CELLS[::-1]]
ms, los, his = [], [], []
for (s, d, a) in CELLS[::-1]:
    v = [cache[key_fact("rest", s, d, a, sd)][PRIMARY_METRIC] for sd in SEEDS]
    m, lo, hi = mean_ci(v)
    ms.append(m), los.append(m - lo), his.append(hi - m)
ax[0, 0].bar(labs, ms, yerr=[los, his], color=["seagreen" if l.endswith("A1") else "indianred" for l in labs],
             capsize=3)
ax[0, 0].set_title(f"A. {PRIMARY_METRIC} mismatch (PRIMARY), 8 conditions, {len(SEEDS)} matched seeds, 95% CI\n"
                   "green = attention ON, red = OFF; S/D/A = STDP/DPP/Attention")
ax[0, 0].tick_params(axis="x", rotation=45)
ax[0, 0].grid(True, alpha=0.3, axis="y")

ys = np.arange(len(prim_rows))[::-1]
ax[0, 1].errorbar([r["mean"] for r in prim_rows], ys,
                  xerr=[[r["mean"] - r["ci_lo"] for r in prim_rows], [r["ci_hi"] - r["mean"] for r in prim_rows]],
                  fmt="o", color="navy", capsize=3)
ax[0, 1].axvline(0, color="k", lw=0.8)
ax[0, 1].set_yticks(ys)
ax[0, 1].set_yticklabels([f"{r['effect']}  (p_Holm={r['p_holm']:.3f})" for r in prim_rows], fontsize=8)
ax[0, 1].set_title(f"B. Factorial effects on {PRIMARY_METRIC} (95% bootstrap CI over seeds)\nnegative = lowers mismatch")
ax[0, 1].grid(True, alpha=0.3, axis="x")

names = list(DPP_VARIANTS)
xx = np.arange(len(names))
ms = [dpp_data[n]["retest"].mean() for n in names]
es = [[dpp_data[n]["retest"].mean() - mean_ci(dpp_data[n]["retest"])[1] for n in names],
      [mean_ci(dpp_data[n]["retest"])[2] - dpp_data[n]["retest"].mean() for n in names]]
ax[1, 0].bar(xx - 0.2, ms, 0.4, yerr=es, color="steelblue", capsize=3, label="retest (PRIMARY, lower=better)")
ax2 = ax[1, 0].twinx()
ax2.bar(xx + 0.2, [dpp_data[n]["alloc_quality"].mean() for n in names], 0.4, color="salmon",
        label="alloc quality (circular)")
ax[1, 0].set_xticks(xx)
ax[1, 0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax[1, 0].set_title("C. DPP decomposition: downstream retest vs construction-level quality\n"
                   "(quality is what greedy selection maximizes -> it rises by construction)")
ax[1, 0].legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

if not args.skip_legacy:
    labs2 = [f"S{int(s)}D{int(d)}A{int(a)}" for (s, d, a) in CELLS[::-1]]
    for off, mode, col in ((-0.2, "legacy", "tab:red"), (0.2, "rest", "tab:green")):
        v = [np.mean([cache[key_fact(mode, s, d, a, sd)]["interference"] for sd in SEEDS]) for (s, d, a) in CELLS[::-1]]
        ax[1, 1].bar(np.arange(8) + off, v, 0.4, color=col, label=mode)
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_xticks(np.arange(8))
    ax[1, 1].set_xticklabels(labs2, rotation=45)
    ax[1, 1].set_title("D. 'Interference' (after-all minus retest), legacy vs fixed state handling\n"
                       "legacy goes negative = warm-start artifact")
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "ablation_summary.png"), dpi=140)
print(f"Figures saved -> {args.out}/ablation_timeseries.png, {args.out}/ablation_summary.png")
