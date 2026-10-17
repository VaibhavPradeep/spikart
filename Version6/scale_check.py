"""
scale_check.py
==============
Scalability and memory benchmark testing substrates up to 1920 excitatory neurons
and 256-bit input patterns with input-gain calibration.
"""

import argparse, resource, time, warnings
import numpy as np
from snn_lib_changing import (build_art_network, run_schedule, probe_only, well_separated_patterns)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
args = ap.parse_args()
warnings.simplefilter("ignore", RuntimeWarning)
CONFIGS = [(16, 5, 144), (64, 13, 480), (128, 26, 960), (256, 51, 1920)]
rows = []
print(f"{'P':>4s} {'bits':>4s} {'nE':>5s} {'gain':>5s} {'recur':>5s} {'build s':>8s} {'ms/tick':>8s} "
      f"{'RSS MB':>7s} {'%E fire(trained)':>17s} {'trained mism':>13s} {'novel mism':>11s}")
for P, bits, nE in CONFIGS:
    pats = well_separated_patterns(P, 5, active_bits=bits, seed=11)
    for gain in (1.0, 5.0 / bits):
        for rec in (True, False):
            if not rec and gain == 1.0 and nE > 480:
                continue
            t0 = time.time()
            sysm = build_art_network(P, nE, 12, 0.6, True, True, True, seed=0, boost=4.0,
                                     memory_recurrence=rec, input_gain=gain)
            build = time.time() - t0
            run_schedule(sysm, [(f"p{i}", pats[i]) for i in range(3)], steps_per_pattern=150,
                         probe_steps=50, probe_warmup=50)
            mem = sysm["memory"]
            spk = np.zeros(nE, bool)

            def hook(t):
                spk[:] |= mem.spiked[:nE]
            t0 = time.time()
            trained = probe_only(sysm, pats[0], steps=200, warmup=300, hook=hook)
            ms_tick = 1000 * (time.time() - t0) / 500
            novel = probe_only(sysm, pats[3], steps=200, warmup=300)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            rows.append((P, bits, nE, gain, rec, build, ms_tick, rss, spk.mean(), trained, novel))
            print(f"{P:4d} {bits:4d} {nE:5d} {gain:5.2f} {str(rec):>5s} {build:8.2f} {ms_tick:8.3f} {rss:7.0f} "
                  f"{100 * spk.mean():17.1f} {trained:13.4f} {novel:11.4f}", flush=True)
