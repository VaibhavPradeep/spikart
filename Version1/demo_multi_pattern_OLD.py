"""
demo_multi_pattern_OLD.py
=========================
Demonstrates basic multi-pattern habituation and novelty signaling on 
the Version 1 spiking network architecture.
"""

import numpy as np
from snn_lib_OLD import LIFPopulation, Projection, Network

np.random.seed(42)

N = 8          # pattern size / excitatory width of each column
N_INH = 2
DT = 0.1

def make_column(name):
    return LIFPopulation(N + N_INH, n_inhibitory=N_INH, name=name)

def internal_weights(n_exc, n_inh):
    w = np.zeros((n_exc + n_inh, n_exc + n_inh))
    w[:n_exc, :n_exc] = np.random.uniform(1.5, 3.5, (n_exc, n_exc))
    np.fill_diagonal(w[:n_exc, :n_exc], 0.0)
    w[:n_exc, n_exc:] = -6.0
    w[n_exc:, :n_exc] = 4.0
    return w

net = Network()

inp = net.add_population(LIFPopulation(N, n_inhibitory=0, name="Input"))
mem = net.add_population(make_column("Memory"))
ev  = net.add_population(make_column("Evaluator"))
ev.rectify_input = True   # matches original: Evaluator's drive is rectified

# recurrent E/I connectivity, static (unchanged from the original scripts)
net.add_projection(Projection(mem, mem, internal_weights(N, N_INH), plasticity=None))
net.add_projection(Projection(ev, ev, internal_weights(N, N_INH), plasticity=None))

# Input -> Memory: REAL STDP (this is the part that used to be fake)
w_in_mem = np.zeros((N + N_INH, N))
w_in_mem[:N, :] = np.random.uniform(4.0, 8.0, (N, N))
proj_in_mem = net.add_projection(Projection(
    inp, mem, w_in_mem, plasticity="stdp",
    A_plus=0.15, A_minus=0.12, w_min=0.0, w_max=60.0, tau_syn=5.0,
    name="Input->Memory"))

# Input -> Evaluator: static direct sensory drive (identity block)
w_in_ev = np.zeros((N + N_INH, N))
w_in_ev[:N, :N] = np.eye(N) * 55.0
net.add_projection(Projection(inp, ev, w_in_ev, plasticity=None, tau_syn=5.0))

# Memory -> Evaluator: static point-to-point inhibition (identity block, negative)
w_mem_ev = np.zeros((N + N_INH, N + N_INH))
w_mem_ev[:N, :N] = -np.eye(N) * 110.0
net.add_projection(Projection(mem, ev, w_mem_ev, plasticity=None, tau_syn=5.0))

INPUT_DRIVE = 180.0

def run_pattern(pattern, steps):
    pattern = np.asarray(pattern, dtype=float)
    mismatches = []
    for _ in range(steps):
        ext = {"Input": pattern * INPUT_DRIVE}
        net.step(ext, dt=DT)
        mismatches.append(ev.spiked[:N].sum())
    return np.array(mismatches)

p1 = [1, 1, 0, 0, 1, 0, 0, 1]
p2 = [0, 0, 1, 1, 0, 1, 1, 0]
p3 = [1, 0, 1, 0, 1, 0, 1, 0]
p4 = [0, 1, 0, 1, 0, 1, 0, 1]

STEPS = 1500

schedule = [
    ("p1 (1st)",  p1), ("p1 (retest)", p1),
    ("p2 (1st)",  p2), ("p2 (retest)", p2),
    ("p3 (1st)",  p3), ("p3 (retest)", p3),
    ("p4 (1st)",  p4), ("p4 (retest)", p4),
    ("p1 (after learning p2-p4)", p1),
    ("p2 (after learning p3-p4)", p2),
]

print(f"{'exposure':32s}  mean mismatch   total mismatch spikes")
print("-" * 70)
for label, pat in schedule:
    m = run_pattern(pat, STEPS)
    print(f"{label:32s}  {m.mean():.4f}          {m.sum():.0f}")
