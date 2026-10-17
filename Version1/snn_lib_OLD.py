"""
snn_lib_OLD.py
==============
Initial prototype implementation of the two-column spiking neural network 
and basic leaky integrate-and-fire (LIF) neuron dynamics.
"""

import numpy as np


class LIFPopulation:
    def __init__(self, n_neurons, n_inhibitory=0, v_rest=-70.0, v_th=-50.0,
                 v_reset=-75.0, tau_m=20.0, R=1.0, t_refractory=2.0,
                 tau_trace=20.0, name="pop"):
        self.name = name
        self.n = n_neurons
        self.n_inh = n_inhibitory
        self.n_exc = n_neurons - n_inhibitory
        self.v_rest, self.v_th, self.v_reset = v_rest, v_th, v_reset
        self.tau_m, self.R, self.t_refractory = tau_m, R, t_refractory
        self.tau_trace = tau_trace

        self.v = np.full(n_neurons, v_rest, dtype=float)
        self.refractory = np.zeros(n_neurons)
        self.spiked = np.zeros(n_neurons, dtype=bool)
        self.trace = np.zeros(n_neurons)   # STDP eligibility trace

    def step(self, I, dt=0.1):
        active = self.refractory <= 0
        dv = np.where(active,
                       (-(self.v - self.v_rest) + self.R * I) / self.tau_m * dt,
                       0.0)
        self.v = np.where(active, self.v + dv, self.v_reset)
        self.refractory = np.where(active, self.refractory, self.refractory - dt)

        fired = active & (self.v >= self.v_th)
        self.v = np.where(fired, self.v_reset, self.v)
        self.refractory = np.where(fired, self.t_refractory, self.refractory)
        self.spiked = fired

        self.trace *= np.exp(-dt / self.tau_trace)
        self.trace[fired] += 1.0
        return fired.astype(float)

    def reset(self):
        self.v[:] = self.v_rest
        self.refractory[:] = 0.0
        self.spiked[:] = False
        self.trace[:] = 0.0


class Projection:
    """
    weights shape: (n_target, n_source).

    Synaptic current is filtered through a decaying trace (tau_syn) rather
    than delivered as a single-timestep impulse — this is the standard
    current-based synapse model used in real SNN simulators (NEST, Brian2),
    and it matters a lot here: without it, a source population that only
    spikes occasionally (a real rate-coded input, instead of the original
    scripts' "inject the raw pattern as continuous current every step"
    shortcut) delivers almost no net charge and nothing downstream ever
    reaches threshold.
    """
    def __init__(self, source, target, weights, plasticity=None,
                 learning_rate=1.0, A_plus=0.06, A_minus=0.072,
                 w_min=0.0, w_max=45.0, tau_syn=5.0, mask=None,
                 gate=None, name=""):
        self.source, self.target = source, target
        self.weights = np.array(weights, dtype=float)
        self.plasticity = plasticity
        self.lr = learning_rate
        self.A_plus, self.A_minus = A_plus, A_minus
        self.w_min, self.w_max = w_min, w_max
        self.tau_syn = tau_syn
        self.psc = np.zeros(source.n)   # filtered synaptic current trace
        self.gate = gate          # optional callable() -> bool, gates 'hebbian'
        self.name = name or f"{source.name}->{target.name}"
        # connectivity mask: which (target, source) pairs are real synapses.
        # Without this, STDP happily "grows" brand-new connections wherever
        # a target neuron fires (e.g. onto inhibitory neurons that were
        # never supposed to receive this input) — a real bug I hit while
        # testing this. Default: only positions with a nonzero initial
        # weight are plastic-eligible.
        self.mask = (self.weights != 0) if mask is None else np.array(mask, dtype=bool)

    def pulse_and_decay(self, dt=0.1):
        """Call once per network tick, before current(). Advances the
        synaptic filter using the source's spikes from the previous tick."""
        self.psc *= np.exp(-dt / self.tau_syn)
        self.psc += self.source.spiked.astype(float)

    def current(self):
        return self.weights @ self.psc

    def update(self, pattern=None, dt=0.1):
        if self.plasticity is None:
            return

        if self.plasticity == "hebbian":
            # legacy rule from the original scripts: gated bulk update on
            # the raw pattern vector, no spike timing involved at all.
            if pattern is None:
                return
            if self.gate is not None and not self.gate():
                return
            self.weights += self.lr * np.outer(pattern, pattern)

        elif self.plasticity == "stdp":
            # real pair-based STDP: uses actual pre/post spikes this step
            # against the *other* side's decayed trace.
            post_fired, pre_fired = self.target.spiked, self.source.spiked
            pre_trace, post_trace = self.source.trace, self.target.trace
            if post_fired.any():
                self.weights[post_fired, :] += (self.A_plus * pre_trace[np.newaxis, :]
                                                 * self.mask[post_fired, :])
            if pre_fired.any():
                self.weights[:, pre_fired] -= (self.A_minus * post_trace[:, np.newaxis]
                                                * self.mask[:, pre_fired])

        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)


class Network:
    def __init__(self):
        self.populations = {}
        self.projections = []

    def add_population(self, pop):
        self.populations[pop.name] = pop
        return pop

    def add_projection(self, proj):
        self.projections.append(proj)
        return proj

    def step(self, external_currents, dt=0.1, pattern=None):
        """external_currents: dict {pop_name: array} of extra input current."""
        totals = {name: np.array(external_currents.get(name, np.zeros(p.n)), dtype=float)
                  for name, p in self.populations.items()}

        for proj in self.projections:
            proj.pulse_and_decay(dt)
            totals[proj.target.name][:proj.weights.shape[0]] += proj.current()

        for name, pop in self.populations.items():
            t = totals[name]
            if getattr(pop, "rectify_input", False):
                t = np.maximum(t, 0.0)
            pop.step(t, dt)

        for proj in self.projections:
            proj.update(pattern=pattern, dt=dt)

    def reset(self):
        for p in self.populations.values():
            p.reset()
        for proj in self.projections:
            proj.psc[:] = 0.0
