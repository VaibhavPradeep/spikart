# spikart

spikart is a biologically inspired Spiking Neural Network (SNN) research library built around Adaptive Resonance Theory (ART) principles and Determinantal Point Process (DPP) slot allocation. Designed for spatial pattern recognition and category learning, it provides diversity-aware memory allocation, top-down attention-routed inhibition, and history-independent memory probing via state restoration.



## Key Features

* **Read-Only State Restoration (`state_mode="rest"`)**: Evaluates network memory and pattern mismatch without state contamination, historical leakage, or altering underlying neuron dynamics.
* **DPP Slot Selection**: Mitigates pattern crosstalk using greedy MAP selection, lowering retest mismatch during multi-pattern learning.
* **Attention-Routed Inhibition**: Top-down inhibition routed via static templates by default, with opt-in min-max normalized learned keys (`use_learned_keys=True`).
* **Excerpt Retrieval**: Partial-pattern fragment matching supported via active-bit containment scoring (`retrieve`) and spiking activity readouts (`retrieve_spiking`).
* **Scalability Calibration**: Tested on memory substrates up to 1,920 excitatory neurons and 256-bit inputs using `input_gain` calibration and optional `memory_recurrence=False` decoupling.



## Installation

Install the core library from PyPI:

```bash
pip install spikart
Or install locally with developer dependencies:
pip install -e .[dev,experiments]
Quickstart Example
The following script illustrates initializing an ART network, storing a set of known patterns via run_schedule, and evaluating read-only mismatch scores between recognized and novel inputs:
import numpy as np
from spikart import build_art_network, run_schedule, probe_only, well_separated_patterns

# 1. Generate sample patterns (4 known, 2 novel)
patterns = well_separated_patterns(pattern_size=16, n_patterns=6, active_bits=5, seed=0)
known, novel = patterns[:4], patterns[4:]

# 2. Build the ART spiking network system
system = build_art_network(
    pattern_size=16,
    substrate_size=144,
    slot_size=12,
    vigilance=0.6,
    use_dpp=True,
    use_attention=True,
    use_stdp=True,
    seed=0,
    boost=4.0
)

# 3. Store the known pattern set
run_schedule(
    system,
    [(f"p{i}", p) for i, p in enumerate(known)],
    steps_per_pattern=500,
    probe_steps=200
)

# 4. Measure read-only mismatch scores (lower mismatch = recognized)
known_mismatch = probe_only(system, known[0], steps=600)
novel_mismatch = probe_only(system, novel[0], steps=600)

print(f"Known Pattern Mismatch: {known_mismatch:.4f}")  # ~0.008
print(f"Novel Pattern Mismatch: {novel_mismatch:.4f}")  # ~0.038
```



## Results and Limitations
Across multi-seed benchmark evaluations (benchmark_baselines.py):
Performance: The spiking network ties classical binary ART-1 models on clean pattern recognition. On noisy bit-swap inputs, it trails symbolic template matchers by approximately 0.03 AUROC.
Ablation Contributions: Top-down attention routing provides the primary performance improvement across evaluated conditions. DPP slot selection provides a modest additional reduction in retest mismatch.
Probing: While state-restoration probing guarantees history-independent measurements, readout scores remain subject to discrete spike quantization over short time windows.

 

## Repository Structure and Versioning
Research iterations V1 through V6 are preserved as archived historical suites in this repository. The first installable PyPI release (0.1.0) packages the refined Version 6 core library from src/spikart/.
```bash
spikart/
├── src/
│   └── spikart/              # PyPI package (v0.1.0)
│       ├── __init__.py
│       └── snn_lib.py        # Core SNN engine
├── Version1/ ... Version6/   # Archived research iterations and benchmark runners
├── pyproject.toml            # Package metadata and build settings
├── README.md                 # Documentation
├── CHANGELOG.md              # Version release notes
└── LICENSE                   # MIT License
```
 

## Research Iteration Archive (Archived Versions)
v6.0.0 (Base for spikart 0.1.0): Extended benchmark probe windows to 600 ticks to reduce spike quantization artifacts on noisy inputs; added min-max key normalization options and partial-pattern retrieval functions (retrieve/retrieve_spiking).
v5.0.0: Introduced read-only state snapshotting (snapshot_state/restore_state), paired permutation statistical tests (stats_utils.py), and modular experiment runners.
v4.0.0: Unified ART substrate allocation, STDP plasticity, and attention routing loops into a consolidated module.
v3.0.0: Integrated pair-based STDP learning rules with decaying synaptic current traces (tau_syn).
v2.0.0: Early prototype testing DPP slot selection and softmax attention-routed inhibition.
v1.0.0: Initial two-column novelty detection prototype implementing basic leaky integrate-and-fire dynamics.

 

## Testing
Run unit tests directly or via pytest:
python Version6/test_snn_lib.py
or:
pytest

 

## License and Acknowledgments
This repository is open-source software released under the terms of the MIT License.
Note: Codebase refactoring, architectural prototyping, unit test expansion, and documentation were completed with AI assistance; the 24 unit tests and benchmark execution logs in this repository provide the validation suite.