"""
spikart
=======
Spiking Neural Network research library using Adaptive Resonance Theory (ART)
and Determinantal Point Process (DPP) slot allocation.
"""

from .snn_lib import (
    LIFPopulation,
    Projection,
    Network,
    SlottedARTMemory,
    AttentionRouter,
    build_art_network,
    run_schedule,
    probe_only,
    snapshot_state,
    restore_state,
    state_digest,
    well_separated_patterns,
    force_overlap,
    retrieve_spiking,
    greedy_dpp_map,
    greedy_dpp_map_from_features,
    internal_weights,
)

__version__ = "0.1.1"

__all__ = [
    "LIFPopulation",
    "Projection",
    "Network",
    "SlottedARTMemory",
    "AttentionRouter",
    "build_art_network",
    "run_schedule",
    "probe_only",
    "snapshot_state",
    "restore_state",
    "state_digest",
    "well_separated_patterns",
    "force_overlap",
    "retrieve_spiking",
    "greedy_dpp_map",
    "greedy_dpp_map_from_features",
    "internal_weights",
]