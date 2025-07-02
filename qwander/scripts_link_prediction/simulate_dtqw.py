#!/usr/bin/env python3
"""
Run node-wise Discrete-Time Quantum Walks (DTQWs) on all training graphs
from link prediction splits.

For every split, this script loops over all combinations, performs node-wise
DTQW simulations, and saves results to a dedicated output directory within
each split. Outputs include probability histories and a config metadata file
for reproducibility.
"""

import json
import time
from pathlib import Path
from typing import Tuple, Optional

import networkx as nx

from qwander.qwander.simulators.dtqw_simulator import DTQWSimulator
from qwander.qwander.utils.logger import logger, setup_logging

# Configure root logger for simple timestamped console output
setup_logging()
log = logger

# You can allow CLI override, or just hardcode the config path
CONFIG_PATH = Path(__file__).parent / "config_link_prediction.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Load dataset and paths from config file
DATASET = config["dataset"]
SPLITS_ROOT = Path(config["paths"]["splits_root"])
FACEBOOK_EDGE_PATH = Path(config["paths"]["facebook_edgelist"])
SPLIT_GLOB = f'{DATASET}*'

# Load simulation parameters from config file
COIN_TYPES = config["simulation"]["coin_types"]
POTENTIALS = config["simulation"]["potentials"]
STEPS = config["simulation"]["steps"]
RANDOM_SEED = config["simulation"]["random_seed"]


def load_train_graph(split_dir: Path) -> Tuple[nx.Graph, nx.Graph]:
    """
    Loads the full reference graph and a training subgraph for a given split.

    Parameters
    ----------
    split_dir: Path
        Path to the split directory (e.g., "karate_test10").

    Returns
    -------
    full: nx.Graph
        The original full graph for reference/visualization.

    train: nx.Graph
        The training subgraph (with held-out test edges removed).
    """
    if split_dir.name.startswith("karate"):
        # Karate: built-in graph (NetworkX), relabel nodes as strings for compatibility
        full = nx.karate_club_graph()
        full = nx.relabel_nodes(full, lambda x: str(x))

    elif split_dir.name.startswith("facebook"):
        # Facebook: load full graph from edge list (update path as needed)
        FACEBOOK_EDGE_PATH = Path("../../datasets/Facebook/facebook_combined.txt")
        full = nx.read_edgelist(FACEBOOK_EDGE_PATH, nodetype=str)

    else:
        raise ValueError(f"Unknown split type: {split_dir.name}")

    # Always load the split's train.edgelist as the training subgraph
    train = nx.read_edgelist(split_dir / "train.edgelist", nodetype=str)

    return full, train


def run_combo(G_train: nx.Graph,
              split_dir: Path,
              coin: str,
              pot: Optional[str]) -> None:
    """
    Run node-wise DTQW simulation on a training graph for a given setting.

    Parameters
    ----------
    G_train: nx.Graph
        The training subgraph for the split.

    split_dir: Path
        Path to the split directory.

    coin: str
        Coin type (e.g., "grover", etc.).

    pot: str or None
        Potential type; None disables potentials.

    Side Effects
    ------------
    Saves output files under a subdirectory of the split folder.
    Writes a config.json with experiment parameters and runtime.
    """
    pot_tag = pot if pot else "nopot"
    out_dir = split_dir / f"walks_{coin}_{pot_tag}_T{STEPS}"

    if out_dir.exists():
        log.info("Skip – already exists: %s", out_dir.name)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()

    # Run the DTQW simulation (one probability history per node)
    DTQWSimulator(G_train, coin_type=coin, random_state=RANDOM_SEED).run_nodewise(
        steps=STEPS,
        potential=pot,
        show_progress=True,
        out_path=str(out_dir)
    )

    elapsed = time.perf_counter() - tic
    log.info("[%s] (%s , %s) done in %.1f s",
             split_dir.name, coin, pot_tag, elapsed)

    # Save configuration and runtime metadata for reproducibility
    meta = {
        "coin_type":   coin,
        "steps":       STEPS,
        "potential":   pot,
        "random_seed": RANDOM_SEED,
        "num_nodes":   G_train.number_of_nodes(),
        "num_edges":   G_train.number_of_edges(),
        "duration_sec": round(elapsed, 2),
        "source_split": split_dir.name,
    }
    with (out_dir / "config.json").open("w") as fh:
        json.dump(meta, fh, indent=2)


def main() -> None:
    """
    Discover all split directories under SPLITS_ROOT, and for each,
    run node-wise DTQW simulations for every (coin, potential) combination.

    - Skips splits that are already simulated for the given settings.
    - Catches and logs any errors per split, continuing with others.
    - Reports at the end when all splits are processed.
    """
    # Find all matching split directories (e.g., karate_test10, facebook_test10, ...)
    split_dirs = sorted(SPLITS_ROOT.glob(SPLIT_GLOB), key=lambda p: p.name)
    if not split_dirs:
        log.error("No split folders found under %s matching '%s'.",
                  SPLITS_ROOT, SPLIT_GLOB)
        return

    # Loop over each split and simulate for all (coin, potential) combos
    for split_dir in split_dirs:
        log.info("=== Processing split: %s ===", split_dir.name)
        try:
            G_full, G_train = load_train_graph(split_dir)
            # Run every combination of coin and potential
            for coin in COIN_TYPES:
                for pot in POTENTIALS:
                    run_combo(G_train, split_dir, coin, pot)
        except Exception:
            log.exception("Failed on split %s", split_dir.name)

    log.info("All splits finished successfully.")


if __name__ == "__main__":
    main()
