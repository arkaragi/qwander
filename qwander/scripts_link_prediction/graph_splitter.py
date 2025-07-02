#!/usr/bin/env python3
"""
Build train/test splits for link prediction on the Karate Club or Facebook
ego-network datasets.

This script generates train/test edge splits for link prediction tasks.
It can handle both the Karate Club graph (from NetworkX) and the Facebook
ego-network graph (from the SNAP dataset). For each dataset, the script
creates multiple splits based on the specified test ratio(s).

Outputs
-------
- splits/{karate_testX}/ or splits/{facebook_testX}/ (depending on the dataset)
    - train.edgelist       : Edges used for training the model.
    - test_pos.edgelist    : Held-out edges for evaluating the model (positive test set).
    - test_neg.edgelist    : Negative samples (non-edges) used for evaluation.
    - split_info.json      : Metadata describing the split, including node/edge counts,
                             graph connectivity, and other statistics.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qwander.qwander.utils.logger import logger, setup_logging

# Configure root logger for simple timestamped console output
setup_logging()
log = logger

# Hardcoded config path for link prediction
CONFIG_PATH = Path(__file__).parent / "config_link_prediction.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Load dataset and paths from config file
DATASET = config["dataset"]
SPLITS_ROOT = Path(config["paths"]["splits_root"])
FACEBOOK_EDGE_PATH = Path(config["paths"]["facebook_edgelist"])

# Load split parameters from config file
TEST_RATIOS = config["split"]["test_ratios"]
DO_REDUCED = config["split"]["do_reduced"]
SEED = config["split"]["seed"]


def plot_graph_split(G_full: nx.Graph,
                     G_train: nx.Graph,
                     out_path: Path,
                     pos_edges: list = None,
                     neg_edges: list = None,
                     node_size: int = 200,
                     seed: int = 42):
    """
    Visualize graph splitting for link prediction.

    Creates a 4-panel plot:
      1. Full original graph.
      2. Training subgraph (edges used for training).
      3. Test positive edges (held out for evaluation).
      4. Test negative edges (sampled non-edges).

    Parameters
    ----------
    G_full : nx.Graph
        The original (unsplit) graph.
    G_train : nx.Graph
        The training graph after removing test edges.
    out_path : Path
        Where to save the resulting figure (PNG).
    pos_edges : list, optional
        List of positive test edges (if None, inferred from difference).
    neg_edges : list, optional
        List of negative samples (node pairs not in the graph).
    node_size : int, default=200
        Size of nodes in the visualization.
    seed : int, default=42
        Seed for node layout for reproducibility.
    """
    # Compute a consistent layout for all subplots
    pos = nx.spring_layout(G_full, seed=seed)

    train_edges = set(G_train.edges())
    test_edges = set(pos_edges) if pos_edges is not None else set(G_full.edges()) - train_edges

    nodes_full = set(G_full.nodes())
    nodes_train = set(G_train.nodes())
    isolates_train = nodes_full - set(n for e in train_edges for n in e)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=110)
    ax_full, ax_train, ax_pos, ax_neg = axes

    # ---- Panel 1: Full Graph ----
    nx.draw_networkx_nodes(
        G_full, pos, ax=ax_full,
        node_size=node_size, node_color="#111a24",
        alpha=0.96, linewidths=0.2, edgecolors="white"
    )
    nx.draw_networkx_edges(
        G_full, pos, ax=ax_full,
        edge_color="#b0b6b8", width=1.45, alpha=0.62
    )
    if G_full.number_of_nodes() <= 60:
        nx.draw_networkx_labels(
            G_full, pos, ax=ax_full, font_size=10, font_color="white"
        )
    ax_full.set_title("Full Graph", fontsize=13, fontweight='bold')
    ax_full.axis("off")

    # ---- Panel 2: Train Graph ----
    nx.draw_networkx_nodes(
        G_full, pos, ax=ax_train,
        nodelist=list(nodes_full - isolates_train),
        node_size=node_size, node_color="#111a24",
        alpha=0.97, linewidths=0.1, edgecolors="white"
    )
    # Mark isolated nodes in gray (nodes not in train edges)
    if isolates_train:
        nx.draw_networkx_nodes(
            G_full, pos, ax=ax_train,
            nodelist=list(isolates_train),
            node_size=node_size, node_color="#dddddd",
            alpha=0.30, linewidths=0.1, edgecolors="gray"
        )
    nx.draw_networkx_edges(
        G_full, pos, ax=ax_train,
        edgelist=list(train_edges), edge_color="deepskyblue",
        width=1.75, alpha=0.94
    )
    if G_full.number_of_nodes() <= 60:
        nx.draw_networkx_labels(
            G_full, pos, ax=ax_train, font_size=8, font_color="white"
        )
    ax_train.set_title("Train Graph", fontsize=13, fontweight='bold')
    ax_train.axis("off")

    # ---- Panel 3: Test Positives ----
    nx.draw_networkx_nodes(
        G_full, pos, ax=ax_pos,
        node_size=node_size, node_color="#111a24",
        alpha=0.97, linewidths=0.1, edgecolors="white"
    )
    if test_edges:
        nx.draw_networkx_edges(
            G_full, pos, ax=ax_pos,
            edgelist=list(test_edges), edge_color="crimson",
            width=2, alpha=0.80, style="dashed"
        )
    if G_full.number_of_nodes() <= 60:
        nx.draw_networkx_labels(
            G_full, pos, ax=ax_pos, font_size=8, font_color="white"
        )
    ax_pos.set_title("Test Positives", fontsize=13, fontweight='bold')
    ax_pos.axis("off")

    # ---- Panel 4: Test Negatives ----
    nx.draw_networkx_nodes(
        G_full, pos, ax=ax_neg,
        node_size=node_size, node_color="#111a24",
        alpha=0.97, linewidths=0.1, edgecolors="white"
    )
    if neg_edges:
        nx.draw_networkx_edges(
            G_full, pos, ax=ax_neg,
            edgelist=list(neg_edges), edge_color="orange",
            width=2.0, alpha=0.55, style="dotted"
        )
    if G_full.number_of_nodes() <= 60:
        nx.draw_networkx_labels(
            G_full, pos, ax=ax_neg, font_size=8, font_color="white"
        )
    ax_neg.set_title("Test Negatives", fontsize=13, fontweight='bold')
    ax_neg.axis("off")

    # ---- Main Figure Title ----
    plt.suptitle(
        "Graph Split Visualization: Full / Train / Test+ / Test-",
        fontsize=16, y=1.05, fontweight='bold', color="#142744"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def safe_edge_split(G: nx.Graph,
                    test_ratio: float,
                    rng: random.Random) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split the edges of a graph into train and test sets for link prediction.
    Ensures the training graph is connected and has minimum degree ≥ 1 for all
    nodes.

    Parameters
    ----------
    G : nx.Graph
        The input graph (must be connected).

    test_ratio : float
        Fraction of edges to hold out as test positives.

    rng : random.Random
        Random number generator instance (for reproducibility).

    Returns
    -------
    e_train : List[Tuple[str, str]]
        List of edges in the training set (used to build the train graph).

    e_test : List[Tuple[str, str]]
        List of edges in the test positive set (held out for evaluation).

    Raises
    ------
    RuntimeError
        If not enough edges can be held out without disconnecting the graph
        or violating min-degree.
    """
    G_work = G.copy()
    edges = list(G_work.edges())
    rng.shuffle(edges)  # Shuffle edges for random selection

    target = int(round(test_ratio * len(edges)))
    e_test = []  # Edges to be removed for testing
    e_train = set(edges)  # All edges start as train edges

    for u, v in edges:
        if len(e_test) >= target:
            break
        # Prevent removing edge if it would isolate either node
        if G_work.degree(u) <= 1 or G_work.degree(v) <= 1:
            continue
        G_work.remove_edge(u, v)
        # Ensure training graph remains connected
        if nx.is_connected(G_work):
            e_test.append((u, v))
            e_train.remove((u, v))
        else:
            # Restoration required if removal disconnects the graph
            G_work.add_edge(u, v)

    if len(e_test) < target:
        raise RuntimeError(
            f"Could not allocate {target} test edges for ratio={test_ratio:.2f}."
            " Try lowering test_ratio."
        )

    return list(e_train), e_test


def sample_non_edges(G: nx.Graph,
                     n_samples: int,
                     rng: random.Random) -> List[Tuple[str, str]]:
    """
    Uniformly sample non-edge node pairs (negative samples) from a graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    n_samples : int
        Number of non-edges (node pairs not connected by an edge) to sample.

    rng : random.Random
        A random number generator instance (for reproducibility).

    Returns
    -------
    non_edges : List[Tuple[str, str]]
        List of randomly chosen node pairs that are not connected in G.

    Raises
    ------
    ValueError
        If more negatives are requested than available in the graph.
    """
    # List all non-edge pairs in the graph
    non_edges = list(nx.non_edges(G))
    if n_samples > len(non_edges):
        raise ValueError(
            f"More negatives requested ({n_samples}) than exist ({len(non_edges)})."
        )
    rng.shuffle(non_edges)
    return non_edges[:n_samples]


def write_edgelist(edges,
                   path: Path) -> None:
    """
    Write a list of (u, v) edge tuples to a file as an edgelist.

    Each line contains two node IDs separated by a space.

    Parameters
    ----------
    edges : iterable of tuple
        Edge list as (node1, node2) pairs.

    path : Path
        File path to write the edgelist.
    """
    with path.open("w") as fh:
        for u, v in edges:
            fh.write(f"{u} {v}\n")


def log_graph_stats(G: nx.Graph,
                    name: str = "Graph"):
    """
    Log summary statistics of a given graph.

    Parameters
    ----------
    G : nx.Graph
        The graph to summarize.

    name : str, default="Graph"
        Optional label for the graph in the output.
    """
    degrees = [d for _, d in G.degree()]
    isolates = list(nx.isolates(G))
    connected = nx.is_connected(G)

    log.info(f"{name} stats:")
    log.info(f"  Nodes:             {G.number_of_nodes()}")
    log.info(f"  Edges:             {G.number_of_edges()}")
    log.info(f"  Average degree:    {np.mean(degrees):.2f}")
    log.info(f"  Min degree:        {min(degrees)}")
    log.info(f"  Max degree:        {max(degrees)}")
    log.info(f"  Isolated nodes:    {len(isolates)}")
    log.info(f"  Connected:         {connected}")
    if not connected:
        largest_cc = max(nx.connected_components(G), key=len)
        log.info(f"  Largest component: {len(largest_cc)} nodes")
    log.info("--------------------------------------------")


def build_and_save_split(G: nx.Graph,
                         split_tag: str,
                         test_ratios: List[float],
                         meta_extra=None):
    """
    Perform edge splits for link prediction, save to disk, and optionally plot splits.

    For each ratio in `test_ratios`, produces a directory with:
      - train.edgelist
      - test_pos.edgelist
      - test_neg.edgelist
      - split_info.json
      - graph_split.png (for small graphs, e.g. karate)

    Parameters
    ----------
    G : nx.Graph
        The input (full) graph to split.
    split_tag : str
        Base tag for naming the split directories (e.g., "karate" or "facebook").
    test_ratios : list of float
        Ratios of test edges to use (e.g., [0.1, 0.25, 0.5]).
    meta_extra : dict, optional
        Additional metadata to include in split_info.json.

    Side Effects
    ------------
    Writes files to disk in directories under SPLITS_ROOT.

    Raises
    ------
    AssertionError
        If the train/test edge sets overlap (should never happen).
    RuntimeError
        If not enough test edges can be safely allocated.
    """
    rng = random.Random(SEED)
    np.random.seed(SEED)

    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    done_dirs: List[str] = []

    for ratio in test_ratios:
        # Create output directory for this split
        out_dir = SPLITS_ROOT / f"{split_tag}_test{int(ratio * 100)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Split edges into train and test (positive) ensuring graph connectivity
        e_train, e_pos = safe_edge_split(G, test_ratio=ratio, rng=rng)
        # Sample an equal number of non-edges as negatives
        e_neg = sample_non_edges(G, n_samples=len(e_pos), rng=rng)

        # Write the edgelist files
        write_edgelist(e_train, out_dir / "train.edgelist")
        write_edgelist(e_pos, out_dir / "test_pos.edgelist")
        write_edgelist(e_neg, out_dir / "test_neg.edgelist")

        # For small graphs, provide a visualization
        if split_tag == "karate":
            plot_path = out_dir / "graph_split.png"
            plot_graph_split(G, nx.Graph(e_train), plot_path, e_pos, e_neg)

        # Gather and save metadata for reproducibility and diagnostics
        meta = {
            "seed": SEED,
            "test_ratio": ratio,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "n_train": len(e_train),
            "n_test_pos": len(e_pos),
            "n_test_neg": len(e_neg),
            "connected": nx.is_connected(nx.Graph(e_train)),
            "min_degree": min(dict(nx.Graph(e_train).degree()).values()),
        }
        if meta_extra:
            meta.update(meta_extra)
        with (out_dir / "split_info.json").open("w") as fh:
            json.dump(meta, fh, indent=2)

        log.info(f"✓ split {ratio:.0%} saved → {out_dir}")
        done_dirs.append(out_dir.as_posix())

        # Assert that there is no overlap between train and test sets
        assert not (set(e_train) & set(e_pos)), "Train and test edge sets overlap!"

    # Summary of all completed splits
    log.info("All splits finished:")
    for d in done_dirs:
        log.info(d)


def main():
    """
    Main function for loading datasets, performing graph splits,
    and saving the results.

    This function handles the Karate Club and Facebook datasets.
    It loads the appropriate graph, performs train/test splits,
    and saves the resulting files in the 'splits' directory.

    The function is designed to handle both a full Facebook graph
    and a reduced subgraph (for scalability with large graphs).

    Raises
    ------
    ValueError
        If an invalid dataset is provided.
    FileNotFoundError
        If the Facebook edge list is not found at the specified path.
    RuntimeError
        If the graph is not connected or contains isolated nodes.
    """

    # Handle the Karate Club dataset
    if DATASET == "karate":
        log.info("Loading Karate Club graph...")
        G_full = nx.karate_club_graph()
        G_full = nx.relabel_nodes(G_full, lambda x: str(x))  # Ensure node labels are strings

        log_graph_stats(G_full, "Karate Club")

        # Build and save train/test splits for all specified test ratios
        tag = "karate"
        meta_extra = {"graph": "Karate Club"}
        build_and_save_split(G_full, tag, TEST_RATIOS, meta_extra)

    # Handle the Facebook dataset
    elif DATASET == "facebook":
        log.info(f"Loading Facebook network from: {FACEBOOK_EDGE_PATH}")

        # Ensure the Facebook edge list file exists
        if not FACEBOOK_EDGE_PATH.exists():
            raise FileNotFoundError(f"Facebook edge list not found at: {FACEBOOK_EDGE_PATH}")

        # Load the full Facebook network from the edge list
        G_full = nx.read_edgelist(FACEBOOK_EDGE_PATH, nodetype=str)
        log.info(f"Loaded Facebook: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges.")

        # --- Optionally: Use a random subgraph for scalability ---
        if DO_REDUCED:
            log.info("Extracting random subgraph (33%% of nodes, using largest connected component)...")
            all_nodes = list(G_full.nodes())
            rng = random.Random(SEED)
            chosen = set(rng.sample(all_nodes, len(all_nodes) // 3))
            G_sub = G_full.subgraph(chosen).copy()

            # Ensure the subgraph is connected
            if not nx.is_connected(G_sub):
                G_sub = G_sub.subgraph(max(nx.connected_components(G_sub), key=len)).copy()

            # Remove isolated nodes (if any)
            isolates = list(nx.isolates(G_sub))
            if isolates:
                log.info(f"Removing {len(isolates)} isolated node(s) from subgraph.")
                G_sub.remove_nodes_from(isolates)

            log_graph_stats(G_sub, "Facebook random subgraph (LCC)")
            log.info(f"Subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges.")

            tag = "facebook_reduced"
            meta_extra = {
                "graph": "Facebook (ego-net SNAP, random reduced-subgraph)",
                "reduced": True,
                "original_nodes": G_full.number_of_nodes(),
                "original_edges": G_full.number_of_edges(),
                "subgraph_nodes": G_sub.number_of_nodes(),
                "subgraph_edges": G_sub.number_of_edges(),
                "seed": SEED,
            }
            build_and_save_split(G_sub, tag, TEST_RATIOS, meta_extra)

        else:
            # Use the full Facebook network (only recommended for classical baselines)
            if not nx.is_connected(G_full):
                raise RuntimeError("Facebook graph is not connected; check your edge list file.")

            # Remove isolated nodes from the full graph, just in case
            isolates = list(nx.isolates(G_full))
            if isolates:
                log.info(f"Removing {len(isolates)} isolated node(s) from full Facebook graph.")
                G_full.remove_nodes_from(isolates)

            log_graph_stats(G_full, "Facebook (cleaned)")

            tag = "facebook"
            meta_extra = {"graph": "Facebook (ego-net SNAP)", "reduced": False}
            build_and_save_split(G_full, tag, TEST_RATIOS, meta_extra)

    else:
        raise ValueError(f"Unknown DATASET: {DATASET}. Please use 'karate' or 'facebook'.")


if __name__ == "__main__":
    main()
