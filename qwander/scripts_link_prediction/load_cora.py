#!/usr/bin/env python3
"""
Load the Cora citation network as a NetworkX graph.

Expects the folder structure:

    Cora/
    ├── cora.content   (node features + labels, one per line)
    └── cora.cites     (directed citations: source target)

This builds an undirected graph over the citation edges.
"""

from pathlib import Path

import networkx as nx


def load_cora(cora_dir: Path) -> nx.Graph:
    """
    Load the Cora dataset from the given directory.

    Parameters
    ----------
    cora_dir : Path
        Path to the folder containing 'cora.content' and 'cora.cites'.

    Returns
    -------
    G : nx.Graph
        Undirected graph where each Cora paper is a node,
        and edges represent citations (treated as undirected).
    """
    content_path = cora_dir / "cora.content"
    cites_path   = cora_dir / "cora.cites"

    # Read all paper IDs (to include isolated nodes)
    nodes = []
    with content_path.open("r") as f:
        for line in f:
            paper_id, *rest = line.split()
            nodes.append(paper_id)

    # Build graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add undirected edges from citations
    with cites_path.open("r") as f:
        for line in f:
            src, dst = line.split()
            if src in G and dst in G:
                G.add_edge(src, dst)

    return G


if __name__ == "__main__":
    # Locate Cora/ under datasets/
    cora_folder = Path(__file__).parents[2] / "datasets" / "Cora"
    G = load_cora(cora_folder)
    print(f"Loaded Cora graph with {G.number_of_nodes()} nodes "
          f"and {G.number_of_edges()} edges.")
