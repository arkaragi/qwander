"""
Module providing functions to generate external potentials V(v) on graph
nodes for `qwander` based on graph centrality measures and attributes.

Each function accepts a NetworkX Graph or DiGraph and returns a mapping
from node to a real-valued potential V(v) = π * c(v), where c(v) is the
chosen centrality or attribute measure.

Available potentials:
  - degree_centrality_potential(G)
  - pagerank_potential(G, alpha)
  - random_potential(G, seed)
"""

from typing import Any, Dict, Optional
import networkx as nx
import numpy as np

__all__ = [
    "degree_centrality_potential",
    "pagerank_potential",
    "random_potential",
]

PI = np.pi


def degree_centrality_potential(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute a potential V(v) based on degree centrality:
    V(v) = π * degree_centrality(v)

    Parameters:
    -----------
    G: nx.Graph
        Undirected, simple graph.

    Returns:
    --------
    Dict[Any, float]
        Mapping from node v to V(v) = π * degree_centrality(v).
    """
    cent = nx.degree_centrality(G)
    return {v: PI * cent[v] for v in G.nodes()}


def pagerank_potential(G: nx.Graph,
                       alpha: float = 0.85) -> Dict[Any, float]:
    """
    Compute a potential V(v) based on PageRank: V(v) = π * pagerank(v)

    Parameters:
    -----------
    G: nx.Graph
        Undirected, simple graph.

    alpha: float, optional
        Damping factor for PageRank (default 0.85).

    Returns:
    --------
    Dict[Any, float]
        Mapping from node v to V(v) = π * pagerank(v).
    """
    pr = nx.pagerank(G, alpha=alpha)
    return {v: PI * pr[v] for v in G.nodes()}


def random_potential(G: nx.Graph,
                     seed: Optional[int] = None) -> Dict[Any, float]:
    """
    Assign a random potential V(v) = π * U(0,1) per node.

    Parameters:
    -----------
    G: nx.Graph
        Undirected, simple graph.

    seed: int, optional
        Seed for random number generator (for reproducibility).

    Returns:
    --------
    Dict[Any, float]
        Mapping from node v to V(v) = π * uniform_random(0,1).
    """
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        rng = seed
    elif seed is None:
        rng = np.random.default_rng()
    else:
        raise TypeError(f"seed must be int, Generator, or None, "
                        f"got {type(seed).__name__}.")
    return {v: PI * float(rng.random()) for v in G.nodes()}
