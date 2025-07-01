"""
This module provides standalone functions to compute node embeddings using
DeepWalk, and Node2Vec.
"""

import functools
import time

import networkx as nx
import numpy as np

from qwander.qwander.embeddings.classical import DeepWalk, Node2Vec

__all__ = [
    "run_deepwalk",
    "run_node2vec"
]


def timeit(func):
    """
    Decorator that measures wall-clock execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f} seconds\n")
        return result

    return wrapper


@timeit
def run_deepwalk(G: nx.Graph,
                 node_list: list,
                 dimensions: int = 64,
                 walk_length: int = 40,
                 num_walks: int = 10,
                 window_size: int = 5,
                 epochs: int = 3,
                 seed: int = 42) -> np.ndarray:
    """
    Fit DeepWalk on G and return an embedding matrix whose rows follow node_list.
    """
    # Initialize DeepWalk model with given parameters
    model = DeepWalk(
        graph=G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        window_size=window_size,
        epochs=epochs,
        seed=seed,
    )

    # Train the model
    model.fit()

    # Get node embeddings as a dictionary {node: embedding vector}
    emb_dict = model.get_embeddings()

    # Stack embeddings in the order specified by node_list
    return np.vstack([emb_dict[n] for n in node_list]).astype(np.float32)


@timeit
def run_node2vec(G: nx.Graph,
                 node_list: list,
                 dimensions: int = 64,
                 walk_length: int = 40,
                 num_walks: int = 10,
                 window_size: int = 5,
                 epochs: int = 3,
                 seed: int = 42,
                 p: float = 0.25,
                 q: float = 0.75) -> np.ndarray:
    """
    Fit Node2Vec on G and return an embedding matrix whose rows follow node_list.
    """
    # Initialize Node2Vec model with given parameters
    model = Node2Vec(
        graph=G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        window_size=window_size,
        epochs=epochs,
        seed=seed,
        p=p,
        q=q,
    )

    # Train the model
    model.fit()

    # Get node embeddings as a dictionary {node: embedding vector}
    emb_dict = model.get_embeddings()

    # Stack embeddings in the order specified by node_list
    return np.vstack([emb_dict[n] for n in node_list]).astype(np.float32)
