"""
This module defines DeepWalk and Node2Vec graph embedding classes.

It provides two concrete subclasses of GraphEmbeddingModel:
- DeepWalk: Implements uniform random walks followed by skip-gram Word2Vec.
- Node2Vec: Implements biased random walks followed by skip-gram Word2Vec.

Both classes inherit training, embedding retrieval, and persistence methods
from the GraphEmbeddingModel base class. Import these classes to generate
node embeddings from any NetworkX graph using classical walk-based approaches.
"""

import random
from typing import List, Optional

import networkx as nx

from qwander.qwander.embeddings.base import GraphEmbeddingModel
from qwander.qwander.utils.logger import logger


class DeepWalk(GraphEmbeddingModel):
    """
    DeepWalk: uniform random walks + skip-gram Word2Vec.

    Inherits all hyperparameters and I/O methods from GraphEmbeddingModel.
    """

    def _generate_walks(self) -> List[List[str]]:
        """
        Generate uniform random walks from each node.

        Returns
        -------
        List[List[str]]
            A list of walks; each walk is a list of node IDs (strings).
        """
        walks: List[List[str]] = []
        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for start in nodes:
                walk = [start]
                for _ in range(self.walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(self.graph.neighbors(curr))
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))
                walks.append(walk)

        return walks


class Node2Vec(GraphEmbeddingModel):
    """
    Node2Vec: biased random walks with return parameter p and in-out parameter q.

    Inherits all hyperparameters from GraphEmbeddingModel plus:
    - p: return parameter (likelihood of revisiting previous node)
    - q: in-out parameter (bias between BFS and DFS)
    """

    def __init__(self,
                 graph: nx.Graph,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 window_size: int = 10,
                 epochs: int = 5,
                 workers: int = 4,
                 seed: Optional[int] = None,
                 negative: int = 5,
                 hs: bool = False,
                 alpha: float = 0.025,
                 p: float = 1.0,
                 q: float = 1.0) -> None:
        super().__init__(
            graph=graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            window_size=window_size,
            epochs=epochs,
            workers=workers,
            seed=seed,
            negative=negative,
            hs=hs,
            alpha=alpha
        )
        self.p: float = p
        self.q: float = q
        logger.info(f"Node2Vec initialized (p={self.p}, q={self.q})")

    def _generate_walks(self) -> List[List[str]]:
        """
        Generate biased random walks from each node using Node2Vec strategy.

        Returns
        -------
        List[List[str]]
            A list of walks; each walk is a list of node IDs (strings).
        """
        walks: List[List[str]] = []
        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for start in nodes:
                walk = [start]
                if self.walk_length == 1:
                    walks.append(walk)
                    continue

                # First step: choose uniformly among neighbors of `start`
                first_neighbors = list(self.graph.neighbors(start))
                if not first_neighbors:
                    walks.append(walk)
                    continue

                walk.append(random.choice(first_neighbors))

                # Subsequent steps: biased transition
                while len(walk) < self.walk_length:
                    prev_node = walk[-2]
                    curr_node = walk[-1]
                    neighbors = list(self.graph.neighbors(curr_node))
                    if not neighbors:
                        break

                    next_node = self._biased_choice(prev_node, neighbors)
                    walk.append(next_node)

                walks.append(walk)

        return walks

    def _biased_choice(self,
                       prev: str,
                       neighbors: List[str]) -> str:
        """
        Select the next node based on Node2Vec transition weights.

        Parameters
        ----------
        prev: str
            The previous node in the walk.

        neighbors: List[str]
            List of neighbor node IDs (strings) of the current node.

        Returns
        -------
        str
            The chosen next node ID.
        """
        weights: List[float] = []
        for nbr in neighbors:
            if nbr == prev:
                # returning to the previous node
                weights.append(1.0 / self.p)
            elif self.graph.has_edge(nbr, prev):
                # neighbor is close to previous (distance = 1)
                weights.append(1.0)
            else:
                # neighbor is “farther away” (distance > 1)
                weights.append(1.0 / self.q)

        total = sum(weights)
        if total == 0:
            return random.choice(neighbors)

        probabilities = [w / total for w in weights]
        return random.choices(neighbors, weights=probabilities, k=1)[0]
