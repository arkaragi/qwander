"""
Base class for classical walk‐based graph embeddings (DeepWalk, Node2Vec).
"""

import abc
import random
from pathlib import Path
from typing import List, Dict, Optional, Union

import networkx as nx
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

from qwander.qwander.utils.logger import logger


class GraphEmbeddingModel(abc.ABC):
    """
    Abstract base class for graph embedding models based on random walks + skip-gram.
    Subclasses must implement the _generate_walks method to produce a list of "sentences"
    over node strings. This base class handles training the Word2Vec skip-gram model and
    exposes methods to retrieve or save embeddings.

    Attributes
    ----------
    graph: nx.Graph
        The input graph with string-labeled nodes.

    dimensions: int
        Embedding dimensionality.

    walk_length: int
        Number of steps (tokens) per walk.

    num_walks: int
        Number of walks to start from each node.

    window_size: int
        Context window size for skip-gram.

    epochs: int
        Number of training epochs for Word2Vec.

    workers: int
        Number of worker threads for Word2Vec.

    seed: int
        Random seed for reproducibility.

    negative: int
        Number of negative samples for skip-gram (negative sampling).

    hs: bool
        Whether to use hierarchical softmax instead of negative sampling.

    alpha: float
        Initial learning rate for Word2Vec.

    walks: List[List[str]]
        List of generated walks (populated after fit()).

    model: Optional[Word2Vec]
        Trained Word2Vec skip-gram model (after fit()).
    """

    def __init__(self,
                 graph: nx.Graph,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 window_size: int = 10,
                 epochs: int = 5,
                 workers: int = 1,
                 seed: Optional[int] = None,
                 negative: int = 5,
                 hs: bool = False,
                 alpha: float = 0.025) -> None:
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        self.seed: int = seed

        # Ensure reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Copy and relabel nodes to strings
        self.graph: nx.Graph = graph.copy()
        self.graph = nx.relabel_nodes(self.graph, lambda x: str(x))

        self.dimensions: int = dimensions
        self.walk_length: int = walk_length
        self.num_walks: int = num_walks
        self.window_size: int = window_size
        self.epochs: int = epochs
        self.workers: int = workers
        self.negative: int = negative
        self.hs: bool = hs
        self.alpha: float = alpha

        # Placeholder for walks and trained model
        self.walks: List[List[str]] = []
        self.model: Optional[Word2Vec] = None

        logger.info(f"{self.__class__.__name__} initialized (seed={self.seed}).")

    @abc.abstractmethod
    def _generate_walks(self) -> List[List[str]]:
        """
        Generate a list of walks (each walk is a list of node-IDs as strings).
        Must be implemented by subclasses (DeepWalk, Node2Vec, etc.).

        Returns
        -------
        List[List[str]]
            List of walks, where each walk is a list of node strings.
        """
        pass

    def fit(self) -> None:
        """
        Generate walks and train the Word2Vec skip-gram model.
        """
        logger.info(f"{self.__class__.__name__}: starting walk generation.")

        self.walks = self._generate_walks()
        if not self.walks:
            raise RuntimeError("No walks generated; check _generate_walks implementation.")

        logger.info(f"{self.__class__.__name__}: "
                    f"generated {len(self.walks)} walks; training skip-gram.")

        # Train skip-gram Word2Vec
        self.model = Word2Vec(
            sentences=self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,  # skip-gram
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
            negative=self.negative,
            hs=self.hs,
            alpha=self.alpha
        )

        logger.info(f"{self.__class__.__name__}: skip-gram training completed.")

    def get_embedding(self,
                      node: Union[str, int]) -> np.ndarray:
        """
        Retrieve the embedding vector for a single node.

        Parameters
        ----------
        node: Union[str, int]
            Node ID (string or integer). Will be cast to string.

        Returns
        -------
        np.ndarray
            Embedding vector of shape (dimensions,).

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        KeyError
            If the node is not present in the trained vocabulary.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        key = str(node)
        return self.model.wv.get_vector(key)

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve embeddings for all nodes in the trained model vocabulary.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from node ID (string) to embedding vector (np.ndarray).

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        return {
            node: self.model.wv.get_vector(node)
            for node in self.model.wv.index_to_key
        }

    def load_embeddings(self,
                        path: Union[str, Path]) -> KeyedVectors:
        """
        Load pre-trained embeddings from disk.

        Parameters
        ----------
        path: Union[str, Path]
            File path to KeyedVectors (produced by save_embeddings).

        Returns
        -------
        KeyedVectors
            Loaded embeddings.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        Notes
        -----
        This does not modify self.model; it returns a standalone KeyedVectors instance.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"No embeddings file found at {in_path}")

        kv = KeyedVectors.load(str(in_path), mmap="r")
        logger.info(f"Loaded embeddings from {in_path}")

        return kv

    def save_embeddings(self,
                        path: Union[str, Path]) -> None:
        """
        Save embeddings to disk in Gensim KeyedVectors format.

        Parameters
        ----------
        path: Union[str, Path]
            File path (e.g., ending in .kv or .bin) where embeddings will be saved.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained; cannot save embeddings.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.wv.save(str(out_path))
        logger.info(f"Embeddings saved to {out_path}")

    def load_model(self,
                   path: Union[str, Path]) -> None:
        """
        Load a pre-trained Word2Vec model from disk and assign it to self.model.

        Parameters
        ----------
        path: Union[str, Path]
            File path to the saved Word2Vec model.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"No model file found at {in_path}")

        self.model = Word2Vec.load(str(in_path))
        logger.info(f"Word2Vec model loaded from {in_path}")

    def save_model(self,
                   path: Union[str, Path]) -> None:
        """
        Save the full Word2Vec model to disk (including its state and vocabulary).

        Parameters
        ----------
        path: Union[str, Path]
            File path (e.g., ending in .model) where the Word2Vec model will be saved.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained; cannot save model.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(out_path))
        logger.info(f"Word2Vec model saved to {out_path}")

    def export_walks(self,
                     path: Union[str, Path]) -> None:
        """
        Save generated walks to disk as a text file.

        Parameters
        ----------
        path: Union[str, Path]
            File path (e.g., ending in .txt) where walks will be saved.
            Overwrites any existing file.

        Raises
        ------
        RuntimeError
            If no walks have been generated (i.e., fit() has not been called).
        """
        if not self.walks:
            raise RuntimeError("No walks to export. Have you called fit()?")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for walk in self.walks:
                f.write(" ".join(walk) + "\n")

        logger.info(f"Walks exported to {out_path}")
