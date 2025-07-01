"""
This module provides a set of tools for converting the quantum dynamics
of Discrete-Time Quantum Walks into classical node embeddings for graph
based machine learning tasks.

The core abstraction is the DTQWEmbedder class, which takes precomputed
DTQW histories (recorded as sequences of complex-valued quantum states)
and converts them into classical feature vectors. These embeddings are
useful for various downstream tasks, such as node classification, link
prediction, clustering, and visualization, enabling the use of quantum
inspired features in classical machine learning models

Features
--------
- Final-Step Probability Embeddings: Convert the final quantum walk state
  into a fixed-length probability vector for each node.
- Time-Averaged Embeddings: Average quantum walk probabilities across time,
  with optional weighting for different time steps.
- Flattened Time-Series Embeddings: Convert the time series of quantum walk
  probabilities into a single vector, followed by dimensionality reduction
  using Truncated SVD.
- Kernel-Based Embeddings: Use Kernel PCA to create embeddings based on
  time-series similarity between quantum walks, with support for various
  similarity metrics.
"""

import pathlib
import pickle
import warnings

from typing import Any, Callable, Optional, Union

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from tqdm import tqdm


class DTQWEmbedderBase:
    """
    Base class for DTQW-based embedding models, providing methods for handling
    DTQW state histories and embedding matrices.

    Parameters
    ----------
    histories_path : str or pathlib.Path
        The path to a pickle file containing the 'nodewise_histories' key,
        which stores the DTQW state evolution for each node.

    show_progress : bool, optional, default=True
        Whether to display progress bars during embedding extraction.

    random_state : int or None, optional
        An optional random seed used in dimensionality reduction methods
        (e.g., TruncatedSVD, KernelPCA) for reproducibility.

    Attributes
    ----------
    histories : dict[Any, list[dict[Any, np.ndarray]]]
        A dictionary mapping each node to a list of quantum state dictionaries
        across time steps during the DTQW process.
    """

    def __init__(self,
                 histories_path: Union[str, pathlib.Path],
                 show_progress: bool = True,
                 random_state: Optional[int] = None):
        path = pathlib.Path(histories_path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        with path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "nodewise_histories" not in payload:
            raise ValueError("Invalid DTQW history file: missing 'nodewise_histories' key.")

        self.histories = payload["nodewise_histories"]
        self.show_progress = show_progress
        self.random_state = random_state
        self._nodes_sorted = sorted(self.histories.keys(), key=lambda x: str(x))

    @property
    def sorted_nodes(self) -> list[Any]:
        """
        Lexicographically sorted list of node labels.

        Returns
        -------
        list[Any]
            Sorted list of node labels used across all output embeddings.
        """
        return self._nodes_sorted

    @staticmethod
    def load_embeddings(filepath: Union[str, pathlib.Path]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Load a previously saved embedding matrix from a pickle file.

        Parameters
        ----------
        filepath: str or pathlib.Path
            Path to the pickle file containing the embeddings.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            The loaded node embeddings.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the unpickled content is not a valid embedding matrix
            (must be a DataFrame or ndarray).
        """
        path = pathlib.Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"No such embedding file: {path}")

        with path.open("rb") as fh:
            obj = pickle.load(fh)

        if not isinstance(obj, (pd.DataFrame, np.ndarray)):
            raise ValueError(f"Invalid object type: "
                             f"expected DataFrame or ndarray, got {type(obj)}")

        return obj

    @staticmethod
    def save_embeddings(embeddings: Union[pd.DataFrame, np.ndarray],
                        filepath: Union[str, pathlib.Path]) -> None:
        """
        Save an embedding matrix to disk using pickle.

        Parameters
        ----------
        embeddings: pandas.DataFrame or numpy.ndarray
            The node embeddings to persist.

        filepath: str or pathlib.Path
            Path to save the embeddings file; parent directories are created automatically.
        """
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(embeddings, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def extract_node_probabilities(history: list[dict[Any, np.ndarray]]) -> pd.DataFrame:
        """
        Convert a DTQW history into a DataFrame of node probabilities per time step.

        Parameters
        ----------
        history: list of dict[Any, np.ndarray]
            A list of dictionaries where each element maps node labels to complex
            amplitude vectors at a particular time step of the quantum walk.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the node-level probabilities:
            - Index   = Time steps (0, 1, …, T)
            - Columns = Node labels (sorted lexicographically)
            - Values  = Σ_i |ψ_t[v][i]|², the total probability at node v at time t.

        Raises
        ------
        ValueError
            If history is not a valid list of quantum state dictionaries or contains invalid data.
        """
        if not history or not isinstance(history, list):
            raise ValueError("History must be a non-empty list of quantum state dictionaries.")

        # Build per-time-step probability snapshots
        prob_list = [
            {v: np.abs(vec).dot(np.abs(vec)) for v, vec in state.items()}
            for state in history
        ]
        df = pd.DataFrame(prob_list).rename_axis(index="Time Step")

        # Sort columns lexicographically
        cols = sorted(df.columns, key=lambda x: str(x))

        return df[cols]


class DTQWEmbedder(DTQWEmbedderBase):
    """
    High-level embedding interface built on Discrete-Time Quantum Walks (DTQWs).
    Provides various strategies to convert quantum walk histories into classical
    node embeddings.

    Parameters
    ----------
    histories_path : str or pathlib.Path
        The path to a pickle file containing a dictionary mapping each node to
        a list of quantum state dictionaries over time.

    show_progress : bool, optional, default=True
        Whether to display progress bars during embedding extraction.

    random_state : int or None, optional
        An optional random seed used in downstream dimensionality reduction
        methods (e.g., TruncatedSVD, KernelPCA) to ensure reproducibility.
    """

    def __init__(self,
                 histories_path: Union[str, pathlib.Path],
                 show_progress: bool = True,
                 random_state: Optional[int] = None):
        super().__init__(histories_path, show_progress, random_state)

    def final_step_embeddings(self,
                              time_step: Optional[int] = None) -> pd.DataFrame:
        """
        Extract final-step DTQW probability distributions as node embeddings.

        For each node v, this method computes the probability distribution over
        all nodes u at the specified time step T of the quantum walk that started
        at node v. If no time step is specified, the last time step (T = -1) will
        be used by default.

        Parameters
        ----------
        time_step: int, optional
            The time step to use for the embeddings.
            If None, the last time step is used by default.

        Returns
        -------
        pd.DataFrame
            A DataFrame (N, N) where:
            - Rows    = Start nodes (sorted lexicographically)
            - Columns = Target nodes (sorted lexicographically)
            - Entry (v, u) = Pr(walker at u | started at v, time = T)
        """
        nodes = self.sorted_nodes
        final_probs = {}

        iterator = tqdm(nodes, desc="Final-step embeddings") if self.show_progress else nodes
        for v in iterator:
            df_probs = self.extract_node_probabilities(self.histories[v])

            # Use the specified time step or default to the last one
            step = time_step if time_step is not None else -1
            final_probs[v] = df_probs.iloc[step].reindex(nodes)

        df_embed = pd.DataFrame.from_dict(final_probs, orient="index", columns=nodes)
        df_embed.index.name = "start_node"
        df_embed.columns.name = "target_node"

        return df_embed

    def concatenated_embeddings(self,
                                n_components: int = 64,
                                time_step: Optional[int] = None,
                                use_pca: bool = False) -> pd.DataFrame:
        """
        Flatten each node's time-series into a single vector at a specified
        time step, then apply dimensionality reduction.

        This forms a compressed representation of the quantum walk evolution
        at the specified time step. You can choose between Truncated SVD or
        PCA for dimensionality reduction.

        Parameters
        ----------
        n_components: int
            Number of dimensions to keep after applying dimensionality reduction.

        time_step: int, optional
            The time step to use for the embeddings.
            If None, the last time step is used by default.

        use_pca: bool, optional, default=False
            Whether to use PCA for dimensionality reduction.
            If False, Truncated SVD is used.

        Returns
        -------
        pandas.DataFrame
            A (N, n_components) DataFrame where:
            - Rows    = nodes (sorted lexicographically)
            - Columns = "svd_0", "svd_1", ..., "svd_{n_components-1}" or
                        "pca_0", "pca_1", ..., "pca_{n_components-1}"
        """
        nodes = self.sorted_nodes
        series = {}

        # Define the iterator for progress
        iterator = tqdm(nodes, desc="Flattening time-series") \
            if self.show_progress else nodes
        for v in iterator:
            # Extract the node probabilities
            df_probs = self.extract_node_probabilities(self.histories[v])

            # Use the specified time step or default to the last one
            step = time_step if time_step is not None else -1
            # Flatten the probabilities at the chosen time step
            series[v] = df_probs.iloc[step][nodes].values.ravel()

        # Create a DataFrame from the flattened time series
        df_matrix = pd.DataFrame.from_dict(series, orient="index")

        # Ensure the requested number of components is within valid limits
        max_dim = min(len(nodes), df_matrix.shape[1])
        if not (1 <= n_components <= max_dim):
            raise ValueError(f"'n_components' must be in [1, {max_dim}]")

        # Apply PCA or Truncated SVD for dimensionality reduction
        if use_pca:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced = pca.fit_transform(df_matrix)
            cols = [f"pca_{i}" for i in range(n_components)]
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            reduced = svd.fit_transform(df_matrix)
            cols = [f"svd_{i}" for i in range(n_components)]

        # Check how many components were actually returned
        n_actual = reduced.shape[1]
        if n_actual < n_components:
            warnings.warn(
                f"Requested {n_components} components but only {n_actual} "
                "were returned (<= number of samples).", RuntimeWarning
            )

        # Return the final DataFrame with the embeddings
        df_embed = pd.DataFrame(reduced, index=nodes, columns=cols)
        df_embed.index.name = "node"
        df_embed.columns.name = "component"

        return df_embed

    def time_averaged_embeddings(self,
                                 weight_fn: Optional[Callable[[int], float]] = None) -> pd.DataFrame:
        """
        Compute the time-averaged DTQW probability distributions as node embeddings,
        weighted by the provided weight function or uniformly if none is provided.

        For each node, this method averages its probability distribution over time,
        with optional weighting based on the time step. The result is a matrix where
        each entry represents the weighted average probability of one node being in
        another node's state across time.

        Parameters
        ----------
        weight_fn: Callable[[int], float], optional
            A function that assigns a non-negative weight to each time step (t).
            If None, uniform averaging is applied.

        Returns
        -------
        pandas.DataFrame
            A (N, N) DataFrame where:
            - Rows    = start nodes (sorted lexicographically)
            - Columns = target nodes (sorted lexicographically)
            - Entry (v, u) = weighted average probability at node u from node v
        """
        nodes = self.sorted_nodes
        T = len(next(iter(self.histories.values())))  # Get the number of time steps

        # If no weight function is provided, use uniform weights
        if weight_fn is None:
            weights = np.ones(T)
        else:
            # Generate weights using the provided function
            weights = np.fromiter((weight_fn(t) for t in range(T)), dtype=float)

            # Ensure all weights are non-negative
            if np.any(weights < 0):
                raise ValueError("Weight function must return non-negative values.")

        # Normalize weights
        if weights.sum() == 0:
            raise ValueError("Sum of weights is zero. Adjust weight_fn or provide a valid weight function.")
        weights /= weights.sum()

        # Initialize dictionary to store averaged probabilities
        averaged = {}

        # Iterate over nodes and compute the weighted average probability distributions
        iterator = tqdm(nodes, desc="Averaging time-series") if self.show_progress else nodes
        for v in iterator:
            df_probs = self.extract_node_probabilities(self.histories[v])

            # Compute the weighted average across time steps
            averaged[v] = (df_probs[nodes].values * weights[:, None]).sum(axis=0)

        # Convert the dictionary to a DataFrame
        df_embed = pd.DataFrame.from_dict(averaged, orient="index", columns=nodes)
        df_embed.index.name = "start_node"
        df_embed.columns.name = "target_node"

        return df_embed

    def kernel_embeddings(self,
                          n_components: int = 64,
                          metric: str = "bhattacharyya") -> pd.DataFrame:
        """
        Apply Kernel PCA to the node similarity matrix computed from DTQW dynamics.

        This method calculates the similarity between nodes based on their quantum walk
        evolution over time using the specified similarity metric. The resulting kernel matrix
        is used in Kernel PCA to produce lower-dimensional embeddings of the nodes.

        Parameters
        ----------
        n_components: int
            The number of principal components to keep after applying Kernel PCA.

        metric: str, optional, default="bhattacharyya"
            The similarity measure to use for computing the kernel matrix. Supported options:
            - "bhattacharyya"
            - "cosine"
            - "hellinger"
            - "euclidean"

        Returns
        -------
        pandas.DataFrame
            A DataFrame of size (N, n_components) where:
            - Rows    = nodes (sorted lexicographically)
            - Columns = "kpca_0", "kpca_1", ..., "kpca_{n_components-1}"

        Raises
        ------
        ValueError
            If `n_components` is less than 1 or if an unsupported similarity metric is provided.
        """
        if n_components < 1:
            raise ValueError("n_components must be positive.")

        if metric not in {"bhattacharyya", "cosine", "hellinger", "euclidean"}:
            raise ValueError(f"Unsupported metric: '{metric}'. Supported metrics are "
                             "'bhattacharyya', 'cosine', 'hellinger', and 'euclidean'.")

        nodes = self.sorted_nodes
        N = len(nodes)
        T = len(next(iter(self.histories.values())))  # Time steps

        # Create the (N, T, N) tensor of node probabilities over time
        P = np.stack([
            self.extract_node_probabilities(self.histories[v])[nodes].values
            for v in nodes
        ])

        # Initialize the kernel matrix (N x N)
        K = np.zeros((N, N))

        # Compute the kernel matrix
        iterator = tqdm(range(N), desc="Computing kernel") if self.show_progress else range(N)
        for i in iterator:
            for j in range(i, N):
                sim = 0.0
                for t in range(T):
                    p_i, p_j = P[i, t], P[j, t]

                    # Compute similarity based on the chosen metric
                    if metric == "bhattacharyya":
                        sim += np.sum(np.sqrt(p_i * p_j))
                    elif metric == "hellinger":
                        sim += 1 - np.linalg.norm(np.sqrt(p_i) - np.sqrt(p_j)) / np.sqrt(2)
                    elif metric == "euclidean":
                        sim -= np.sum((p_i - p_j) ** 2)
                    elif metric == "cosine":
                        norm_i = np.linalg.norm(p_i)
                        norm_j = np.linalg.norm(p_j)
                        if norm_i > 0 and norm_j > 0:
                            sim += np.dot(p_i, p_j) / (norm_i * norm_j)

                K[i, j] = K[j, i] = sim  # Symmetric kernel matrix

        # Apply Kernel PCA to the precomputed kernel matrix
        kpca = KernelPCA(n_components=n_components,
                         kernel="precomputed",
                         random_state=self.random_state)
        reduced = kpca.fit_transform(K)

        # Check if the number of components is less than requested
        n_actual = reduced.shape[1]
        if n_actual < n_components:
            warnings.warn(
                f"Requested {n_components} KPCA components but only {n_actual} "
                "were returned (<= number of samples).", RuntimeWarning
            )

        # Create column names for the KPCA components
        cols = [f"kpca_{i}" for i in range(n_actual)]

        # Create a DataFrame for the embeddings
        df_embed = pd.DataFrame(reduced, index=nodes, columns=cols)
        df_embed.index.name = "node"
        df_embed.columns.name = "component"

        return df_embed
