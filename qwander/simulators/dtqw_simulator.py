"""
Discrete-Time Quantum Walk (DTQW) Simulator for Undirected Graphs
==================================================================

This module provides a flexible and extensible simulator for coined
Discrete-Time Quantum Walks on simple, undirected graphs. It is designed
for reproducible and research-grade experimentation in quantum-inspired
machine learning, network analysis, and algorithmic physics.

Overview
--------
1. Create a DTQWSimulator instance with a valid NetworkX graph and coin type.
2. Initialize the walker's state (uniform or localized).
3. Run the quantum walk for a desired number of timesteps.
4. Optionally apply scalar potentials to bias the walk dynamics.
5. Extract classical observables for use in downstream tasks.
"""

import pathlib
import pickle
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from qwander.qwander.utils.coins import (
    hadamard_coin,
    grover_coin,
    fourier_coin,
    identity_coin,
    random_coin
)

from qwander.qwander.utils.potentials import (
    degree_centrality_potential,
    pagerank_potential,
    random_potential,
)

from qwander.qwander.utils.dtqw_visualizer import DTQWVisualizer

_POTENTIAL_DISPATCH = {
    "degree": degree_centrality_potential,
    "pagerank": pagerank_potential,
    "random": lambda graph, rng=None: random_potential(
        graph,
        seed=int(rng.integers(2 ** 32)) if rng is not None else None,
    )
}


class DTQWSimulatorBase:
    """
    Base class for quantum walk simulators on arbitrary undirected graphs.

    This class provides the foundation for simulating Discrete-Time Quantum Walks
    on undirected graphs. It handles validation, initializes the quantum walker's
    state, precomputes coin operators and shift maps, and stores the walk history.

    Attributes
    ----------
    G: networkx.Graph
        The input graph, which must be an undirected, simple graph without
        self-loops or isolated nodes.

    coin_type: str
        The type of coin operator to be used.
        Valid options include "grover", "hadamard", "fourier", "identity", and "random".

    random_state: Optional[int]
        The seed for the random number generator used in the simulation.
        If None, the RNG will be seeded randomly.

    rng: numpy.random.Generator
        The random number generator instance used throughout the simulation.

    N: int
        The total number of nodes in the graph (|V|).

    neighbors: dict[Any, list]
        A dictionary mapping each node to its sorted list of neighbors.
        The sorting ensures consistent ordering across simulations.

    degrees: dict[Any, int]
        A dictionary mapping each node to its degree.

    coins: dict[Any, np.ndarray]
        A dictionary mapping each node to its coin matrix, which is a unitary matrix
        representing the coin operation for that node's degree.

    shift_map: dict[tuple[Any, int], tuple[Any, int]]
        A dictionary mapping each half-edge (node, local_index) to its corresponding
        reverse half-edge (neighbor, reverse_index).

    psi0: Optional[dict[Any, np.ndarray]]
        The initial quantum state of the walk.
        This is a dictionary where the keys are nodes and the values are the
        coin vectors.

    history: Optional[list[dict[Any, np.ndarray]]]
        A list tracking the quantum walk's history.
        Each element is a dictionary representing the state of the walk after
        a given time step.
    """

    def __init__(self,
                 graph: nx.Graph,
                 coin_type: str = "grover",
                 random_state: Optional[int] = None):
        # Basic graph checks
        if not isinstance(graph, nx.Graph):
            raise ValueError("DTQWSimulator requires an undirected networkx.Graph.")

        if graph.is_multigraph():
            raise ValueError("Parallel edges detected; supply a simple graph (no MultiGraph).")

        if any(graph.has_edge(v, v) for v in graph):
            raise ValueError("Self-loops are not supported in this DTQW implementation.")

        iso = [v for v in graph if graph.degree[v] == 0]
        if iso:
            raise ValueError(f"Graph contains isolated node(s) {iso}; DTQW undefined.")

        # Work on an immutable copy so later graph mutations don’t break internal maps
        self.G = nx.freeze(graph.copy())
        self.coin_type = coin_type.lower()
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.N = self.G.number_of_nodes()

        # Build a deterministic, sorted list of neighbors for each node
        self.neighbors = {
            v: sorted(self.G.neighbors(v), key=lambda x: str(x))
            for v in self.G.nodes()
        }

        # Store the degree of each node as the length of its neighbor list
        self.degrees = {v: len(nbrs) for v, nbrs in self.neighbors.items()}

        # Precompute coin operators and shift map
        self.coins = self._initialize_coin_map()
        self.shift_map = self._initialize_shift_map()

        self.psi0: Optional[dict[Any, np.ndarray]] = None
        self.history: Optional[list[dict[Any, np.ndarray]]] = None

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__}: |V|={self.N}, "
                f"coin='{self.coin_type}', history_len={len(self.history or [])}>")

    def _initialize_coin_map(self) -> dict[Any, np.ndarray]:
        """
        Construct a per-vertex coin matrix C_v for the entire graph.

        For vertices with degree = 1, all coins collapse to the scalar matrix [[1.0]]
        because there is only one direction to walk.

        Returns
        -------
        dict
            Mapping {node → numpy.ndarray}, where each array is a unitary matrix
            of shape (deg(node), deg(node)) and dtype complex128.

        Raises
        ------
        ValueError
            Raised when coin_type is not one of the supported keywords above,
            or when coin_type is "hadamard" and the graph contains nodes with
            degree different from 2.
        """
        # Fast path: Hadamard needs a degree check, so handle it explicitly
        if self.coin_type == "hadamard":
            for v, d in self.degrees.items():
                if d != 2:
                    raise ValueError(
                        f"Hadamard coin only defined for degree 2 (node {v}, degree={d})")
            return {v: hadamard_coin() for v in self.G.nodes()}

        # Dispatch table for other coin families
        _COIN_DISPATCH = {
            "grover": grover_coin,
            "fourier": fourier_coin,
            "identity": identity_coin,
            "random": lambda deg: random_coin(deg, self.rng),
        }

        # Ensure the coin type is valid
        try:
            gen = _COIN_DISPATCH[self.coin_type]
        except KeyError:
            raise ValueError(
                f"Unsupported coin type '{self.coin_type}'. Valid options are: "
                f"{', '.join(sorted(_COIN_DISPATCH.keys() | {'hadamard'}))}."
            )

        # Construct the coins dict using the selected generator
        coins_map_dict = {v: gen(self.degrees[v]) for v in self.G.nodes()}

        return coins_map_dict

    def _initialize_shift_map(self) -> dict[tuple[Any, int], tuple[Any, int]]:
        """
        Precompute the shift map that routes coin-basis amplitudes.

        For every half-edge (v, i) where i indexes the i-th neighbor in the list
        self.neighbors[v], the map returns the unique counterpart (u, j) such as
        u = self.neighbors[v][i] and v = self.neighbors[u][j].

        Time complexity is O(|E|), where |E| is the number of edges in the graph.

        Returns
        -------
        dict[tuple[Any, int], tuple[Any, int]]
            A mapping from half-edge (node, local_index) to the corresponding
            reverse half-edge (neighbor, reverse_index).
        """
        shift_map_dict: dict[tuple[Any, int], tuple[Any, int]] = {}

        # Create a mapping for the position of each neighbor in the node's sorted neighbor list
        neighbor_index: dict[Any, dict[Any, int]] = {
            v: {u: idx for idx, u in enumerate(nbrs)}
            for v, nbrs in self.neighbors.items()
        }

        # Populate the shift map for every edge in the graph — overall O(|E|)
        for v, nbrs in self.neighbors.items():
            for i, u in enumerate(nbrs):
                j = neighbor_index[u][v]  # Get reverse neighbor's index
                shift_map_dict[(v, i)] = (u, j)

        return shift_map_dict

    def _apply_potential(self,
                         psi: dict[Any, np.ndarray],
                         potential: Union[dict[Any, float], str],
                         t: float = 1.0) -> dict[Any, np.ndarray]:
        """
        Apply a scalar potential to each node of the current quantum state
        before the coin operation. The potential acts as a phase factor
        exp(-i * V(v) * t) that multiplies the entire coin vector at node v.

        Parameters
        ----------
        psi: dict[Any, np.ndarray]
            Current quantum state. Keys are node labels, and the values are
            the corresponding coin vectors (one-dimensional numpy arrays).

        potential: dict or str
            Defines the scalar potential V(v).
            - dict:
                A mapping {node: float}. Nodes not in the dictionary default to V = 0.
            - str:
                One of the predefined potential types ("degree", "pagerank", "random", etc.)

        t: float, optional
            Time-scaling factor for the potential phase shift (default is 1.0).

        Returns
        -------
        dict[Any, np.ndarray]
            A new quantum state with the phase factor applied at each node.

        Raises
        ------
        ValueError
            If `potential` is neither a valid string nor a dictionary.
        """
        # Resolve the potential specification into a node → V(v) mapping
        if isinstance(potential, dict):
            # User-supplied mapping; nodes not present default to 0 later on
            v_map: dict[Any, float] = potential

        elif isinstance(potential, str):
            key = potential.lower()
            try:
                func = _POTENTIAL_DISPATCH[key]
            except KeyError:
                raise ValueError(
                    f"Unsupported potential '{potential}'.  "
                    f"Valid options are: {', '.join(sorted(_POTENTIAL_DISPATCH))}."
                ) from None

            # Apply the selected potential function
            v_map = func(self.G) if key != "random" else func(self.G, rng=self.rng)

        else:
            raise ValueError(
            "Argument 'potential' must be either a dict or one of the "
            "recognized keywords: 'degree', 'pagerank', and 'random'."
        )

        # Apply the phase shift for each node: ψ_v ← e^(−i V(v) t) · ψ_v
        new_psi: dict[Any, np.ndarray] = {}
        for v, coin_vec in psi.items():
            phase = np.exp(-1j * v_map.get(v, 0.0) * t)
            new_psi[v] = phase * coin_vec  # Apply the phase to the coin vector

        return new_psi

    def _step(self,
              psi: dict[Any, np.ndarray],
              potential: Optional[Union[dict[Any, float], str]] = None,
              t: float = 1.0) -> dict[Any, np.ndarray]:
        """
        Advance the walker by one discrete-time quantum-walk step.

        The step consists of an optional potential-induced phase, a local coin
        flip at every vertex, and a global shift of amplitudes along edges.

        Parameters
        ----------
        psi: dict[Any, np.ndarray]
            Current quantum state.
            Keys are node labels, and values are the corresponding coin vectors,
            where each vector's length is equal to the degree of the node.

        potential: dict, str, or None, optional
            Scalar potential specification applied before the coin operation.
            - dict: Custom mapping {node: float}.
            - str: Keyword selecting a predefined potential.
            - None: Skip potential application (default).

        t: float, optional
            Time-scaling factor for phase shift when a potential is applied.
            Defaults to 1.

        Returns
        -------
        dict[Any, np.ndarray]
            The updated quantum state after applying the coin flip and shift
            operation. Keys are node labels, and values are the new coin vectors.

        Notes
        -----
        The method first applies an optional potential-induced phase to each node's
        coin vector, then performs a coin flip using the corresponding coin operator
        for each node, and finally shifts the resulting amplitudes along the graph
        edges.
        """
        # Apply potential-induced phase (if specified)
        if potential is not None:
            psi = self._apply_potential(psi, potential, t)

        # Coin flip at each vertex:  ψ_v ← C_v · ψ_v
        psi_after_coin = {v: self.coins[v] @ vec for v, vec in psi.items()}

        # Initialize next_psi based on the degree of each node
        next_psi = {v: np.zeros_like(coin_vec, dtype=complex) for v, coin_vec in psi_after_coin.items()}

        # Shift operation: route amplitude to neighbors
        for v, coin_vec in psi_after_coin.items():
            for i, amp in enumerate(coin_vec):
                u, j = self.shift_map[(v, i)]  # Get the reverse half-edge (u, j)
                next_psi[u][j] += amp  # Update the amplitude at neighbor node u, slot j

        return next_psi

    def load_history(self,
                     path: str) -> None:
        """
        Load a walk history from a given path and attach it to the simulator.

        Parameters
        ----------
        path: str
            Filesystem location of the pickle file created by save_history.

        Raises
        ------
        FileNotFoundError
            If the file at the given path does not exist.
        ValueError
            If the loaded file does not contain a valid history or has an invalid format.
        """
        p = pathlib.Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"No history file found at {p}. Please check the path.")

        with p.open("rb") as f:
            data = pickle.load(f)

        # Check for potential metadata
        if isinstance(data, dict) and "history" in data:
            hist = data["history"]
        else:
            hist = data

        # Validate the loaded history format
        if not isinstance(hist, list) or not all(isinstance(s, dict) for s in hist):
            raise ValueError("Loaded file does not contain a valid list of history dictionaries.")

        self.history = hist

    def save_history(self,
                     path: str) -> None:
        """
        Save the current walk history to a given path in pickle format.

        Parameters
        ----------
        path: str
            Destination file. Parent directories are created if necessary.

        Raises
        ------
        ValueError
            If the simulator has no history to save.
        """
        if self.history is None:
            raise ValueError("No history to save; run the simulation first to generate history.")

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Include relevant metadata and the history
        payload = {
            "history": self.history,
            "coin_type": self.coin_type,
            "random_state": self.random_state,
            "rng_state": self.rng.bit_generator.state,
        }

        with p.open("wb") as f:
            pickle.dump(payload, f)

    def save_nodewise_histories(self,
                                histories: dict[Any, list[dict[Any, np.ndarray]]],
                                out_path: str) -> None:
        """
        Save all node-localized DTQW histories to a single pickle file.

        Parameters
        ----------
        histories: dict[Any, list[dict[Any, np.ndarray]]]
            Output from run_nodewise(), mapping each node to its walk history.

        out_path: str
            Destination directory where 'histories.pkl' will be saved.

        Raises
        ------
        ValueError
            If any history is malformed, empty, or improperly structured.
        """
        # Validate each history entry
        for node, hist in histories.items():
            if not isinstance(hist, list) or not all(isinstance(h, dict) for h in hist):
                raise ValueError(f"Malformed history for node {node}: "
                                 f"Must be a list of dictionaries.")

        out_dir = pathlib.Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Include relevant metadata and the nodewise history
        payload = {
            "nodewise_histories": histories,
            "coin_type": self.coin_type,
            "random_state": self.random_state,
            "rng_state": self.rng.bit_generator.state,
        }

        file_path = out_dir / "histories.pkl"
        with file_path.open("wb") as f:
            pickle.dump(payload, f)

    def extract_coin_phase_distributions(self,
                                         history: Optional[list[dict[Any, np.ndarray]]] = None) -> pd.DataFrame:
        """
        Return a DataFrame of coin–phase probability distributions per time step.

        For each time step t and each (node, phase_mode), computes
        |FFT(ψ_v)[phase_mode]|² normalized so all entries sum to 1.

        Parameters
        ----------
        history: list[dict[Any, np.ndarray]] or None, optional
            Walk history from run(). If None, uses self.history.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of node-level coin-phase distributions.
            - Index   = time steps (0, 1, …, T)
            - Columns = MultiIndex [(node, phase_mode), …], lexicographically sorted
            - Values  = coin–phase probabilities at each mode

        Raises
        ------
        ValueError
            If no valid history is available.
        """
        hist = history if history is not None else self.history
        if not isinstance(hist, list) or not hist:
            raise ValueError("History must be a non-empty list of state dictionaries.")

        # Determine sorted (node, phase) slots
        nodes = sorted(self.G.nodes(), key=lambda x: str(x))
        slots = [(v, k) for v in nodes for k in range(self.degrees[v])]
        mi = pd.MultiIndex.from_tuples(slots, names=("Node", "Phase"))

        rows = []
        for t, state in enumerate(hist):
            row = []
            for v, k in slots:
                vec = state[v]
                fft = np.fft.fft(vec)
                # Probability at phase mode k
                row.append(np.abs(fft[k]) ** 2)
            # Normalize entire row
            row = np.array(row)
            denom = row.sum()
            # Avoid division by zero
            row = row / denom if denom > 1e-15 else row
            rows.append(row)

        df = pd.DataFrame(rows, columns=mi)
        df.index.name = "Time Step"

        return df

    def extract_node_probabilities(self,
                                   history: Optional[list[dict[Any, np.ndarray]]] = None) -> pd.DataFrame:
        """
        Convert a DTQW history into a DataFrame of node probabilities per time step.

        Parameters
        ----------
        history: list[dict[Any, np.ndarray]] or None, optional
            Walk history from run(). If None, uses self.history.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of node-level probabilities.
            - Index   = time steps (0, 1, …, T)
            - Columns = node labels, lexicographically sorted
            - Values  = Σ_i |ψ_t[v][i]|²

        Raises
        ------
        ValueError
            If no valid history is available.
        """
        hist = self.history if history is None else history
        if not hist or not isinstance(hist, list):
            raise ValueError("History must be a non-empty list of quantum state dictionaries.")

        # Build probability snapshots
        prob_list = [
            {v: np.abs(vec).dot(np.abs(vec)) for v, vec in state.items()}
            for state in hist
        ]
        df = pd.DataFrame(prob_list).rename_axis(index="Time Step")

        # Sort columns lexicographically
        cols = sorted(df.columns, key=lambda x: str(x))

        return df[cols]

    def extract_state_amplitudes(self,
                                 history: Optional[list[dict[Any, np.ndarray]]] = None) -> pd.DataFrame:
        """
        Return a DataFrame of complex amplitudes for each (node, slot) per time step.

        Parameters
        ----------
        history: list[dict[Any, np.ndarray]] or None, optional
            Walk history from run(). If None, uses self.history.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of node-level quantum states.
            - Index   = time steps (0, 1, …, T)
            - Columns = MultiIndex [(node, slot), …], lexicographically sorted
            - Values  = complex amplitudes ψ_v[slot]

        Raises
        ------
        ValueError
            If no valid history is available.
        """
        hist = history if history is not None else self.history
        if not isinstance(hist, list) or not hist:
            raise ValueError("History must be a non-empty list of state dictionaries.")

        # Fixed (node, slot) ordering
        slots = sorted(self.shift_map.keys(), key=lambda x: (str(x[0]), x[1]))
        mi = pd.MultiIndex.from_tuples(slots, names=("Node", "Slot"))

        # Build rows of amplitudes
        rows = []
        for t, state in enumerate(hist):
            try:
                rows.append([state[node][slot] for node, slot in slots])
            except KeyError:
                raise ValueError(f"Malformed state at time step {t}; missing entries.")

        df = pd.DataFrame(rows, columns=mi)
        df.index.name = "Time Step"

        return df


class DTQWSimulator(DTQWSimulatorBase):
    """
    Discrete-Time Quantum Walk Simulator on Arbitrary Undirected Graphs.

    This class implements a flexible and reproducible simulator for coined
    DTQWs on arbitrary simple and undirected graphs.

    Features
    --------
    - Supports Grover, Hadamard, Fourier, Identity and Haar-random coin operators.
    - Optional graph-derived or custom potentials applied as phase shifts.
    - Complete history storage with save/load for experiment reproducibility.
    - Deterministic neighbor ordering ⇒ stable coin basis across runs.
    - Helper to convert amplitude history to a DataFrame of node probabilities.

    Parameters
    ----------
    graph: networkx.Graph
        Undirected graph with no isolated nodes.

    coin_type: str
        Case-insensitive keyword selecting the local coin operator.
        Supported operators:
            - "hadamard": 2×2 Hadamard coin (d = 2)
            - "grover": d×d Grover diffusion coin (d ≥ 1)
            - "fourier": d×d discrete Fourier coin (d ≥ 1)
            - "identity": d×d identity coin (d ≥ 1)
            - "random": d×d Haar-random unitary coin (d ≥ 1)

    random_state: int or None, optional
        Seed for the simulator’s internal random-number generator.

    Raises
    ------
    ValueError
        If the graph violates simplicity requirements or if coin_type is not
        supported.
    """

    def __init__(self,
                 graph: nx.Graph,
                 coin_type: str = "grover",
                 random_state: Optional[int] = None):
        super().__init__(graph, coin_type, random_state)

    def initialize_state(self,
                         start_node: Any = None) -> dict[Any, np.ndarray]:
        """
        Initialize the walker’s state.

        Mode 1 – uniform superposition
            If start_node is None (default), each coin slot on every node receives
            amplitude 1 / √(d_v · N), where d_v is the node’s degree, and N is
            the total number of vertices.

        Mode 2 – localized state
            If start_node is provided, that node’s d_start coin slots each receive
            amplitude 1 / √d_start. All other nodes are set to zero.

        Parameters
        ----------
        start_node: Any or None, optional
            Node at which to localize the walker. If None, uses uniform mode.

        Returns
        -------
        dict[Any, np.ndarray]
            Mapping node → complex amplitude vector (length = degree).

        Raises
        ------
        ValueError
            If start_node is not found in the graph.
        """
        # Verify start_node is valid (if provided)
        if start_node is not None and start_node not in self.G:
            raise ValueError(f"start_node {start_node} is not in the graph.")

        psi0: dict[Any, np.ndarray] = {}
        inv_sqrt_n = 1.0 / np.sqrt(self.N)

        for v, d in self.degrees.items():
            if start_node is None:
                # Uniform superposition
                psi0[v] = np.full(d, inv_sqrt_n / np.sqrt(d), dtype=complex)
            else:
                # Localized state at start_node
                psi0[v] = np.zeros(d, dtype=complex)
                if v == start_node:
                    psi0[v] = np.full(d, 1.0 / np.sqrt(d), dtype=complex)

        # Record and clear previous trajectory
        self.psi0 = psi0

        return psi0

    def reset(self) -> None:
        """
        Reset the simulator by clearing the recorded walk history.
        The next run() will begin from the current psi0.
        """
        self.history = None

    def run(self,
            steps: int,
            potential: Optional[Union[dict[Any, float], str]] = None,
            t: float = 1.0,
            reset: bool = False,
            show_progress: bool = False) -> list[dict[Any, np.ndarray]]:
        """
        Simulate the quantum walk for a fixed number of time steps and return
        the complete state history.

        Parameters
        ----------
        steps: int
            Non-negative number of steps to advance.

        potential: dict, str, or None, optional
            Scalar potential specification passed to _apply_potential.
            - dict: Custom mapping {node: float}.
            - str: Keyword selecting a predefined potential.
            - None: Skip phase application. (Default)

        t: float, optional
            Time-scaling factor used when a potential is applied (default 1.0).

        reset: bool, optional
            If True, clears any existing history and restarts from psi0.
            Default is False.

        show_progress: bool, optional
            If True, displays a progress bar during the walk.
            Default is False.

        Returns
        -------
        list[dict[Any, np.ndarray]]
            History of quantum states. Element 0 is the initial state; each
            subsequent element is the state after one additional step.
            Returns the cumulative walk history unless reset=True, in which
            case it starts fresh from psi0.

        Raises
        ------
        ValueError
            If steps is negative, or if psi0 is None on the first run.
        """
        if steps < 0:
            raise ValueError(f"Number of steps must be non-negative, got {steps}.")

        if reset:
            self.reset()

        if self.history is None:
            if self.psi0 is None:
                raise ValueError("Call initialize_state() before run().")
            # Defensive copy of the initial state
            psi = {v: vec.copy() for v, vec in self.psi0.items()}
            self.history = [psi]
        else:
            # Continue from the last recorded state
            psi = {v: vec.copy() for v, vec in self.history[-1].items()}

        # Use tqdm for progress bar if show_progress is True
        iterator = tqdm(range(steps), desc="Running DTQW", unit="step") \
            if show_progress else range(steps)

        for _ in iterator:
            # Apply step function to simulate the quantum walk
            psi = self._step(psi, potential, t)
            self.history.append(psi)

        return self.history

    def run_nodewise(self,
                     steps: int,
                     potential: Optional[Union[dict[Any, float], str]] = None,
                     t: float = 1.0,
                     show_progress: bool = False,
                     out_path: Optional[str] = None) -> None:
        """
        Run independent DTQWs from every node as the initial position.

        Parameters
        ----------
        steps: int
            Number of walk steps (≥ 0).

        potential: dict, str, or None
            Potential passed to the simulator.

        t: float
            Time-scaling factor used with the potential.

        show_progress: bool
            Whether to show a progress bar.

        out_path: str or None
            If provided, saves all histories to this pickle path.

        Raises
        ------
        ValueError
            If steps is negative.
        """
        # Ensure steps is non-negative
        if steps < 0:
            raise ValueError(f"Number of steps must be non-negative, got {steps}.")

        # Dictionary to store histories for each node
        histories = {}

        # Get the list of nodes in the graph
        nodes = list(self.G.nodes())

        # Setup progress bar if requested
        iterator = tqdm(nodes, desc="Running DTQW from all nodes") \
            if show_progress else nodes

        # Run the walk from each node as the initial position
        for v in iterator:
            # Initialize walker state from this node
            self.initialize_state(start_node=v)

            # Run the walk from this node and get its history
            traj = self.run(
                steps=steps,
                potential=potential,
                t=t,
                reset=True,
                show_progress=False
            )

            # Store the trajectory (history) for this node
            histories[v] = traj

        # Save the histories to disk if an output path is provided
        if out_path:
            self.save_nodewise_histories(histories, out_path)


def visualize_walk_results(G: nx.Graph,
                           df_no_pot: pd.DataFrame,
                           df_with_pot: pd.DataFrame,
                           start_node: Any,
                           cmap_prob: str = "viridis",
                           cmap_pot: str = "plasma",
                           layout: str = "spring",
                           seed: int = 42,
                           walk_desc: Optional[str] = None) -> None:
    """
    Given a graph and two probability‐DataFrames (with and without potential),
    render all the standard DTQW visualizations.

    Parameters
    ----------
    G: networkx.Graph
        The underlying graph.
    df_no_pot: pd.DataFrame
        Probabilities over time without any potential.
    df_with_pot: pd.DataFrame
        Probabilities over time with the chosen potential.
    start_node: Any
        Node label used as the walk’s starting position.
    cmap_prob: str, optional
        Colormap for probability plots.
    cmap_pot: str, optional
        Colormap for potential plots.
    layout: str, optional
        Graph‐drawing layout for visualization (default="spring").
    seed: int, optional
        Random seed for layout reproducibility (default=42).
    walk_desc: str, optional
        A short description of the walk (e.g. "Hadamard coin, T=40"); if provided,
        it will be prepended to each plot title.
    """

    from qwander.qwander.utils.dtqw_visualizer import DTQWVisualizer

    # Initialize visualizer with walk description
    visualizer = DTQWVisualizer(G, layout=layout, seed=seed, walk_desc=walk_desc)

    # 1) Final‐step probability (no potential)
    visualizer.plot_final_step(df_no_pot, cmap=cmap_prob)

    # 2) First 6 time‐steps side by side
    visualizer.plot_steps_subplots(df_no_pot, cmap=cmap_prob)

    # 3) Potential‐colored + final‐step probabilities (with potential)
    pot_map = nx.degree_centrality(G)
    visualizer.plot_potential_and_final(
        potential=pot_map,
        prob_df=df_with_pot,
        cmap_pot=cmap_pot,
        cmap_prob=cmap_prob,
        title_pot="Degree‐Centrality Potential"
    )

    # 4) Compare the two final distributions side by side
    visualizer.plot_compare_distributions(
        prob_df1=df_no_pot,
        prob_df2=df_with_pot,
        label1="No Potential",
        label2="With Potential",
        cmap=cmap_prob
    )

    # 5) Bar‐chart of final distribution, highlighting the start node
    visualizer.plot_final_histogram(
        prob_df=df_no_pot,
        start_node=start_node
    )

    # 6) Shannon entropy over time (no potential)
    visualizer.plot_entropy_over_time(df_no_pot, base=2.0)

    # 7) Probability time‐series for the start node
    visualizer.plot_node_time_series(df_no_pot, node=start_node)

    # 8) Animate walk
    # visualizer.animate_walk(df_no_pot)

def main_execution_visualization():
    # Build the graph and choose parameters
    G = nx.karate_club_graph()

    # Initialize the DTQW (uses Grover coin, localized at start_node)
    coin_type = "grover"
    steps = 40
    start_node = list(G.nodes())[0]
    walk_desc = f"{coin_type.capitalize()} Coin, Total Steps={steps}, Start Node={start_node}"
    dtqw = DTQWSimulator(G, coin_type=coin_type)

    # Initialize the state of the position register
    dtqw.initialize_state(start_node=start_node)

    # Run DTQW with an external potential (e.g., eigenvector‐centrality)
    history_with_pot = dtqw.run(
        steps=steps,
        potential="degree",
        t=1.0,
        show_progress=True
    )
    df_probs_with_pot = dtqw.extract_node_probabilities(history_with_pot)

    # Run DTQW without any potential
    history_no_pot = dtqw.run(steps=steps, reset=True)
    df_probs_no_pot = dtqw.extract_node_probabilities(history_no_pot)

    # Print out the final‐step distributions for comparison
    print("Final‐step distribution WITH potential:")
    print(df_probs_with_pot.iloc[-1])
    print("\nFinal‐step distribution WITHOUT potential:")
    print(df_probs_no_pot.iloc[-1])

    # Visualize the results
    visualize_walk_results(
        G,
        df_no_pot=df_probs_no_pot,
        df_with_pot=df_probs_with_pot,
        start_node=start_node,
        walk_desc=walk_desc)


def main_resume():
    # Build the graph and choose parameters
    G = nx.karate_club_graph()

    # Initialize visualizer
    visualizer = DTQWVisualizer(G, layout="spring", seed=42)

    # Initialize the DTQW (uses Grover coin, localized at start_node)
    coin_type = "grover"
    start_node = list(G.nodes())[0]
    sim = DTQWSimulator(G, coin_type=coin_type)
    sim2 = DTQWSimulator(G, coin_type=coin_type)

    # Initialize the state of the position register
    sim.initialize_state(start_node=start_node)

    # Run an initial walk of 10 steps
    history1 = sim.run(steps=10)
    df_probs = sim.extract_node_probabilities(history1)
    print(f"Initial run: recorded {len(history1) - 1} steps (history length = {len(history1)})")
    # Save that history to disk
    sim.save_history("karate_history.pkl")
    print("→ Saved initial history to 'karate_history.pkl'")

    # Plot Shannon Entropy over time (without potential)
    visualizer.plot_entropy_over_time(df_probs, base=2.0)

    # In a fresh simulator, load and resume
    sim2.load_history("karate_history.pkl")
    print(f"Loaded history of length {len(sim2.history)} into new simulator")

    # Extend by another 10 steps (no potential)
    history2 = sim2.run(steps=10)
    df_probs2 = sim.extract_node_probabilities(history2)
    print(f"After extend: recorded total {len(history2) - 1} steps (history length = {len(history2)})")
    # (Cleanup) Save the extended history
    sim2.save_history("karate_history2.pkl")
    print("→ Saved extended history to 'karate_history_extended.pkl'")

    # Plot Shannon Entropy over time (without potential)
    visualizer.plot_entropy_over_time(df_probs2, base=2.0)


if __name__ == "__main__":
    main_execution_visualization()
    # main_resume()
