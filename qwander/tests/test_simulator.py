"""
Unit tests for qw_embed.utils.simulator module.
"""

import pathlib
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from qwander.qwander.simulators.dtqw_simulator import DTQWSimulator


class TestDTQWSimulator(unittest.TestCase):
    """
    Unit-tests that verify constructor-level validation logic in DTQWSimulator.
    """

    def test_invalid_graph_type(self):
        """
        Directed graphs are rejected: DTQWSimulator expects an undirected nx.Graph.
        """
        with self.assertRaises(ValueError):
            DTQWSimulator(nx.DiGraph([(0, 1)]))

    def test_multigraph_rejection(self):
        """
        Graphs with parallel edges raise, because the walk is defined only on simple graphs.
        """
        G = nx.MultiGraph([(0, 1), (0, 1)])
        with self.assertRaises(ValueError):
            DTQWSimulator(G)

    def test_self_loop_rejection(self):
        """
        Self-loops are disallowed; they break the coin/shift definition.
        """
        G = nx.Graph()
        G.add_edge(0, 0)
        with self.assertRaises(ValueError):
            DTQWSimulator(G)

    def test_isolate_rejection(self):
        """
        Graphs with isolated nodes are invalid for a coined walk.
        """
        G = nx.Graph()
        G.add_node(0)
        with self.assertRaises(ValueError):
            DTQWSimulator(G)

    def test_unknown_coin_type(self):
        """
        Requesting an unsupported coin keyword triggers a ValueError.
        """
        with self.assertRaises(ValueError):
            DTQWSimulator(nx.path_graph(3), coin_type="foo")

    def test_shift_map_symmetry(self):
        """
        Shift map must be an involution: S(S(h)) == h for every half-edge.
        """
        sim = DTQWSimulator(nx.cycle_graph(6))
        for k, v in sim.shift_map.items():
            self.assertEqual(sim.shift_map[v], k)

    def test_random_potential_keeps_total_prob(self):
        """
        Applying any potential (e.g. 'random') must conserve norm.
        """
        sim = DTQWSimulator(nx.path_graph(3), random_state=123)
        psi = sim.initialize_state()
        psi2 = sim._apply_potential(psi, "random")
        p1 = sum(np.vdot(v, v).real for v in psi.values())
        p2 = sum(np.vdot(v, v).real for v in psi2.values())
        self.assertAlmostEqual(p1, p2, places=12)

    def test_save_and_load_history_roundtrip(self):
        """
        History saved to disk should load back with identical length.
        """
        sim = DTQWSimulator(nx.path_graph(3))
        sim.initialize_state()
        sim.run(steps=2)
        tmp = pathlib.Path("./_tmp_hist.pkl")
        sim.save_history(tmp)

        sim2 = DTQWSimulator(nx.path_graph(3))
        sim2.load_history(tmp)
        self.assertEqual(len(sim2.history), len(sim.history))

        tmp.unlink(missing_ok=True)

    def test_uniform_state_probability_sum(self):
        """
        Uniform initialisation: total probability should sum to 1.
        """
        sim = DTQWSimulator(nx.cycle_graph(5))
        psi0 = sim.initialize_state()
        total_prob = sum(np.vdot(vec, vec).real for vec in psi0.values())
        self.assertAlmostEqual(total_prob, 1.0, places=12)

    def test_localised_state_only_one_node_nonzero(self):
        """
        Localised state: only the chosen start_node has non-zero amplitude.
        """
        sim = DTQWSimulator(nx.path_graph(4))
        psi0 = sim.initialize_state(start_node=2)
        non_zero = [v for v, vec in psi0.items() if np.any(vec)]
        self.assertEqual(non_zero, [2])

    def test_run_probability_conservation(self):
        """
        Total probability stays 1 after many steps of the walk.
        """
        sim = DTQWSimulator(nx.cycle_graph(4))
        sim.initialize_state()
        history = sim.run(steps=30)
        final_prob = sum(np.vdot(vec, vec).real for vec in history[-1].values())
        self.assertAlmostEqual(final_prob, 1.0, places=12)

    def test_run_negative_steps_error(self):
        """
        run() should reject negative step counts.
        """
        sim = DTQWSimulator(nx.path_graph(3))
        sim.initialize_state()
        with self.assertRaises(ValueError):
            sim.run(-5)

    def test_coin_phase_distribution_normalisation(self):
        """
        extract_coin_phase_distributions returns a valid probability distribution at each step.
        """
        sim = DTQWSimulator(nx.path_graph(4))
        sim.initialize_state()
        hist = sim.run(steps=2)
        df = sim.extract_coin_phase_distributions(hist)

        self.assertEqual(df.shape[0], 3)  # steps 0, 1, 2
        self.assertIsInstance(df.columns, pd.MultiIndex)
        self.assertEqual(df.columns.names, ["Node", "Phase"])
        self.assertTrue((df.sum(axis=1) - 1.0).abs().max() < 1e-12)

    def test_extract_node_probabilities_shape(self):
        """
        extract_probabilities returns (T+1)×|V| DataFrame whose rows sum to 1.
        """
        sim = DTQWSimulator(nx.path_graph(3))
        sim.initialize_state()
        hist = sim.run(steps=2)
        df = sim.extract_node_probabilities(hist)
        self.assertEqual(df.shape, (3, 3))          # steps 0,1,2 × 3 nodes
        self.assertTrue((df.sum(axis=1) - 1.0).abs().max() < 1e-12)

    def test_state_amplitudes_dataframe_structure(self):
        """
        extract_state_amplitudes returns a MultiIndex DataFrame with correct shape and labels.
        """
        sim = DTQWSimulator(nx.cycle_graph(4))
        sim.initialize_state()
        hist = sim.run(steps=3)
        df = sim.extract_state_amplitudes(hist)

        self.assertEqual(df.shape[0], 4)  # steps 0, 1, 2, 3
        self.assertIsInstance(df.columns, pd.MultiIndex)
        self.assertEqual(df.columns.names, ["Node", "Slot"])


if __name__ == "__main__":
    unittest.main()
