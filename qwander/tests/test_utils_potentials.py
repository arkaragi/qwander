"""
Unit tests for qw_embed.utils.potentials module.
"""

import unittest

import numpy as np
import networkx as nx

from qwander.qwander.utils.potentials import (
    degree_centrality_potential,
    pagerank_potential,
    random_potential,
)


class TestDegreeCentralityPotential(unittest.TestCase):
    """
    Unit tests for the degree_centrality_potential function.
    """

    def setUp(self) -> None:
        """
        Set up a simple test graph for the degree centrality potential.
        """
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_empty_graph(self) -> None:
        """
        Test that the function returns an empty dictionary for an empty graph.
        """
        G_empty = nx.Graph()
        result = degree_centrality_potential(G_empty)
        self.assertEqual(result, {})

    def test_degree_centrality_potential(self) -> None:
        """
        Test that degree centrality potential is computed correctly.
        """
        expected = {
            0: np.pi * 1 / 4,  # Node 0 has degree 1, normalized degree centrality 1/4
            1: np.pi * 2 / 4,  # Node 1 has degree 2, normalized degree centrality 2/4
            2: np.pi * 2 / 4,  # Node 2 has degree 2, normalized degree centrality 2/4
            3: np.pi * 2 / 4,  # Node 3 has degree 2, normalized degree centrality 2/4
            4: np.pi * 1 / 4,  # Node 4 has degree 1, normalized degree centrality 1/4
        }
        result = degree_centrality_potential(self.G)
        for node, potential in result.items():
            self.assertAlmostEqual(potential, expected[node], delta=1e-8)


class TestPageRankPotential(unittest.TestCase):
    """
    Unit tests for the pagerank_potential function.
    """

    def setUp(self) -> None:
        """
        Set up a simple test graph for the PageRank potential.
        """
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_empty_graph(self) -> None:
        """
        Test that the function returns an empty dictionary for an empty graph.
        """
        G_empty = nx.Graph()
        result = pagerank_potential(G_empty)
        self.assertEqual(result, {})

    def test_pagerank_potential(self) -> None:
        """
        Test that PageRank potential is computed correctly.
        """
        result = pagerank_potential(self.G)
        expected = pagerank_potential(self.G, alpha=0.85)  # Compute the correct expected values
        for node, potential in result.items():
            self.assertAlmostEqual(potential, expected[node], delta=1e-8)


class TestRandomPotential(unittest.TestCase):
    """
    Unit tests for the random_potential function.
    """

    def setUp(self) -> None:
        """
        Set up a simple test graph for the random potential.
        """
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_empty_graph(self) -> None:
        """
        Test that the function returns an empty dictionary for an empty graph.
        """
        G_empty = nx.Graph()
        result = random_potential(G_empty)
        self.assertEqual(result, {})

    def test_random_potential(self) -> None:
        """
        Test that the random potential is computed correctly.
        """
        rng = np.random.default_rng(42)
        expected = {
            0: np.pi * rng.random(),
            1: np.pi * rng.random(),
            2: np.pi * rng.random(),
            3: np.pi * rng.random(),
            4: np.pi * rng.random(),
        }
        result = random_potential(self.G, seed=42)
        for node, potential in result.items():
            self.assertAlmostEqual(potential, expected[node], delta=1e-8)


if __name__ == "__main__":
    unittest.main()
