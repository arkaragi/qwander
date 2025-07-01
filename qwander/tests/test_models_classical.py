"""
Test suite for models/classical.py module.
"""

import shutil
import tempfile
import unittest

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from qwander.qwander.embeddings.classical import DeepWalk, Node2Vec


class TestDeepWalkModel(unittest.TestCase):
    def setUp(self):
        # Create a simple graph: path graph with 4 nodes: "0"-"1"-"2"-"3"
        G = nx.path_graph(4)
        self.graph = nx.relabel_nodes(G, lambda x: str(x))
        self.temp_dir = tempfile.mkdtemp()

        # Common hyperparameters for testing
        self.dimensions = 8
        self.walk_length = 5
        self.num_walks = 3
        self.window_size = 2
        self.epochs = 2
        self.workers = 1
        self.seed = 123

        self.model = DeepWalk(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_generate_walks_length_and_content(self):
        # Directly test _generate_walks
        walks = self.model._generate_walks()
        # Should generate num_walks * num_nodes walks
        expected_count = self.num_walks * self.graph.number_of_nodes()
        self.assertEqual(len(walks), expected_count)

        # Each walk should have length between 1 and walk_length, and only valid nodes
        valid_nodes = set(self.graph.nodes())
        for walk in walks:
            self.assertTrue(1 <= len(walk) <= self.walk_length)
            for node in walk:
                self.assertIn(node, valid_nodes)

    def test_fit_trains_model_and_embeddings(self):
        # Before fit, model.model should be None
        self.assertIsNone(self.model.model)

        # Fit the model
        self.model.fit()

        # After fit, model.model should be a Word2Vec instance
        self.assertIsInstance(self.model.model, Word2Vec)

        # Retrieve embeddings
        embeddings = self.model.get_embeddings()
        valid_nodes = set(self.graph.nodes())
        self.assertEqual(set(embeddings.keys()), valid_nodes)
        for vec in embeddings.values():
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (self.dimensions,))

    def test_reproducibility_given_fixed_seed(self):
        # Create model1 and generate its walks immediately
        model1 = DeepWalk(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed
        )
        walks1 = model1._generate_walks()

        # Create model2 and generate its walks immediately
        model2 = DeepWalk(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed
        )
        walks2 = model2._generate_walks()

        # Walks must match
        self.assertEqual(walks1, walks2)

        # Fit both models (each one resets RNG at construction)
        model1.fit()
        model2.fit()

        emb1 = model1.get_embeddings()
        emb2 = model2.get_embeddings()
        for node in self.graph.nodes():
            np.testing.assert_allclose(emb1[str(node)], emb2[str(node)], atol=1e-6)


class TestNode2VecModel(unittest.TestCase):
    def setUp(self):
        # Create a simple graph: cycle graph with 5 nodes: "0"-"1"-"2"-"3"-"4"-"0"
        G = nx.cycle_graph(5)
        self.graph = nx.relabel_nodes(G, lambda x: str(x))
        self.temp_dir = tempfile.mkdtemp()

        # Common hyperparameters for testing
        self.dimensions = 8
        self.walk_length = 6
        self.num_walks = 2
        self.window_size = 2
        self.epochs = 2
        self.workers = 1
        self.seed = 456
        self.p = 0.5
        self.q = 2.0

        self.model = Node2Vec(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            p=self.p,
            q=self.q
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_generate_walks_length_and_content(self):
        walks = self.model._generate_walks()
        expected_count = self.num_walks * self.graph.number_of_nodes()
        self.assertEqual(len(walks), expected_count)

        valid_nodes = set(self.graph.nodes())
        for walk in walks:
            self.assertTrue(1 <= len(walk) <= self.walk_length)
            for node in walk:
                self.assertIn(node, valid_nodes)

    def test_reproducibility_of_walks_given_fixed_seed(self):
        # Generate walks on model1
        model1 = Node2Vec(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            p=self.p,
            q=self.q
        )
        walks1 = model1._generate_walks()

        # Generate walks on model2
        model2 = Node2Vec(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            p=self.p,
            q=self.q
        )
        walks2 = model2._generate_walks()

        # Walk sequences must match exactly
        self.assertEqual(walks1, walks2)

    def test_fit_and_get_embeddings(self):
        self.model.fit()
        self.assertIsInstance(self.model.model, Word2Vec)

        embeddings = self.model.get_embeddings()
        valid_nodes = set(self.graph.nodes())
        self.assertEqual(set(embeddings.keys()), valid_nodes)
        for vec in embeddings.values():
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (self.dimensions,))

    def test_p_and_q_effects(self):
        base_walks = self.model._generate_walks()

        # Increase p (discourage return); walks should differ
        model_high_p = Node2Vec(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            p=10.0,
            q=self.q
        )
        walks_high_p = model_high_p._generate_walks()
        self.assertNotEqual(base_walks, walks_high_p)

        # Increase q (discourage exploration); walks should differ
        model_high_q = Node2Vec(
            graph=self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            p=self.p,
            q=10.0
        )
        walks_high_q = model_high_q._generate_walks()
        self.assertNotEqual(base_walks, walks_high_q)


if __name__ == "__main__":
    unittest.main()
