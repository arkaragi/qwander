"""
Test suite for models/base.py module.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from qwander.qwander.embeddings.base import GraphEmbeddingModel


class DummyEmbeddingModel(GraphEmbeddingModel):
    """
    Dummy subclass that implements _generate_walks with a fixed pattern:
    For each node, if it has a neighbor, return [node, first_neighbor];
    otherwise [node].
    """
    def _generate_walks(self) -> list:
        walks = []
        for node in self.graph.nodes():
            node_str = str(node)
            neighbors = list(self.graph.neighbors(node_str))
            if neighbors:
                walks.append([node_str, str(neighbors[0])])
            else:
                walks.append([node_str])
        return walks


class TestGraphEmbeddingModel(unittest.TestCase):
    def setUp(self):
        # Create a simple path graph with 3 nodes: "0" - "1" - "2"
        G = nx.path_graph(3)
        self.simple_graph = nx.relabel_nodes(G, lambda x: str(x))

        # Temp directory for file operations
        self.temp_dir = tempfile.mkdtemp()

        # Common hyperparameters
        self.dimensions = 10
        self.window_size = 2
        self.epochs = 2
        self.workers = 1
        self.seed = 42

        # Instantiate dummy model
        self.model = DummyEmbeddingModel(
            graph=self.simple_graph,
            dimensions=self.dimensions,
            walk_length=2,
            num_walks=1,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fit_and_get_embeddings(self):
        """
        After fit(), verify that:
        - walks list is nonempty
        - model.model is a Word2Vec instance
        - get_embeddings() returns correct keys and shapes
        - get_embedding(node) returns a vector of correct shape
        - get_embedding(invalid) raises KeyError
        """
        # Before fit:
        self.assertEqual(self.model.walks, [])
        self.assertIsNone(self.model.model)

        # Fit
        self.model.fit()

        # After fit:
        self.assertTrue(len(self.model.walks) > 0)
        self.assertIsInstance(self.model.model, Word2Vec)

        embeddings = self.model.get_embeddings()
        valid_nodes = {"0", "1", "2"}
        self.assertEqual(set(embeddings.keys()), valid_nodes)
        for vec in embeddings.values():
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (self.dimensions,))

        # get_embedding for valid node
        emb_1 = self.model.get_embedding("1")
        self.assertIsInstance(emb_1, np.ndarray)
        self.assertEqual(emb_1.shape, (self.dimensions,))

        # get_embedding for invalid node
        with self.assertRaises(KeyError):
            self.model.get_embedding("nonexistent")

    def test_get_embedding_before_fit(self):
        """
        Calling get_embedding before fit() should raise RuntimeError.
        """
        with self.assertRaises(RuntimeError):
            self.model.get_embedding("0")

    def test_get_embeddings_before_fit(self):
        """
        Calling get_embeddings before fit() should raise RuntimeError.
        """
        with self.assertRaises(RuntimeError):
            self.model.get_embeddings()

    def test_save_and_load_embeddings(self):
        """
        After fitting:
        - save_embeddings writes a .kv file that exists
        - load_embeddings returns a KeyedVectors matching original vectors
        """
        self.model.fit()
        embeddings = self.model.get_embeddings()

        emb_path = Path(self.temp_dir) / "embeddings.kv"
        self.model.save_embeddings(emb_path)
        self.assertTrue(emb_path.exists())

        kv = self.model.load_embeddings(emb_path)
        self.assertIsInstance(kv, KeyedVectors)

        # Compare one node's vector
        orig = embeddings["1"]
        loaded = kv.get_vector("1")
        np.testing.assert_allclose(orig, loaded, atol=1e-6)

    def test_save_embeddings_before_fit(self):
        """
        Calling save_embeddings before fit() should raise RuntimeError.
        """
        emb_path = Path(self.temp_dir) / "dummy.kv"
        with self.assertRaises(RuntimeError):
            self.model.save_embeddings(emb_path)

    def test_load_embeddings_file_not_found(self):
        """
        Loading embeddings from a nonexistent path should raise FileNotFoundError.
        """
        fake = Path(self.temp_dir) / "no_file.kv"
        with self.assertRaises(FileNotFoundError):
            self.model.load_embeddings(fake)

    def test_save_and_load_model(self):
        """
        After fitting:
        - save_model writes a .model file
        - load_model recovers the Word2Vec model so get_embedding works
        """
        self.model.fit()
        model_path = Path(self.temp_dir) / "test_model.model"
        self.model.save_model(model_path)
        self.assertTrue(model_path.exists())

        new_model = DummyEmbeddingModel(
            graph=self.simple_graph,
            dimensions=self.dimensions,
            walk_length=2,
            num_walks=1,
            window_size=self.window_size,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed
        )
        self.assertIsNone(new_model.model)
        new_model.load_model(model_path)
        self.assertIsInstance(new_model.model, Word2Vec)

        orig = self.model.get_embedding("2")
        loaded = new_model.get_embedding("2")
        np.testing.assert_allclose(orig, loaded, atol=1e-6)

    def test_save_model_before_fit(self):
        """
        Calling save_model before fit() should raise RuntimeError.
        """
        model_path = Path(self.temp_dir) / "dummy.model"
        with self.assertRaises(RuntimeError):
            self.model.save_model(model_path)

    def test_load_model_file_not_found(self):
        """
        Loading a model from a nonexistent path should raise FileNotFoundError.
        """
        fake = Path(self.temp_dir) / "no_model.model"
        with self.assertRaises(FileNotFoundError):
            self.model.load_model(fake)

    def test_export_walks(self):
        """
        After fitting:
        - export_walks writes a .txt file
        - file contents match model.walks
        """
        self.model.fit()
        walks_path = Path(self.temp_dir) / "walks.txt"
        self.model.export_walks(walks_path)
        self.assertTrue(walks_path.exists())

        with open(walks_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f.readlines()]
        expected = self.model.walks
        self.assertEqual(len(lines), len(expected))
        for line_tokens, walk in zip(lines, expected):
            self.assertListEqual(line_tokens, walk)

    def test_export_walks_before_fit(self):
        """
        Calling export_walks before fit() should raise RuntimeError.
        """
        walks_path = Path(self.temp_dir) / "dummy_walks.txt"
        with self.assertRaises(RuntimeError):
            self.model.export_walks(walks_path)


if __name__ == "__main__":
    unittest.main()
