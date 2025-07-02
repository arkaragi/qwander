"""
Reusable classes for node classification and link prediction
benchmarking, including common metrics and reporting tools.
"""

from typing import Dict, Optional, Iterable, Any, Sequence, Tuple, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def _dot(u: np.ndarray, v: np.ndarray) -> float:
    return float(u.dot(v))


def _neg_l2(u: np.ndarray, v: np.ndarray) -> float:
    return -float(np.linalg.norm(u - v))


def _neg_l1(u: np.ndarray, v: np.ndarray) -> float:
    return -float(np.linalg.norm(u - v, ord=1))


_SCORE_MAP = {
    "cosine": _cosine,
    "dot": _dot,
    "neg_l2": _neg_l2,
    "neg_l1": _neg_l1,
}


class NodeClassificationEvaluator:
    """
    Utility for evaluating node embeddings via supervised node classification.

    Parameters
    ----------
    test_size: float, default=0.3
        Fraction of samples to use for the test split (0 < test_size < 1).

    random_state: int or None, optional
        Random seed for reproducibility.

    classifier: sklearn estimator, optional
        Classifier with fit/predict interface. Defaults to LogisticRegression.

    Attributes
    ----------
    test_size: float
        Test set fraction.

    random_state: int or None
        Seed for splits and classifier.

    classifier: sklearn estimator
        The classifier used for evaluation.

    _results: dict or None
        Stores the results of the most recent evaluation.
    """

    def __init__(self,
                 test_size: float = 0.3,
                 random_state: Optional[int] = None,
                 classifier: Optional[Any] = None):
        self.test_size = test_size
        self.random_state = random_state
        self.classifier = classifier or LogisticRegression(max_iter=1000, random_state=random_state)
        self._results = None

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """
        Last computed results dictionary from evaluate(), if available.
        """
        return self._results

    def _validate_and_split(self,
                            embeddings: np.ndarray,
                            labels: np.ndarray):
        """
        Validate input shapes and perform train/test split.

        Parameters
        ----------
        embeddings: np.ndarray, shape (n_samples, n_features)
            Node embeddings.

        labels: np.ndarray, shape (n_samples,)
            Node class labels.

        Returns
        -------
        X_train, X_test, y_train, y_test: tuple of arrays
            Data split into train/test sets.

        Raises
        ------
        ValueError
            On shape or size mismatches.
        """
        X = np.asarray(embeddings)
        y = np.asarray(labels)

        # Sanity checks on input
        if X.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Labels must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Embeddings and labels must align: {X.shape[0]} vs {y.shape[0]}")
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size must be in (0,1), got {self.test_size}")

        # Stratify split if more than one class
        stratify = y if len(np.unique(y)) > 1 else None

        # Train/test split with optional stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=stratify,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test

    def _train_classifier(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray):
        """
        Fit the classifier to training data.

        Parameters
        ----------
        X_train: np.ndarray
            Training features.

        y_train: np.ndarray
            Training labels.

        Returns
        -------
        clf: sklearn estimator
            The fitted classifier.
        """
        clf = self.classifier
        clf.fit(X_train, y_train)
        return clf

    def _predict(self,
                 clf,
                 X_train,
                 X_test):
        """
        Predict class labels for train and test data.

        Parameters
        ----------
        clf: sklearn estimator
            The fitted classifier.

        X_train: np.ndarray
            Training features.

        X_test: np.ndarray
            Test features.

        Returns
        -------
        y_pred_train, y_pred_test: np.ndarray
            Predicted class labels for train and test sets.
        """
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        return y_pred_train, y_pred_test

    def _compute_metrics(self,
                         y_train,
                         y_pred_train,
                         y_test,
                         y_pred_test):
        """
        Compute a variety of classification metrics.

        Parameters
        ----------
        y_train: np.ndarray
            True training labels.

        y_pred_train: np.ndarray
            Predicted training labels.

        y_test: np.ndarray
            True test labels.

        y_pred_test: np.ndarray
            Predicted test labels.

        Returns
        -------
        metrics: dict
            Various accuracy, precision, recall, f1, macro,
            and confusion matrix metrics.
        """
        # Training and test accuracy
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Weighted metrics (accounts for class imbalance)
        precision = precision_score(
            y_test, y_pred_test, average="weighted", zero_division=0)
        recall = recall_score(
            y_test, y_pred_test, average="weighted", zero_division=0)
        f1 = f1_score(
            y_test, y_pred_test, average="weighted", zero_division=0)

        # Macro metrics (simple mean across classes)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average="macro", zero_division=0
        )

        # Full classification report and confusion matrices
        report = classification_report(y_test, y_pred_test, zero_division=0)
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)

        return {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "classification_report": report,
            "confusion_matrix_train": cm_train,
            "confusion_matrix_test": cm_test,
        }

    def evaluate(self,
                 embeddings: np.ndarray,
                 labels: np.ndarray) -> Dict[str, Any]:
        """
        Run node classification pipeline and compute metrics.

        Parameters
        ----------
        embeddings: np.ndarray
            Node embeddings, shape (n_samples, n_features).

        labels: np.ndarray
            Ground-truth node labels, shape (n_samples,).

        Returns
        -------
        results: dict
            Dictionary of performance metrics and confusion matrices.
        """
        # 1. Validate input, split train/test
        X_train, X_test, y_train, y_test = (
            self._validate_and_split(embeddings, labels))

        # 2. Train classifier
        clf = self._train_classifier(X_train, y_train)

        # 3. Predict labels for train/test sets
        y_pred_train, y_pred_test = self._predict(clf, X_train, X_test)

        # 4. Compute metrics
        results = self._compute_metrics(y_train, y_pred_train, y_test, y_pred_test)
        self._results = results

        return results


class LinkPredictionEvaluator:
    """
    Utility for evaluating node embeddings in link prediction tasks.

    Parameters
    ----------
    score: str, default="cosine"
        Node pair similarity function.
        Supported methods are: ("cosine", "dot", "neg_l2", "neg_l1").

    Attributes
    ----------
    score_fn: callable
        Function mapping two vectors to a similarity/distance score.

    _results: dict or None
        Stores the results of the last evaluation.
    """

    def __init__(self,
                 score: str = "cosine"):
        # Select similarity/distance scoring function
        if isinstance(score, str):
            try:
                self.score_fn = _SCORE_MAP[score.lower()]
            except KeyError:
                raise ValueError(f"Unknown score='{score}'. Valid: {list(_SCORE_MAP)}")
        else:
            raise TypeError("`score` must be a string key.")
        self._results = None

    @property
    def results(self) -> Optional[Dict[str, float]]:
        """
        Last computed results dictionary from evaluate(), if available.
        """
        return self._results

    @staticmethod
    def _check_inputs(embeddings: np.ndarray,
                      nodes_list: Sequence[str],
                      test_pos: Iterable[Tuple[str, str]],
                      test_neg: Iterable[Tuple[str, str]]) -> tuple:
        """
        Validate and preprocess input arrays and edge lists.

        Returns
        -------
        X: np.ndarray
            Node embedding matrix.

        nodes_list: list[str]
            Node labels (order matches embeddings).

        test_pos: list[Tuple[str, str]]
            Positive edge tuples.

        test_neg: list[Tuple[str, str]]
            Negative edge tuples.

        Raises
        ------
        ValueError
            On input shape/size/format errors.
        """
        X = np.asarray(embeddings)
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {X.shape}")
        if len(nodes_list) != X.shape[0]:
            raise ValueError("nodes_list length does not match embedding rows")

        test_pos = list(test_pos)
        test_neg = list(test_neg)
        if not test_pos or not test_neg or len(test_pos) != len(test_neg):
            raise ValueError("test_pos/test_neg must be non-empty and equal sized")

        return X, nodes_list, test_pos, test_neg

    @staticmethod
    def _edges_to_indices(nodes_list: List[str],
                          test_pos: List[Tuple[str, str]],
                          test_neg: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert (u, v) edge pairs to indices using nodes_list.

        Returns
        -------
        pos_idx: np.ndarray
        neg_idx: np.ndarray

        Raises
        ------
        ValueError
            If an edge contains an unknown node.
        """
        idx_of = {node: i for i, node in enumerate(nodes_list)}

        def idx_pairs(edges: List[Tuple[str, str]]) -> np.ndarray:
            try:
                return np.asarray([[idx_of[u], idx_of[v]] for u, v in edges], dtype=int)
            except KeyError as err:
                raise ValueError(f"Unknown node in edge list: {err.args[0]!r}")

        pos_idx = idx_pairs(test_pos)
        neg_idx = idx_pairs(test_neg)

        return pos_idx, neg_idx

    def _compute_scores(self,
                        X: np.ndarray,
                        pos_idx: np.ndarray,
                        neg_idx: np.ndarray) -> np.ndarray:
        """
        Compute similarity/distance scores for all candidate edges.

        Returns
        -------
        y_scores: np.ndarray, shape (2K,)
            Scores for positive and negative edge pairs.
        """
        all_idx = np.vstack([pos_idx, neg_idx])
        vec_u = X[all_idx[:, 0]]
        vec_v = X[all_idx[:, 1]]

        # Vectorized score computation
        y_scores = np.fromiter(
            (self.score_fn(a, b) for a, b in zip(vec_u, vec_v, strict=True)),
            dtype=float,
            count=len(vec_u),
        )
        return y_scores

    def _compute_metrics(self,
                         y_scores: np.ndarray,
                         K: int) -> Dict[str, float]:
        """
        Compute standard link prediction metrics.

        Parameters
        ----------
        y_scores: np.ndarray
            Scores for positive/negative edges (pos then neg).
        K: int
            Number of positive (and negative) edges.

        Returns
        -------
        metrics: dict
            Dictionary of AUC, average_precision, max_f1, max_precision,
            max_recall, hits@K, best_threshold.
        """
        y_true = np.concatenate([np.ones(K, dtype=int), np.zeros(K, dtype=int)])

        # ROC AUC & AP
        auc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))

        return {
            "auc": auc,
            "average_precision": ap
        }

    def evaluate(self,
                 embeddings: np.ndarray,
                 nodes_list: Sequence[str],
                 test_pos: Iterable[Tuple[str, str]],
                 test_neg: Iterable[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate link prediction performance for held-out positive/negative links.

        Parameters
        ----------
        embeddings: np.ndarray, shape (n_nodes, dim)
            Node embeddings.

        nodes_list: Sequence[str]
            Node labels (order matches embeddings).

        test_pos: Iterable[Tuple[str, str]]
            List of positive edges (ground truth links).

        test_neg: Iterable[Tuple[str, str]]
            List of negative edges (non-links).

        Returns
        -------
        dict
            Dictionary of metrics: auc, average_precision, precision, recall,
            f1, hits@K.
        """
        # 1. Input validation
        X, nodes_list, test_pos, test_neg = (
            self._check_inputs(embeddings, nodes_list, test_pos, test_neg))

        # 2. Map edges to embedding indices
        pos_idx, neg_idx = (
            self._edges_to_indices(nodes_list, test_pos, test_neg))

        # 3. Score all edges
        y_scores = self._compute_scores(X, pos_idx, neg_idx)

        # 4. Compute evaluation metrics
        K = len(test_pos)
        results = self._compute_metrics(y_scores, K)
        self._results = results

        return results
