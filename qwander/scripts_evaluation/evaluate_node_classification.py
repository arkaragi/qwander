#!/usr/bin/env python3
"""
Node-classification evaluation on all Karate-Club or Cora experiments.

Set DATASET = "karate" or "cora" to process only the requested experiments.

Folder layout assumed (per split):

splits/{karate_testXX,cora_testXX_<classes>}/
├─ classical/embeddings/      (only for Karate)
│   ├─ deepwalk_*.pkl
│   └─ node2vec_*.pkl
├─ walks_<coin>_<pot>_T<steps>/
│   └─ embeddings/…
├─ train.edgelist             (ignored for node classification)
└─ …

This script:
  • loads precomputed embeddings (classical & DTQW);
  • splits nodes for classification using TEST_RATIO;
  • trains & evaluates logistic-regression classifiers via
    `evaluate_models_and_plot_node_cls`;
  • aggregates metrics and saves CSV and confusion matrices.
Supports 'karate' and 'cora' experiments.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

import networkx as nx
import numpy as np
import pandas as pd

from qwander.qwander.embeddings.dtqw_embedder import DTQWEmbedder
from qwander.qwander.utils.logger import logger, setup_logging
from qwander.qwander.utils.eval_utils import (
    generate_all_embeddings_karate_lp,
    evaluate_models_and_plot_node_cls,
    plot_grouped_bars,
    timestep_from_name,
)

# Configure logging
setup_logging()
log = logger

# Config
CONFIG_PATH = Path(__file__).parent / "evaluation_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

DATASET = config["dataset"]  # 'karate' or 'cora'
SPLITS_ROOT = Path(config["paths"]["splits_root"])
RESULTS_ROOT = Path(config["paths"]["results_root"])

# Evaluation options
TEST_RATIO = config["evaluation_nodecls"]["test_ratio"]


def _all_split_dirs(root: Path, dataset: str) -> List[Path]:
    prefixes = [f"{dataset}_test"]
    return sorted([
        p for p in root.iterdir()
        if (p / "split_info.json").is_file() and
           any(p.name.startswith(pref) for pref in prefixes)
    ])


def _load_nodes_and_labels(split_dir: Path) -> tuple[list[str], np.ndarray]:
    """
    Load the sorted list of nodes and their integer labels for a given dataset and split.

    Parameters
    ----------
    split_dir: Path
        The path to the split directory.

    Returns
    -------
    tuple[list[str], np.ndarray]
        - node_list: sorted list of node IDs (as strings)
        - labels: 1D array of integers aligned with node_list
    """
    if DATASET == "karate":
        # Fixed Karate clubs
        G = nx.karate_club_graph()
        G = nx.relabel_nodes(G, lambda x: str(x))
        node_list = sorted(G.nodes(), key=lambda n: int(n))
        # Map 'Mr. Hi' → 0, 'Officer' → 1
        labels = np.array([0 if G.nodes[n]["club"] == "Mr. Hi" else 1
                           for n in node_list], dtype=int)
        return node_list, labels

    elif DATASET == "cora":
        # Nodes from this split's train.edgelist
        node_list = sorted(
            nx.read_edgelist(split_dir / "train.edgelist", nodetype=str).nodes(),
            key=lambda n: int(n)
        )
        # Read full cora.content to get label strings
        id_to_label: dict[str, str] = {}
        content_path = Path(config["paths"]["cora_root"]) / "cora.content"
        with content_path.open() as f:
            for line in f:
                pid, *_, lbl = line.strip().split()
                id_to_label[pid] = lbl
        # Build consistent integer mapping
        uniq_labels = sorted(set(id_to_label.values()))
        mapping = {lbl: i for i, lbl in enumerate(uniq_labels)}
        labels = np.array([mapping[id_to_label[n]] for n in node_list], dtype=int)
        return node_list, labels

    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")


def _eval_classical_baselines(split_dir: Path,
                              labels: np.ndarray,
                              test_ratio: float) -> Optional[pd.DataFrame]:
    """
    Evaluate classical baselines (DeepWalk, Node2Vec) for node classification.

    Parameters
    ----------
    split_dir: Path
        The path to the split directory.

    labels: np.ndarray
        The array of labels.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the results if embeddings exist, else None.
    """
    cls_dir = split_dir / "classical" / "embeddings"
    if not cls_dir.is_dir():
        log.warning("[%s] no classical embeddings – skipped", split_dir.name)
        return None

    def find_embedding_file(prefix):
        # Find the first .pkl file that matches the prefix
        matches = list(cls_dir.glob(f"{prefix}*.pkl"))
        return matches[0] if matches else None

    deepwalk_file = find_embedding_file("deepwalk")
    node2vec_file = find_embedding_file("node2vec")

    if not deepwalk_file and not node2vec_file:
        log.warning("[%s] no DeepWalk or Node2Vec embeddings found", split_dir.name)
        return None

    embs = {}
    if deepwalk_file:
        embs["DeepWalk"] = DTQWEmbedder.load_embeddings(deepwalk_file)
    if node2vec_file:
        embs["Node2Vec"] = DTQWEmbedder.load_embeddings(node2vec_file)

    title = f"{split_dir.name} — Classical  ({int(test_ratio * 100)}% hold-out)"
    out_dir = split_dir / "classical" / "node_cls"
    df = evaluate_models_and_plot_node_cls(
        embeddings_dict=embs,
        labels=labels,
        test_size=test_ratio,
        random_state=42,
        out_dir=out_dir,
        title=title,
    )
    df.insert(0, "experiment", "Classical")
    df.insert(1, "split_folder", split_dir.name)
    df.insert(2, "test_ratio", test_ratio)
    return df


def _eval_dtqw_experiment(split_dir: Path,
                          exp_dir: Path,
                          node_list: list[str],
                          labels: np.ndarray,
                          test_ratio: float) -> Optional[pd.DataFrame]:
    """Evaluate all DTQW-based embeddings in this split."""
    embed_dir = exp_dir / "embeddings"
    if not embed_dir.is_dir():
        return None

    embs = generate_all_embeddings_karate_lp(
        node_list=node_list,
        dtqw_embed_dir=embed_dir
    )

    title = f"{split_dir.name} — {exp_dir.name}  (hold-out = {int(test_ratio * 100)} %)"
    out_dir = exp_dir / "node_cls"
    df = evaluate_models_and_plot_node_cls(
        embeddings_dict=embs,
        labels=labels,
        test_size=test_ratio,
        random_state=42,
        out_dir=out_dir,
        title=title,
    )
    df.insert(0, "experiment", exp_dir.name)
    df.insert(1, "split_folder", split_dir.name)
    df.insert(2, "test_ratio", test_ratio)
    return df


def main() -> None:
    # Check if splits root directory exists
    if not SPLITS_ROOT.is_dir():
        log.error("Splits root directory not found: %s", SPLITS_ROOT)
        return

    all_records: list[pd.DataFrame] = []

    # Get all split directories for the selected dataset
    split_dirs = _all_split_dirs(SPLITS_ROOT, DATASET)
    if not split_dirs:
        log.error("No split folders found for dataset '%s' in %s", DATASET, SPLITS_ROOT)
        return

    # Iterate over each split directory and evaluate
    for split_dir in split_dirs:
        for ratio in TEST_RATIO:
            node_list, labels = _load_nodes_and_labels(split_dir)

            # Evaluate classical baselines (once per split)
            df_cls = _eval_classical_baselines(split_dir, labels, ratio)
            if df_cls is not None:
                all_records.append(df_cls)

            # Evaluate all DTQW experiments in the split
            for exp_dir in sorted(split_dir.glob("walks_*")):
                df_dtqw = _eval_dtqw_experiment(split_dir, exp_dir, node_list, labels, ratio)
                if df_dtqw is not None:
                    all_records.append(df_dtqw)

    # Check if any results were generated
    if not all_records:
        log.warning("No evaluations produced any metrics.")
        return

    # Consolidate all evaluation records into a master DataFrame
    df_master = (
        pd.concat(all_records)
        .reset_index()
        .rename(columns={"index": "embedding", "test_ratio": "test_split"})
        .sort_values(["split_folder", "experiment", "embedding"])
    )

    out_root = RESULTS_ROOT / f"{DATASET}_node_cls_results"
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "all_nodecls_results.csv"
    df_master.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Master CSV written → %s", csv_path)

    # summary plot for test_accuracy
    plot_grouped_bars(
        df_master=df_master,
        metric="test_accuracy",
        splits=sorted(df_master["test_split"].unique()),
        exp_order=sorted(df_master["experiment"].unique(), key=timestep_from_name),
        out_path=out_root / "nodecls_accuracy_summary.png",
    )
    log.info("Summary plot saved → %s", out_root)


if __name__ == "__main__":
    main()
