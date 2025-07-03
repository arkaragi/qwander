#!/usr/bin/env python3
"""
Link-prediction evaluation on all Karate-Club or Facebook experiments.

Set DATASET = "karate" or "facebook" to process only the requested experiments.

Folder layout assumed (per split):

splits/{karate_testXX,facebook_testXX,facebook_reducedXX}/
├─ train.edgelist
├─ test_pos.edgelist
├─ test_neg.edgelist
├─ classical/embeddings/
│   ├─ deepwalk_32.pkl
│   └─ node2vec_32.pkl
├─ walks_<coin>_<pot>_T40/
│   └─ embeddings/…
└─ …

This script:
  • evaluates *classical* baselines once per split;
  • evaluates every DTQW experiment in that split;
  • aggregates metrics and produces summary plots.
Supports both 'karate' and 'facebook' experiments.
"""

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from qwander.qwander.embeddings.dtqw_embedder import DTQWEmbedder
from qwander.qwander.utils.logger import logger, setup_logging
from qwander.qwander.utils.eval_utils import (
    generate_all_embeddings_karate_lp,
    evaluate_models_and_plot_link_prediction,
    plot_grouped_bars,
    timestep_from_name,
)

# Configure root logger for simple timestamped console output
setup_logging()
log = logger

# You can allow CLI override, or just hardcode the config path
CONFIG_PATH = Path(__file__).parent / "evaluation_config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Paths and dataset info
DATASET = config["dataset"]
SPLITS_ROOT = Path(config["paths"]["splits_root"])
RESULTS_ROOT = Path(config["paths"]["results_root"])

# Evaluation options
SCORE = config["evaluation_linkpred"]["score"]


def _all_split_dirs(root: Path,
                    dataset: str) -> list[Path]:
    """
    Return all split folders for the selected dataset (Karate or Facebook).

    Parameters
    ----------
    root: Path
        Root directory containing the splits.

    dataset: str
        Dataset name ("karate" or "facebook").

    Returns
    -------
    List of split directories that match the dataset pattern.
    """
    valid_prefixes = [f"{dataset}_test", f"{dataset}_reduced"]
    return sorted([
        p for p in root.iterdir()
        if (p / "split_info.json").is_file() and
           any(p.name.startswith(prefix) for prefix in valid_prefixes)
    ])


def _test_ratio(split_dir: Path) -> float:
    """
    Extracts the test ratio from the split's metadata.

    Parameters
    ----------
    split_dir: Path
        The path to the split directory.

    Returns
    -------
    float
        The test ratio for the current split.
    """
    with (split_dir / "split_info.json").open() as fh:
        return float(json.load(fh)["test_ratio"])


def _load_nodes_for_dataset(dataset: str,
                            split_dir: Path) -> list[str]:
    """
    Load the sorted list of nodes for a given dataset and split.

    Parameters
    ----------
    dataset: str
        The dataset name ("karate" or "facebook").

    split_dir: Path
        The path to the split directory.

    Returns
    -------
    list[str]
        A sorted list of node identifiers.
    """
    if dataset == "karate":
        G = nx.karate_club_graph()
        G = nx.relabel_nodes(G, lambda x: str(x))
        return sorted(G.nodes(), key=int)
    elif dataset == "facebook":
        train = nx.read_edgelist(split_dir / "train.edgelist", nodetype=str)
        return sorted(train.nodes(), key=str)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def _eval_classical_baselines(split_dir: Path,
                              node_list: list[str],
                              ratio: float) -> Optional[pd.DataFrame]:
    """
    Evaluate classical baselines (DeepWalk, Node2Vec) for link prediction.

    Parameters
    ----------
    split_dir: Path
        The path to the split directory.

    node_list: list[str]
        The list of node identifiers.

    ratio: float
        The test ratio used for the split.

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

    title = f"{split_dir.name} — Classical baselines  (hold-out = {int(ratio * 100)} %)"
    out_dir = split_dir / "classical" / "link_pred"
    df = evaluate_models_and_plot_link_prediction(
        embeddings_dict=embs,
        nodes_list=node_list,
        split_dir=split_dir,
        out_dir=out_dir,
        score=SCORE,
        title=title,
    )
    df.insert(0, "experiment", "Classical")
    df.insert(1, "split_folder", split_dir.name)
    df.insert(2, "test_ratio", ratio)
    return df



def _eval_dtqw_experiment(split_dir: Path,
                          exp_dir: Path,
                          node_list: list[str],
                          ratio: float) -> Optional[pd.DataFrame]:
    """
    Evaluate DTQW-based embeddings for link prediction.

    Parameters
    ----------
    split_dir: Path
        The path to the split directory.

    exp_dir: Path
        The directory containing the DTQW experiment.

    node_list: list[str]
        The list of node identifiers.

    ratio: float
        The test ratio used for the split.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the results if embeddings exist, else None.
    """
    embed_dir = exp_dir / "embeddings"
    if not embed_dir.is_dir():
        return None

    embs = generate_all_embeddings_karate_lp(
        node_list=node_list,
        dtqw_embed_dir=embed_dir
    )
    title = f"{split_dir.name} — {exp_dir.name}  (hold-out = {int(ratio * 100)} %)"
    out_dir = exp_dir / "link_pred"
    df = evaluate_models_and_plot_link_prediction(
        embeddings_dict=embs,
        nodes_list=node_list,
        split_dir=split_dir,
        out_dir=out_dir,
        score=SCORE,
        title=title,
    )
    df.insert(0, "experiment", exp_dir.name)
    df.insert(1, "split_folder", split_dir.name)
    df.insert(2, "test_ratio", ratio)
    return df


def main() -> None:
    """
    Main function to evaluate link prediction experiments for classical baselines
    (DeepWalk, Node2Vec) and DTQW-based embeddings for each split in the selected
    dataset.

    The function performs the following tasks:
    1. Validates that the splits root directory exists.
    2. Iterates through each split and evaluates classical and DTQW experiments.
    3. Consolidates the evaluation results into a master DataFrame.
    4. Saves the results as a CSV file.
    5. Generates a summary plot comparing the performance across experiments.
    """

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
        ratio = _test_ratio(split_dir)
        node_list = _load_nodes_for_dataset(DATASET, split_dir)

        # Evaluate classical baselines (once per split)
        df_cls = _eval_classical_baselines(split_dir, node_list, ratio)
        if df_cls is not None:
            all_records.append(df_cls)

        # Evaluate all DTQW experiments in the split
        for exp_dir in sorted(split_dir.glob("walks_*")):
            df_dtqw = _eval_dtqw_experiment(split_dir, exp_dir, node_list, ratio)
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

    # Save the consolidated results as a CSV file
    out_root = RESULTS_ROOT / f"{DATASET}_link_pred_results"
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"all_linkpred_results_{SCORE}.csv"
    df_master.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Master CSV written → %s", csv_path)

    # Generate and save a summary plot for AUC
    plot_grouped_bars(
        df_master=df_master,
        metric="auc",
        splits=sorted(df_master["test_split"].unique()),
        exp_order=sorted(df_master["experiment"].unique(), key=timestep_from_name),
        out_path=out_root / f"{DATASET}_linkpred_auc_per_experiment_{SCORE}.png",
    )
    log.info("Summary plot for %s saved in %s", DATASET, out_root.resolve())


if __name__ == "__main__":
    main()
