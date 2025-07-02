#!/usr/bin/env python3
"""
Driver for extracting graph embeddings from DTQW simulation results
and classical baselines.

Typical structure:
- Loads all switches and hyperparameters from config.
- Processes all splits and simulation subfolders.
- Applies embedding extraction and dimensionality reduction as requested.
"""

import json
from pathlib import Path
from typing import Tuple

import networkx as nx

from qwander.qwander.embeddings.dtqw_embedder import DTQWEmbedder
from qwander.qwander.utils.logger import logger, setup_logging
from qwander.qwander.utils.runners import run_deepwalk, run_node2vec

# Configure root logger for simple timestamped console output
setup_logging()
log = logger

# You can allow CLI override, or just hardcode the config path
CONFIG_PATH = Path(__file__).parent / "config_link_prediction.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Paths and dataset info
DATASET = config["dataset"]
SPLITS_ROOT = Path(config["paths"]["splits_root"])
DTQW_SPLIT_GLOB = f"{DATASET}*"
WALK_GLOB = "walks*"

# DTQW-based embedding options
DTQW_CFG = config.get("dtqw_embedding", {})

DO_FINAL = DTQW_CFG.get("do_final", True)
DO_AVERAGE = DTQW_CFG.get("do_average", True)
DO_SVD = DTQW_CFG.get("do_svd", True)
DO_KERNEL_PCA = DTQW_CFG.get("do_kernel_pca", True)
SVD_DIM = DTQW_CFG.get("svd_dim", 32)
KPCA_DIM = DTQW_CFG.get("kpca_dim", 32)
KPCA_METRIC = DTQW_CFG.get("kpca_metric", "bhattacharyya")
RANDOM_STATE = DTQW_CFG.get("random_state", 42)

# Classical embedding options
CLAS_CFG = config.get("classical_embedding", {})

DO_DEEPWALK = CLAS_CFG.get("do_deepwalk", True)
DO_NODE2VEC = CLAS_CFG.get("do_node2vec", True)
DW_DIM = CLAS_CFG.get("dw_dim", 32)
N2V_DIM = CLAS_CFG.get("n2v_dim", 32)
WALK_LENGTH = CLAS_CFG.get("walk_length", 20)
NUM_WALKS = CLAS_CFG.get("num_walks", 100)
WINDOW_SIZE = CLAS_CFG.get("window_size", 5)
EPOCHS = CLAS_CFG.get("epochs", 5)
P_PARAM = CLAS_CFG.get("p_param", 1)
Q_PARAM = CLAS_CFG.get("q_param", 1)


def load_train_graph(split_dir: Path) -> Tuple[nx.Graph, nx.Graph]:
    """
    Loads the full reference graph and a training subgraph for a given split.

    Parameters
    ----------
    split_dir: Path
        Path to the split directory (e.g., "karate_test10").

    Returns
    -------
    full: nx.Graph
        The original full graph for reference/visualization.

    train: nx.Graph
        The training subgraph (with held-out test edges removed).
    """
    if split_dir.name.startswith("karate"):
        # Karate: built-in graph (NetworkX), relabel nodes as strings for compatibility
        full = nx.karate_club_graph()
        full = nx.relabel_nodes(full, lambda x: str(x))

    elif split_dir.name.startswith("facebook"):
        # Facebook: load full graph from edge list (update path as needed)
        FACEBOOK_EDGE_PATH = Path("../../datasets/Facebook/facebook_combined.txt")
        full = nx.read_edgelist(FACEBOOK_EDGE_PATH, nodetype=str)

    else:
        raise ValueError(f"Unknown split type: {split_dir.name}")

    # Always load the split's train.edgelist as the training subgraph
    train = nx.read_edgelist(split_dir / "train.edgelist", nodetype=str)

    return full, train


def embed_one_sim(walk_dir: Path) -> None:
    """
    Generate DTQW-based node embeddings in the given walk simulation directory,
    if not already present.

    - Checks for the required histories.pkl.
    - Skips if embeddings already exist.
    - Computes enabled embedding variants and saves results to walk_dir/embeddings/.
    - Stores a config.json summarizing run parameters and outputs.

    Parameters
    ----------
    walk_dir : Path
        Directory containing a completed DTQW simulation (must include histories.pkl).
    """
    hist_path = walk_dir / "histories.pkl"
    if not hist_path.is_file():
        log.warning("No histories.pkl in %s – skip", walk_dir.relative_to(SPLITS_ROOT))
        return

    embed_dir = walk_dir / "embeddings"
    cfg_path = embed_dir / "config.json"
    if cfg_path.is_file():
        log.info("Embeddings exist – skip %s", embed_dir.relative_to(SPLITS_ROOT))
        return

    embed_dir.mkdir(exist_ok=True)

    log.info("DTQW ↪︎ %s", walk_dir.relative_to(SPLITS_ROOT))
    emb = DTQWEmbedder(
        histories_path=str(hist_path),
        show_progress=True,
        random_state=RANDOM_STATE
    )
    outputs: dict[str, str] = {}

    # Always try each embedding type as configured.
    if DO_FINAL:
        p = embed_dir / "final_step.pkl"
        emb.save_embeddings(emb.final_step_embeddings(), p)
        outputs["final_step"] = p.name

    if DO_AVERAGE:
        p = embed_dir / "time_avg.pkl"
        emb.save_embeddings(emb.time_averaged_embeddings(), p)
        outputs["time_avg"] = p.name

    if DO_SVD:
        p = embed_dir / f"svd_{SVD_DIM}.pkl"
        emb.save_embeddings(emb.concatenated_embeddings(SVD_DIM), p)
        outputs[f"svd_{SVD_DIM}"] = p.name

    if DO_KERNEL_PCA:
        p = embed_dir / f"kpca_{KPCA_METRIC}_{KPCA_DIM}.pkl"
        emb.save_embeddings(
            emb.kernel_embeddings(n_components=KPCA_DIM, metric=KPCA_METRIC), p
        )
        outputs[f"kpca_{KPCA_METRIC}_{KPCA_DIM}"] = p.name

    # Write config with full provenance and output artifact paths
    cfg = {
        "random_state": RANDOM_STATE,
        "do_final": DO_FINAL,
        "do_average": DO_AVERAGE,
        "do_svd": DO_SVD,
        "do_kernel_pca": DO_KERNEL_PCA,
        "svd_dim": SVD_DIM,
        "kpca_dim": KPCA_DIM,
        "kpca_metric": KPCA_METRIC,
        "outputs": outputs,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))


def process_all_dtqw_sims() -> None:
    """
    Process all DTQW simulation folders and generate embeddings for each.

    - Searches for split directories matching DTQW_SPLIT_GLOB.
    - For each, finds all walk simulation subfolders matching WALK_GLOB.
    - Calls embed_one_sim for each, catching and logging any exceptions.
    """
    split_dirs = sorted(SPLITS_ROOT.glob(DTQW_SPLIT_GLOB), key=lambda p: p.name)
    if not split_dirs:
        log.warning("No DTQW split folders found.")
        return

    for split in split_dirs:
        walk_dirs = sorted((d for d in split.glob(WALK_GLOB) if d.is_dir()),
                           key=lambda p: p.name)
        for w in walk_dirs:
            try:
                embed_one_sim(w)
            except Exception as ex:
                log.exception("DTQW embedding failed for %s: %s",
                              w.relative_to(SPLITS_ROOT), ex)


def generate_classical_embeddings() -> None:
    """
    For every split, generate classical node embeddings (DeepWalk, Node2Vec)
    on the split's training graph.

    Embeddings and config are stored under:
        {split}/classical/embeddings/
        {split}/classical/classical_config.json

    Handles absent splits gracefully, logs progress.
    """
    split_dirs = sorted(SPLITS_ROOT.glob(DTQW_SPLIT_GLOB), key=lambda p: p.name)
    if not split_dirs:
        log.warning("No split folders found – nothing to do.")
        return

    for split in split_dirs:
        root_out = split / "classical"
        embed_dir = root_out / "embeddings"
        cfg_path = root_out / "classical_config.json"

        embed_dir.mkdir(parents=True, exist_ok=True)
        log.info("[%s] training classical baselines …", split.name)

        # Load train graph and sort nodes for stable ordering
        G_full, G_train = load_train_graph(split)
        if DTQW_SPLIT_GLOB.startswith("karate"):
            nodes_sorted = sorted(G_train.nodes(), key=int)
        elif DTQW_SPLIT_GLOB.startswith("facebook"):
            nodes_sorted = sorted(G_train.nodes(), key=str)
        else:
            nodes_sorted = sorted(G_train.nodes())

        outputs: dict[str, str] = {}

        # DeepWalk
        if DO_DEEPWALK:
            emb = run_deepwalk(
                G_train, nodes_sorted, DW_DIM, WALK_LENGTH,
                NUM_WALKS, WINDOW_SIZE, EPOCHS, seed=RANDOM_STATE
            )
            p = embed_dir / f"deepwalk_{DW_DIM}.pkl"
            DTQWEmbedder.save_embeddings(emb, p)
            outputs[f"deepwalk_{DW_DIM}"] = p.name

        # Node2Vec
        if DO_NODE2VEC:
            emb = run_node2vec(
                G_train, nodes_sorted, N2V_DIM, WALK_LENGTH, NUM_WALKS,
                WINDOW_SIZE, EPOCHS, seed=RANDOM_STATE,
                p=P_PARAM, q=Q_PARAM
            )
            p = embed_dir / f"node2vec_{N2V_DIM}.pkl"
            DTQWEmbedder.save_embeddings(emb, p)
            outputs[f"node2vec_{N2V_DIM}"] = p.name

        # Save full config and artifact paths
        cfg = {
            "random_state": RANDOM_STATE,
            "deepwalk": {"enabled": DO_DEEPWALK, "dim": DW_DIM},
            "node2vec": {
                "enabled": DO_NODE2VEC, "dim": N2V_DIM,
                "p": P_PARAM, "q": Q_PARAM
            },
            "outputs": outputs,
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log.info("[%s] classical embeddings saved → %s",
                 split.name, embed_dir.relative_to(SPLITS_ROOT))


def main() -> None:
    """
    Entry point for embedding extraction pipeline.

    - Processes all DTQW simulation folders under SPLITS_ROOT:
        * Extracts and saves various DTQW-based embeddings for each simulation.
    - Trains classical node embedding baselines (DeepWalk, Node2Vec) on each split.
    - Logs status and summaries for traceability.

    Raises
    ------
    Exception
        Any unexpected error in the pipeline is caught and logged,
        and the script will exit with a nonzero status.
    """
    try:
        log.info("=== [1/2] Extracting DTQW embeddings ===")
        process_all_dtqw_sims()
    except Exception:
        log.exception("Error during DTQW embedding extraction!")

    try:
        log.info("=== [2/2] Generating classical embeddings ===")
        generate_classical_embeddings()
    except Exception:
        log.exception("Error during classical embedding extraction!")

    log.info("✓ All embeddings (DTQW + classical) are ready.")


if __name__ == "__main__":
    main()
