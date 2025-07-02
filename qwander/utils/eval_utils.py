"""
Utility functions for loading, aligning, generating, and evaluating
classical and DTQW-based graph embeddings.
"""

import pathlib
import pickle
import re
from typing import Dict, Union, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from qwander.qwander.utils.evaluation import LinkPredictionEvaluator
from qwander.qwander.utils.evaluation import NodeClassificationEvaluator


def _load_edges(path: Union[str, pathlib.Path]) -> List[Tuple[str, str]]:
    """Helper that reads a whitespace-delimited edgelist into a list[(u,v)]."""
    edges: List[Tuple[str, str]] = []
    with pathlib.Path(path).open() as fh:
        for line in fh:
            u, v, *_ = line.strip().split()
            edges.append((u, v))
    return edges


def _load_embeddings(filepath: Union[str, pathlib.Path]) -> Union[pd.DataFrame, np.ndarray]:
    """
    Load a pickled embedding from disk, supporting glob patterns.

    If a glob pattern is passed, finds the first matching file.
    Supports pandas.DataFrame or numpy.ndarray pickled via either
    pandas.to_pickle or pickle.dump.

    Parameters
    ----------
    filepath : str or Path
        Path (or glob pattern) to the embedding file.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The loaded embeddings.

    Raises
    ------
    FileNotFoundError
        If `filepath` does not exist or matches no file.
    ValueError
        If the loaded object is not a DataFrame or ndarray.
    """
    path = pathlib.Path(filepath)
    if "*" in str(path) or "?" in str(path) or "[" in str(path):
        # It's a glob pattern; resolve it
        matches = list(path.parent.glob(path.name))
        if not matches:
            raise FileNotFoundError(f"No embedding file found matching: {filepath}")
        if len(matches) > 1:
            raise FileNotFoundError(f"Multiple files found matching: {filepath} → {matches}")
        path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(f"No such embedding file: {path}")

    with path.open("rb") as fh:
        obj = pickle.load(fh)

    if not isinstance(obj, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"Unsupported embedding type {type(obj).__name__}; "
            "expected pandas.DataFrame or numpy.ndarray."
        )

    return obj



def _ensure_order(emb: Union[pd.DataFrame, np.ndarray],
                  node_list: List[str]) -> np.ndarray:
    """
    Align an embedding to a specific node ordering and return as an ndarray.

    - If `emb` is a DataFrame, reindexes its rows to `node_list`, ensuring
      all nodes are present (raises KeyError if any are missing).
    - If `emb` is an ndarray, checks that its first dimension matches
      `len(node_list)`.

    Parameters
    ----------
    emb : pandas.DataFrame or numpy.ndarray
        The embedding to reorder.
        - DataFrame must use node labels as its index.
        - ndarray must already be in the desired row order.
    node_list : List[str]
        Desired ordering of node labels.

    Returns
    -------
    numpy.ndarray
        Embedding matrix with rows in the order of `node_list`.

    Raises
    ------
    KeyError
        If the DataFrame is missing any nodes from `node_list`.
    ValueError
        If the ndarray’s first axis does not match `len(node_list)`.
    TypeError
        If `emb` is neither a DataFrame nor an ndarray.
    """
    if isinstance(emb, pd.DataFrame):
        df = emb.copy()
        df.index = df.index.map(str)
        # this will introduce NaNs if any node is missing
        df = df.reindex(index=node_list)
        if df.isna().any(axis=None):
            missing = [
                n for n, row in zip(node_list, df.values) if np.all(np.isnan(row))]
            raise KeyError(f"Missing embeddings for nodes: {missing}")
        return df.values

    if isinstance(emb, np.ndarray):
        if emb.shape[0] != len(node_list):
            raise ValueError(
                f"Embedding array has {emb.shape[0]} rows but expected {len(node_list)}."
            )
        return emb

    raise TypeError(
        f"Cannot ensure order for object of type {type(emb).__name__}; "
        "expected pandas.DataFrame or numpy.ndarray."
    )


def timestep_from_name(name: str) -> int:
    """
    Extract the integer after the last 'T' in a folder name.

    For example:
        'karate_grover_nopot_T40' -> 40

    Parameters
    ----------
    name : str
        Directory or experiment name ending in '_T<digits>'.

    Returns
    -------
    int
        The extracted timestep, or 0 if the pattern is not found.
    """
    m = re.search(r"_T(\d+)$", name)
    return int(m.group(1)) if m else 0


def plot_all_tsne(embeddings_dict: Dict[str, np.ndarray],
                  report_df: pd.DataFrame,
                  labels: np.ndarray,
                  out_dir: pathlib.Path,
                  random_state: int = 42) -> None:
    """
    Take a dictionary of embeddings and plot a 2D t-SNE projection for each model,
    sorted by descending test_accuracy. Displays them in a grid of subplots,
    colors points by `labels`, and saves the figure to `out_dir/tsne_embeddings.png`.

    Parameters
    ----------
    embeddings_dict : Dict[str, np.ndarray]
        Mapping from model name to embedding matrix of shape (n_nodes, dim).
    report_df : pd.DataFrame
        DataFrame indexed by model name (must match keys of embeddings_dict),
        containing a column "test_accuracy" for ranking.
    labels : np.ndarray
        1D array of length n_nodes of integer class labels.
    out_dir : pathlib.Path
        Directory where the figure will be saved (created if needed).
    random_state : int
        Seed for t-SNE reproducibility.
    """
    # ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare color mapping
    classes = np.unique(labels)
    norm = Normalize(vmin=classes.min(), vmax=classes.max())
    cmap = plt.cm.Set1

    # sort by descending test_accuracy
    sorted_models = report_df.sort_values("test_accuracy", ascending=False).index.tolist()
    n_models = len(sorted_models)

    # grid layout
    n_cols = 2
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, model_name in enumerate(sorted_models):
        ax = axes[idx]
        emb = embeddings_dict[model_name]
        emb_2d = TSNE(n_components=2, random_state=random_state).fit_transform(emb)

        ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.8,
            edgecolor="k"
        )
        acc = report_df.loc[model_name, "test_accuracy"]
        ax.set_title(f"{model_name}\n", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # drop unused axes
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    # build legend from same norm and cmap
    handles = [Patch(color=cmap(norm(c)), label=str(c)) for c in classes]
    fig.legend(handles=handles, loc="upper right", title="Label")

    plt.tight_layout()
    fig.savefig(out_dir / "tsne_embeddings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_umap(embeddings_dict: Dict[str, np.ndarray],
                  report_df: pd.DataFrame,
                  labels: np.ndarray,
                  out_dir: pathlib.Path,
                  random_state: int = 42) -> None:
    """
    Take a dictionary of embeddings and plot a 2D UMAP projection for each model,
    sorted by descending test_accuracy. Displays them in a grid of subplots,
    colors points by `labels`, and saves the figure to `out_dir/umap_embeddings.png`.

    Parameters
    ----------
    embeddings_dict : Dict[str, np.ndarray]
        Mapping from model name to embedding matrix of shape (n_nodes, dim).
    report_df : pd.DataFrame
        DataFrame indexed by model name (must match keys of embeddings_dict),
        containing a column "test_accuracy" for ranking.
    labels : np.ndarray
        1D array of length n_nodes of integer class labels.
    out_dir : pathlib.Path
        Directory where the figure will be saved (created if needed).
    random_state : int
        Seed for UMAP reproducibility.
    """
    # ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare color mapping
    classes = np.unique(labels)
    norm = Normalize(vmin=classes.min(), vmax=classes.max())
    cmap = plt.cm.Set1

    # sort by descending test_accuracy
    sorted_models = report_df.sort_values("test_accuracy", ascending=False).index.tolist()
    n_models = len(sorted_models)

    # grid layout
    n_cols = 2
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
    for idx, model_name in enumerate(sorted_models):
        ax = axes[idx]
        emb = embeddings_dict[model_name]
        emb_2d = reducer.fit_transform(emb)

        ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.8,
            edgecolor="k"
        )
        acc = report_df.loc[model_name, "test_accuracy"]
        ax.set_title(f"{model_name}\n", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # drop unused axes
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    # build legend from same norm and cmap
    handles = [Patch(color=cmap(norm(c)), label=str(c)) for c in classes]
    fig.legend(handles=handles, loc="upper right", title="Label")

    plt.tight_layout()
    fig.savefig(out_dir / "umap_embeddings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(df_master: pd.DataFrame,
                      metric: str,
                      splits: list[float],
                      exp_order: list[str],
                      out_path: pathlib.Path,
                      bar_w: float = 0.35,
                      figsize_scale: float = 0.6) -> None:
    """
    Create horizontal bar‐chart panels, one per experiment, showing `metric`
    across embedding methods and test splits. Automatically reserves extra
    space on the right so annotations and legends never collide with bars
    that reach 1.0.

    Parameters
    ----------
    df_master : pandas.DataFrame
        Master results table, with columns ["experiment","embedding","test_split",<metrics>].
    metric : str
        One of the metric column names (e.g. "auc" or "average_precision").
    splits : list of float
        List of test_split values (fractions) in the order to plot.
    exp_order : list of str
        Experiment names in the desired vertical order.
    out_path : pathlib.Path
        File path where the figure will be saved (PNG).
    bar_w : float
        Bar thickness factor (default 0.35).
    figsize_scale : float
        Vertical scaling factor for figure height (default 0.6).
    """
    n_exp = len(exp_order)
    n_emb = df_master["embedding"].nunique()
    fig_h = figsize_scale * (n_emb * n_exp)

    # compute global max and add ~12% headroom up to 1.0
    global_max = df_master[df_master["test_split"].isin(splits)][metric].max()
    x_max = min(1.0, global_max * 1.12 + 0.02)

    fig, axes = plt.subplots(n_exp, 1, figsize=(8, fig_h), sharex=False, squeeze=False)

    for row, exp in enumerate(exp_order):
        ax = axes[row, 0]
        df_exp = (
            df_master[df_master["experiment"] == exp]
            .pivot_table(index="embedding",
                         columns="test_split",
                         values=metric,
                         aggfunc="mean")
            .loc[:, splits]
            .sort_values(by=max(splits))
        )

        y_pos = np.arange(len(df_exp))
        for i, split in enumerate(df_exp.columns):
            bars = ax.barh(
                y_pos + (i - len(df_exp.columns) / 2) * bar_w,
                df_exp[split],
                height=bar_w,
                label=f"{int(split * 100)}% hold-out"
            )
            for bar in bars:
                w = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                # place annotation inside if near right edge
                if w > x_max * 0.90:
                    ax.text(w - 0.04, y, f"{w:.3f}", va="center", ha="right", fontsize=10)
                else:
                    ax.text(w + 0.01, y, f"{w:.3f}", va="center", ha="left", fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_exp.index, fontsize=9)

        pretty = metric.upper() if metric == "auc" else metric.replace("_", " ").title()
        ax.set_xlabel(pretty, fontsize=9)
        ax.set_title(f"Case: {exp} — {pretty}", loc="left", fontsize=10, pad=4)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # legend on right
        if row == 0:
            ax.legend(
                fontsize=10,
                frameon=False,
                bbox_to_anchor=(1.25, 1.0),
                loc="upper left"
            )

    plt.tight_layout(h_pad=1.0, rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_embeddings_karate(node_list: List[str],
                                   dtqw_embed_dir: Union[str, pathlib.Path],
                                   svd_dim: int = 32,
                                   kpca_dim: int = 32,
                                   kpca_metric: str = "bhattacharyya") -> Dict[str, np.ndarray]:
    """
    Load classical (DeepWalk, Node2Vec) and DTQW-based embeddings for the Karate Club graph,
    returning a dict mapping method names to ordered embedding arrays.

    Parameters
    ----------
    node_list : List[str]
        Nodes in the graph, in the desired row order.
    dtqw_embed_dir : str or Path
        Directory containing DTQW embeddings (final_step.pkl, time_avg.pkl, etc.).
    svd_dim : int
        Dimension for the SVD-reduced DTQW embeddings.
    kpca_dim : int
        Dimension for the KPCA-reduced DTQW embeddings.
    kpca_metric : str
        Kernel metric for KPCA (e.g. "bhattacharyya", "cosine").

    Returns
    -------
    Dict[str, np.ndarray]
        Embedding name → (n_nodes × dim) array, rows aligned to `node_list`.
    """
    base = pathlib.Path(dtqw_embed_dir)
    classical = base.parent.parent / "karate_classical"

    embeddings: Dict[str, np.ndarray] = {}

    # classical embeddings
    for name, fname in [
        ("DeepWalk", "deepwalk_32.pkl"),
        ("Node2Vec", "node2vec_32.pkl"),
    ]:
        path = classical / fname
        emb = _load_embeddings(path)
        embeddings[name] = _ensure_order(emb, node_list)

    # DTQW embeddings
    dtqw_files = [
        ("DTQW_Final", "final_step.pkl"),
        ("DTQW_Average", "time_avg.pkl"),
        (f"DTQW_SVD", f"svd_{svd_dim}.pkl"),
        (f"DTQW_KPCA", f"kpca_{kpca_metric}_{kpca_dim}.pkl"),
    ]
    for name, fname in dtqw_files:
        path = base / fname
        emb = _load_embeddings(path)
        embeddings[name] = _ensure_order(emb, node_list)

    return embeddings


def generate_all_embeddings_karate_lp(node_list: List[str],
                                      dtqw_embed_dir: Union[str, pathlib.Path]) -> Dict[str, np.ndarray]:
    """
    Load classical (DeepWalk, Node2Vec) and DTQW-based embeddings for the Cora citation network,
    returning a dict mapping method names to ordered embedding arrays.

    Returns
    -------
    Dict[str, np.ndarray]
        Embedding name → (n_nodes × dim) array, rows aligned to `node_list`.
    """
    base = pathlib.Path(dtqw_embed_dir)

    embeddings: Dict[str, np.ndarray] = {}

    # DTQW embeddings
    dtqw_files = [
        ("DTQW_Final", "final_step.pkl"),
        ("DTQW_Average", "time_avg.pkl"),
        (f"DTQW_SVD", f"svd_*.pkl"),
        (f"DTQW_KPCA", f"kpca_*.pkl"),
    ]
    for name, fname in dtqw_files:
        path = base / fname
        emb = _load_embeddings(path)
        embeddings[name] = _ensure_order(emb, node_list)

    return embeddings


def generate_all_embeddings_cora(node_list: List[str],
                                 dtqw_embed_dir: Union[str, pathlib.Path],
                                 svd_dim: int = 64,
                                 kpca_dim: int = 64,
                                 kpca_metric: str = "bhattacharyya") -> Dict[str, np.ndarray]:
    """
    Load classical (DeepWalk, Node2Vec) and DTQW-based embeddings for the Cora citation network,
    returning a dict mapping method names to ordered embedding arrays.

    Parameters
    ----------
    node_list : List[str]
        Nodes in the graph, in the desired row order.
    dtqw_embed_dir : str or Path
        Directory containing DTQW embeddings (final_step.pkl, time_avg.pkl, etc.).
    svd_dim : int
        Dimension for the SVD-reduced DTQW embeddings.
    kpca_dim : int
        Dimension for the KPCA-reduced DTQW embeddings.
    kpca_metric : str
        Kernel metric for KPCA (e.g. "bhattacharyya", "cosine").

    Returns
    -------
    Dict[str, np.ndarray]
        Embedding name → (n_nodes × dim) array, rows aligned to `node_list`.
    """
    base = pathlib.Path(dtqw_embed_dir)

    embeddings: Dict[str, np.ndarray] = {}

    # classical embeddings
    for name, fname in [
        ("DeepWalk", "deepwalk_64.pkl"),
        ("Node2Vec", "node2vec_64.pkl"),
    ]:
        path = base / fname
        emb = _load_embeddings(path)
        embeddings[name] = _ensure_order(emb, node_list)

    # DTQW embeddings
    dtqw_files = [
        ("DTQW_Final", "final_step.pkl"),
        ("DTQW_Average", "time_avg.pkl"),
        (f"DTQW_SVD", f"svd_{svd_dim}.pkl"),
        (f"DTQW_KPCA", f"kpca_{kpca_metric}_{kpca_dim}.pkl"),
    ]
    for name, fname in dtqw_files:
        path = base / fname
        emb = _load_embeddings(path)
        embeddings[name] = _ensure_order(emb, node_list)

    return embeddings


def evaluate_models_and_plot_link_prediction(embeddings_dict: Dict[str, np.ndarray],
                                             nodes_list: List[str],
                                             test_edges: Optional[Tuple[list, list]] = None,
                                             split_dir: Optional[pathlib.Path] = None,
                                             score: str = "cosine",
                                             out_dir: pathlib.Path = pathlib.Path("../scripts_link_prediction"),
                                             title: str | None = None) -> pd.DataFrame:
    """
    Evaluate *pre-trained* embeddings on a **fixed** link-prediction split.

    Parameters
    ----------
    embeddings_dict
        {model_name → (n_nodes, dim) ndarray}.  Row order must follow `nodes_list`.
    nodes_list
        Node IDs in embedding-row order.
    test_edges
        Tuple ``(test_pos, test_neg)`` – each a list of (u,v) strings.
    split_dir
        Alternative: point at the folder produced by *split_edges.py*.
        The function will read ``test_pos.edgelist`` & ``test_neg.edgelist``.
    out_dir
        Where to write CSV + PNG.
    title
        Custom plot title.  If None, a sensible default is used.

    Returns
    -------
    pd.DataFrame  – index = model names, columns = all metrics returned
                    by ``evaluate_link_prediction``.
    """
    if (test_edges is None) == (split_dir is None):
        raise ValueError("Pass *either* test_edges=(pos,neg) *or* split_dir, not both.")

    if split_dir is not None:
        test_pos = _load_edges(pathlib.Path(split_dir) / "test_pos.edgelist")
        test_neg = _load_edges(pathlib.Path(split_dir) / "test_neg.edgelist")
    else:
        test_pos, test_neg = test_edges

    if len(test_pos) != len(test_neg):
        raise ValueError(f"|E⁺|={len(test_pos)}  !=  |E⁻|={len(test_neg)}")

    n_nodes = len(nodes_list)
    for name, emb in embeddings_dict.items():
        if emb.ndim != 2 or emb.shape[0] != n_nodes:
            raise ValueError(
                f"Embedding '{name}' has shape {emb.shape}; expected ({n_nodes}, dim)."
            )

    lpe = LinkPredictionEvaluator(score=score)
    records: Dict[str, Dict[str, float]] = {}
    for name, emb in embeddings_dict.items():
        metrics = lpe.evaluate(
            embeddings=emb,
            nodes_list=nodes_list,
            test_pos=test_pos,
            test_neg=test_neg
        )
        records[name] = metrics

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "embedding"

    aucs = df["auc"].sort_values()
    best = aucs.idxmax()
    colors = ["skyblue"] * len(aucs)
    colors[list(aucs.index).index(best)] = "mediumseagreen"

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(aucs)))
    bars = ax.barh(aucs.index, aucs.values,
                   color=colors, edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9)

    default_title = f"Link-Prediction AUC  (|E⁺| = {len(test_pos)})"
    ax.set_title(title or default_title)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # ------------------------------------------------------------------- writing --
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "linkpred_metrics.csv", float_format="%.6f")
    fig.savefig(out_dir / "linkpred_auc.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return df


# def evaluate_models_and_plot_node_cls(embeddings_dict: Dict[str, np.ndarray],
#                                       labels: np.ndarray,
#                                       test_size: float = 0.3,
#                                       random_state: int = 42,
#                                       out_dir: pathlib.Path = pathlib.Path("."),
#                                       title: str | None = None) -> pd.DataFrame:
#     """
#     Train & evaluate logistic‐regression classifiers on multiple embeddings,
#     build a summary table, and plot their test‐set confusion matrices side by side.
#
#     Parameters
#     ----------
#     embeddings_dict : Dict[str, np.ndarray]
#         Mapping from model name to embedding matrix of shape (n_samples, dim).
#         Rows must align with `labels`.
#     labels : numpy.ndarray of shape (n_samples,)
#         Ground-truth labels for each node.
#     test_size : float, default=0.3
#         Fraction of data held out for testing (0 < test_size < 1).
#     random_state : int, default=42
#         Seed for train/test splitting and classifier.
#     out_dir : pathlib.Path
#         Directory to save the summary CSV and confusion‐matrix figure.
#     title : str or None
#         Custom title for the confusion‐matrix figure. Defaults to
#         "Node Classification (test size = XX%)".
#
#     Returns
#     -------
#     pandas.DataFrame
#         Indexed by model name, with columns:
#         ["train_accuracy","test_accuracy","precision","recall","f1",
#          "precision_macro","recall_macro","f1_macro"].
#
#     Raises
#     ------
#     ValueError
#         If inputs are invalid (empty embeddings, shape mismatches, invalid test_size).
#     """
#     logger = logging.getLogger(__name__)
#
#     # ─────────── Validation ───────────
#     if not embeddings_dict:
#         raise ValueError("No embeddings provided for evaluation.")
#     if not (0.0 < test_size < 1.0):
#         raise ValueError(f"test_size must be in (0,1); got {test_size!r}.")
#     n = labels.shape[0]
#     for name, emb in embeddings_dict.items():
#         if emb.ndim != 2 or emb.shape[0] != n:
#             raise ValueError(
#                 f"Embedding '{name}' has shape {emb.shape}; expected ({n}, dim)."
#             )
#
#     # ─────────── Evaluation ───────────
#     records: Dict[str, Dict[str, float]] = {}
#     confusion_matrices = []
#     model_names = []
#
#     for name, emb in embeddings_dict.items():
#         res = evaluate_node_classification(
#             embeddings=emb,
#             labels=labels,
#             test_size=test_size,
#             random_state=random_state
#         )
#         model_names.append(name)
#         confusion_matrices.append(res["confusion_matrix_test"])
#
#         # extract scalar metrics
#         records[name] = {
#             "train_accuracy": res["train_accuracy"],
#             "test_accuracy": res["test_accuracy"],
#             "precision": res["precision"],
#             "recall": res["recall"],
#             "f1": res["f1"],
#             "precision_macro": res["precision_macro"],
#             "recall_macro": res["recall_macro"],
#             "f1_macro": res["f1_macro"],
#         }
#
#     df = pd.DataFrame.from_dict(records, orient="index")
#     df.index.name = "embedding"
#
#     # ─────────── Persist metrics CSV ───────────
#     if out_dir is not None:
#         out_dir.mkdir(parents=True, exist_ok=True)
#         csv_path = out_dir / f"classif_metrics_{int(test_size * 100)}.csv"
#         df.to_csv(csv_path, float_format="%.6f")
#         logger.info("Saved metrics → %s", csv_path)
#
#     # ─────────── Plot confusion matrices ───────────
#     n_models = len(model_names)
#     cols = 2
#     rows = ceil(n_models / cols)
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
#
#     for idx, (name, cm) in enumerate(zip(model_names, confusion_matrices)):
#         r, c = divmod(idx, cols)
#         ax = axes[r][c]
#         im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
#         ax.set_title(f"{name} — Test CM")
#         ax.set_xlabel("Predicted")
#         ax.set_ylabel("True")
#         # remove ticks
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # annotate cells
#         thresh = cm.max() / 2
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 color = "white" if cm[i, j] > thresh else "black"
#                 ax.text(j, i, cm[i, j],
#                         ha="center", va="center", color=color, fontsize=16)
#     # remove any unused subplots
#     for idx in range(n_models, rows * cols):
#         r, c = divmod(idx, cols)
#         fig.delaxes(axes[r][c])
#
#     split_pct = int(test_size * 100)
#     default_title = f"Node Classification Confusion Matrices (test size = {split_pct}%)"
#     fig.suptitle(title or default_title, fontsize=14, y=1.02)
#     plt.tight_layout()
#
#     if out_dir is not None:
#         cm_path = out_dir / f"classif_cm_{split_pct}.png"
#         fig.savefig(cm_path, dpi=300, bbox_inches="tight")
#         logger.info("Saved confusion matrices → %s", cm_path)
#     plt.close(fig)
#
#     return df
