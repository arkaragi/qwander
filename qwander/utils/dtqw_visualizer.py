"""
Provides the DTQWVisualizer class for visualizing the evolution and outcomes
of a discrete-time quantum walk (DTQW) on a graph.

The visualizer supports:
- Plotting log₁₀-scaled node probability distributions at the final time step.
- Plotting log₁₀-scaled node probability distributions for the first N time steps.
- Displaying external potential values applied to each node.
- Side-by-side comparison of node potentials and final-step probabilities.
- Side-by-side comparison of two final-step distributions.
- Bar-chart of the final distribution with a highlighted start node.
- Animating the DTQW probability evolution over time.
- Plotting Shannon entropy of the distribution versus time.
- Plotting a single node’s probability as a function of time.
"""

import warnings
from typing import Any, Dict, Optional

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class DTQWVisualizer:
    """
    A toolkit for visualizing Discrete‐Time Quantum Walks (DTQW) on arbitrary graphs.

    This class wraps common plotting patterns—layout computation, log-scaling,
    consistent color‐mapping, and colorbar generation—into a simple interface.
    Key features include:

    - Automatic layout selection with graceful fallback for non-planar graphs.
    - Centralized graph‐drawing helper that handles colormap normalization,
      colorbar creation, node styling, and axis cleanup.
    - Public methods for:
        • Final‐step probability snapshots (log₁₀‐scaled).
        • Side-by-side subplots of the first N time steps.
        • Node potential visualizations.
        • Combined potential vs. probability comparisons.
        • (Further methods for distribution comparison, histograms,
          animations, entropy curves, and node time series.)
    - Fully documented, type‐hinted, and easily extensible structure.

    Parameters
    ----------
    G: networkx.Graph
        The graph on which the DTQW is performed.

    layout: str, optional
        Layout algorithm for node positions (default="spring").

    seed: int, optional
        Random seed for reproducible layouts (default=42).

    walk_desc: str, optional
        A human-readable description of the walk (e.g. "Hadamard coin, T=10");
        if provided, it will be prepended to plot titles where relevant.
    """

    def __init__(self,
                 G: nx.Graph,
                 layout: str = "spring",
                 seed: int = 42,
                 walk_desc: Optional[str] = None):
        if not isinstance(G, nx.Graph):
            raise ValueError("G must be a networkx.Graph.")
        self.G = G
        self.pos = self._compute_layout(layout, seed)
        self.walk_desc = walk_desc

    def _compute_layout(self,
                        layout: str,
                        seed: int) -> Dict[Any, np.ndarray]:
        """
        Compute node positions according to the selected layout algorithm.

        Parameters
        ----------
        layout: str
            Layout name, one of:
            "spring", "circular", "kamada_kawai", "planar", "spectral", "shell".

        seed: int
            Random seed for deterministic layouts (applies to spring fallback).

        Returns
        -------
        pos: Dict[Any, np.ndarray]
            Mapping from each node in self.G to its 2D position array [x, y].

        Raises
        ------
        ValueError
            If layout is not one of the supported options.
        """
        layout = layout.lower()
        if layout == "spring":
            return nx.spring_layout(self.G, seed=seed)
        elif layout == "circular":
            return nx.circular_layout(self.G)
        elif layout == "kamada_kawai":
            return nx.kamada_kawai_layout(self.G)
        elif layout == "planar":
            try:
                return nx.planar_layout(self.G)
            except nx.NetworkXException:
                warnings.warn(
                    "Graph is not planar; falling back to spring layout.",
                    RuntimeWarning
                )
                return nx.spring_layout(self.G, seed=seed)
        elif layout == "spectral":
            return nx.spectral_layout(self.G)
        elif layout == "shell":
            return nx.shell_layout(self.G)
        else:
            raise ValueError(
                f"Unsupported layout '{layout}'. "
                "Choose from 'spring', 'circular', 'kamada_kawai', "
                "'planar', 'spectral', or 'shell'."
            )

    def _draw_colored_graph(self,
                            ax: plt.Axes,
                            values: np.ndarray,
                            cmap: str,
                            cb_label: str,
                            node_size: int = 600) -> None:
        """
        Draw the graph on ax, coloring nodes by values and adding a colorbar.
        """
        vmin, vmax = float(values.min()), float(values.max())
        nx.draw(
            self.G,
            self.pos,
            node_color=values,
            cmap=plt.get_cmap(cmap),
            with_labels=True,
            node_size=node_size,
            edge_color="gray",
            font_color="white",
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        sm = cm.ScalarMappable(
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=plt.get_cmap(cmap)
        )
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, label=cb_label)
        ax.axis("off")

    def animate_walk(self,
                     prob_df: pd.DataFrame,
                     cmap: str = "viridis",
                     eps: float = 1e-10,
                     interval: int = 300,
                     repeat: bool = False,
                     node_size: int = 600,
                     font_size: int = 16,
                     edge_alpha: float = 0.8,
                     save_as_mp4: bool = False,
                     save_path: str = "walk_animation") -> animation.FuncAnimation:
        """
        Animate the DTQW probability distribution over time, coloring nodes by log₁₀(probability).
        Improved for professional quality with a customizable save path.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        cmap: str, optional
            Matplotlib colormap for node coloring (default="viridis").

        eps: float, optional
            Small constant to avoid log(0) (default=1e-10).

        interval: int, optional
            Delay between frames in milliseconds (default=300 for smoother transitions).

        repeat: bool, optional
            Whether the animation should loop when finished (default=False).

        node_size: int, optional
            Size of the nodes (default=600).

        font_size: int, optional
            Font size for node labels and titles (default=16).

        edge_alpha: float, optional
            Transparency for edges (default=0.8 for better visual contrast).

        save_as_mp4: bool, optional
            Whether to save the animation as an mp4. If False (default) the animation
            will be saved as a GIF file.

        save_path: str, optional
            The path (including filename) to save the animation.
            Defaults to "walk_animation".
            It will append ".mp4" or ".gif" based on `save_as_mp4`.

        Returns
        -------
        ani: animation.FuncAnimation
            The FuncAnimation object; keep a reference if you plan to display or save it.
        """
        # Input validation
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        # Precompute log₁₀-scaled probabilities
        log_df = np.log10(prob_df + eps)
        times = list(log_df.index)

        # Shared color range
        vmin, vmax = float(log_df.values.min()), float(log_df.values.max())

        # Set up figure and static colorbar
        fig, ax = plt.subplots(figsize=(8, 6))
        sm = cm.ScalarMappable(
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=plt.get_cmap(cmap)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="log₁₀(Probability)", shrink=0.8)
        ax.axis("off")

        # Initial frame
        ax.set_title(f"DTQW – Time Step {times[0]}",
                     fontsize=font_size, fontweight="bold")
        nx.draw(
            self.G,
            self.pos,
            node_color=log_df.iloc[0].values,
            cmap=plt.get_cmap(cmap),
            with_labels=True,
            node_size=node_size,
            edge_color="gray",
            alpha=edge_alpha,
            vmin=vmin,
            vmax=vmax,
            font_size=font_size,
            ax=ax
        )

        # Update function
        def update(frame_idx: int):
            ax.clear()
            ax.set_title(f"DTQW – Time Step {times[frame_idx]}",
                         fontsize=font_size, fontweight="bold")
            ax.axis("off")
            nx.draw(
                self.G,
                self.pos,
                node_color=log_df.iloc[frame_idx].values,
                cmap=plt.get_cmap(cmap),
                with_labels=True,
                node_size=node_size,
                edge_color="gray",
                alpha=edge_alpha,
                vmin=vmin,
                vmax=vmax,
                font_size=font_size,
                ax=ax
            )

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=interval,
            repeat=repeat
        )

        # Save as high-quality .mp4 or .gif
        if save_as_mp4:
            ani.save(f"{save_path}.mp4", writer='ffmpeg', dpi=300)
        else:
            ani.save(f"{save_path}.gif", writer='pillow', fps=1, dpi=300)

        plt.close(fig)
        return ani

    def plot_final_step(self,
                        prob_df: pd.DataFrame,
                        cmap: str = "viridis",
                        eps: float = 1e-10) -> None:
        """
        Log₁₀-scale and plot node probabilities at the final time step.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        cmap: str, optional
            Name of a Matplotlib colormap (default="viridis").

        eps: float, optional
            Small constant to avoid log(0) (default=1e-10).
        """
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        final_idx = prob_df.index[-1]
        log_probs = np.log10(prob_df.loc[final_idx] + eps).values

        # build title using the stored walk_desc if present
        base_title = f"Final Step {final_idx}"
        if self.walk_desc:
            title = f"{self.walk_desc} — {base_title}"
        else:
            title = f"DTQW – {base_title}"

        fig, ax = plt.subplots(figsize=(12, 10))
        self._draw_colored_graph(
            ax,
            values=log_probs,
            cmap=cmap,
            cb_label="log₁₀(Probability)",
            node_size=700
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_steps_subplots(self,
                            prob_df: pd.DataFrame,
                            cmap: str = "viridis",
                            eps: float = 1e-10,
                            log_scale: bool = True,
                            title_prefix: str = "Step") -> None:
        """
        Plot up to the first 6 time steps of node probabilities in a 3×2 grid.
        Can use log₁₀-scaling or raw probabilities.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        cmap: str, optional
            Matplotlib colormap name (default="viridis").

        eps: float, optional
            Small constant to avoid log(0) when log_scale=True (default=1e-10).

        log_scale: bool, optional
            If True, color nodes by log₁₀(p + eps); otherwise by raw p (default=True).

        title_prefix: str, optional
            Prefix for each subplot title (default="Step").
        """
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        max_panels = 6
        n_available = len(prob_df)
        n_panels = min(max_panels, n_available)

        # apply scaling
        if log_scale:
            data_df = np.log10(prob_df + eps)
            cb_label = "log₁₀(Probability)"
        else:
            data_df = prob_df
            cb_label = "Probability"

        times = data_df.index[:n_panels]
        block = data_df.loc[times].values
        vmin, vmax = float(block.min()), float(block.max())

        # 3×2 grid
        fig, axes = plt.subplots(nrows=3,
                                 ncols=2,
                                 figsize=(10, 14),
                                 squeeze=False)
        axes_flat = axes.flatten()

        # draw each
        for i, t in enumerate(times):
            ax = axes_flat[i]
            vals = data_df.loc[t].values

            nx.draw(
                self.G,
                self.pos,
                node_color=vals,
                cmap=plt.get_cmap(cmap),
                with_labels=True,
                node_size=400,
                edge_color="gray",
                font_color="white",
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )

            # add one colorbar (here on the fourth panel)
            if i == 3:
                sm = cm.ScalarMappable(
                    norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    cmap=plt.get_cmap(cmap)
                )
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label=cb_label)

            ax.set_title(f"{title_prefix} {t}", fontsize=14, fontweight="bold")
            ax.axis("off")

        # remove unused panels
        for ax in axes_flat[n_panels:]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.show()

    def plot_node_potentials(self,
                             potential: Dict[Any, float],
                             cmap: str = "plasma",
                             title: Optional[str] = None) -> None:
        """
        Color-map and plot external potential values on each node.

        Parameters
        ----------
        potential: dict
            Mapping from node label to real potential value.

        cmap: str, optional
            Name of a Matplotlib colormap (default="plasma").

        title: str or None, optional
            Custom plot title. If None, uses:
              "{walk_desc} — External Potential Applied to Nodes"
            or, if no walk_desc was set, just
              "External Potential Applied to Nodes".
        """
        nodes = list(self.G.nodes())
        if set(potential.keys()) != set(nodes):
            raise ValueError("potential keys must match graph nodes exactly.")

        # Determine title
        base_title = "External Potential Applied to Nodes"
        if title:
            plot_title = title
        elif self.walk_desc:
            plot_title = f"{self.walk_desc} — {base_title}"
        else:
            plot_title = base_title

        vals = np.array([potential[n] for n in nodes], dtype=float)
        fig, ax = plt.subplots(figsize=(12, 10))
        self._draw_colored_graph(
            ax,
            values=vals,
            cmap=cmap,
            cb_label="Node Potential",
            node_size=700
        )
        ax.set_title(plot_title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_potential_and_final(self,
                                 potential: Optional[Dict[Any, float]],
                                 prob_df: pd.DataFrame,
                                 cmap_pot: str = "plasma",
                                 cmap_prob: str = "viridis",
                                 eps: float = 1e-10,
                                 title_pot: str = "Node Potential",
                                 title_prob: Optional[str] = None) -> None:
        """
        Side‐by‐side: external potentials and final‐step probabilities,
        with a single overarching figure title.

        Parameters
        ----------
        potential: dict or None
            Node→potential mapping, or None to show a bare graph.

        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        cmap_pot: str, optional
            Colormap for potentials (default="plasma").

        cmap_prob: str, optional
            Colormap for probabilities (default="viridis").

        eps: float, optional
            Small constant to avoid log(0) (default=1e-10).

        title_pot: str, optional
            Label for the left panel (default="Node Potential").

        title_prob: str or None, optional
            Label for the right panel; if None, defaults to "DTQW – Final Step {t}".
        """
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        # Prepare data
        final_idx = prob_df.index[-1]
        log_probs = np.log10(prob_df.loc[final_idx] + eps).values

        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 10))

        # Left panel: either bare graph or colored potentials
        if potential is None:
            nx.draw(self.G, self.pos,
                    with_labels=True,
                    node_size=600,
                    node_color="lightgray",
                    edge_color="gray",
                    ax=ax1)
            ax1.axis("off")
        else:
            nodes = list(self.G.nodes())
            pot_vals = np.array([potential[n] for n in nodes], dtype=float)
            self._draw_colored_graph(
                ax1,
                values=pot_vals,
                cmap=cmap_pot,
                cb_label="Node Potential"
            )

        # Right panel: log₁₀-scaled final probabilities
        self._draw_colored_graph(
            ax2,
            values=log_probs,
            cmap=cmap_prob,
            cb_label="log₁₀(Probability)"
        )

        # Compute base titles
        right_base = title_prob or f"DTQW – Final Step {final_idx}"
        base_title = f"{title_pot} vs. {right_base}"

        # Add one figure‐level (super) title=
        fig.suptitle(base_title, fontsize=18, fontweight="bold")

        # Remove individual axis titles (we only have the suptitle)
        ax1.set_title("")
        ax2.set_title("")

        # Adjust layout to make room for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_compare_distributions(self,
                                   prob_df1: pd.DataFrame,
                                   prob_df2: pd.DataFrame,
                                   label1: str = "Distribution 1",
                                   label2: str = "Distribution 2",
                                   cmap: str = "viridis",
                                   eps: float = 1e-10) -> None:
        """
        Compare two final-step node-probability distributions side by side
        (log₁₀-scaled) with a single, overarching figure title.

        Parameters
        ----------
        prob_df1, prob_df2: pd.DataFrame
            Each indexed by time step, columns = node labels, values = probabilities.
            Uses the last row of each for the comparison.

        label1, label2: str, optional
            Descriptive labels for the left and right panels.

        cmap: str, optional
            Colormap name for both plots (default="viridis").

        eps: float, optional
            Small constant to avoid log(0) (default=1e-10).
        """
        # Extract final-step series
        final1 = prob_df1.iloc[-1]
        final2 = prob_df2.iloc[-1]

        # Ensure same node labels
        if set(final1.index) != set(final2.index):
            raise ValueError("The two distributions must have the same node labels.")

        # Compute log₁₀-scaled probabilities
        log1 = np.log10(final1 + eps)
        log2 = np.log10(final2 + eps)

        # Determine an ordering for the nodes (prefer graph order)
        nodes = list(self.G.nodes())
        if set(nodes) != set(final1.index):
            nodes = list(final1.index)

        # Build color arrays in that order
        vals1 = np.array([log1[n] for n in nodes])
        vals2 = np.array([log2[n] for n in nodes])

        # Shared color range
        vmin = float(min(vals1.min(), vals2.min()))
        vmax = float(max(vals1.max(), vals2.max()))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        sm.set_array([])

        # Create side-by-side axes
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8))

        # Left panel
        nx.draw(self.G, self.pos,
                node_color=vals1,
                cmap=plt.get_cmap(cmap),
                with_labels=True,
                node_size=600,
                edge_color="gray",
                vmin=vmin,
                vmax=vmax,
                ax=ax1)
        ax1.axis('off')

        # Right panel
        nx.draw(self.G, self.pos,
                node_color=vals2,
                cmap=plt.get_cmap(cmap),
                with_labels=True,
                node_size=600,
                edge_color="gray",
                vmin=vmin,
                vmax=vmax,
                ax=ax2)
        fig.colorbar(sm, ax=ax2, label='log₁₀(Probability)')
        ax2.axis('off')

        # Build a single title
        base_title = f"{label1} vs. {label2}"
        if self.walk_desc:
            fig.suptitle(f"{self.walk_desc} — {base_title}",
                         fontsize=18, fontweight='bold')
        else:
            fig.suptitle(base_title, fontsize=18, fontweight='bold')

        # Make room for the super-title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_final_histogram(self,
                             prob_df: pd.DataFrame,
                             start_node: Any,
                             highlight_color: str = "red",
                             base_color: str = "skyblue",
                             xlabel: str = "Node",
                             ylabel: str = "Probability",
                             title: Optional[str] = None) -> None:
        """
        Bar‐chart of final‐step node probabilities, sorted by integer node label,
        with the start_node highlighted.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        start_node: Any
            Node label to highlight.

        highlight_color, base_color: str, optional
            Colors for the highlighted and other bars.

        xlabel, ylabel: str, optional
            Axis labels.

        title: str, optional
            Custom plot title. If None, defaults to
            "{walk_desc} — Final Distribution (T={final_step})" or
            "Final Distribution (T={final_step})" if no walk_desc set.
        """
        # Validate input
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        final_idx = prob_df.index[-1]
        final = prob_df.loc[final_idx]

        # Sort nodes by integer value
        try:
            nodes_int = sorted(final.index, key=lambda x: int(x))
        except Exception:
            raise ValueError("Node labels must be convertible to int for sorting.")

        if start_node not in final.index:
            raise ValueError(f"start_node '{start_node}' not found in prob_df columns.")

        # Prepare data
        probs_sorted = final.loc[nodes_int].values
        highlight_idx = nodes_int.index(start_node)
        x = np.arange(len(nodes_int))
        bar_colors = [
            highlight_color if i == highlight_idx else base_color
            for i in range(len(nodes_int))
        ]

        # Determine title
        base_title = f"Final Probability Distribution at T={final_idx}"
        if title:
            plot_title = title
        elif self.walk_desc:
            plot_title = f"{self.walk_desc} — {base_title}"
        else:
            plot_title = base_title

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x, probs_sorted, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(nodes_int, rotation=45, ha='right')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')

        # Annotate highlighted bar
        ax.annotate(
            "Start Node",
            xy=(highlight_idx, probs_sorted[highlight_idx]),
            xytext=(highlight_idx, probs_sorted.max() * 1.05),
            ha='center',
            color=highlight_color,
            arrowprops=dict(arrowstyle="->", color=highlight_color),
        )

        plt.tight_layout()
        plt.show()

    def plot_entropy_over_time(self,
                               prob_df: pd.DataFrame,
                               base: float = 2.0,
                               xlabel: str = "Time Step",
                               ylabel: str = "Shannon Entropy",
                               title: Optional[str] = None) -> None:
        """
        Compute and plot Shannon entropy versus time.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        base: float, optional
            Log base for entropy (default=2).

        xlabel: str, optional
            Label for the x‐axis.

        ylabel: str, optional
            Label for the y‐axis.

        title: str, optional
            Custom plot title. If None, defaults to
            "Shannon Entropy Over Time", or if walk_desc is set,
            "{walk_desc} — Shannon Entropy Over Time".
        """
        # Validate input
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")

        eps = 1e-12
        entropies = []
        for t in prob_df.index:
            p = prob_df.loc[t].values + eps
            H = -np.sum(p * np.log(p) / np.log(base))
            entropies.append(H)

        # Determine title
        base_title = "Shannon Entropy Over Time"
        if title:
            plot_title = title
        elif getattr(self, "walk_desc", None):
            plot_title = f"{self.walk_desc} — {base_title}"
        else:
            plot_title = base_title

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prob_df.index, entropies, marker='o', linestyle='-')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_node_time_series(self,
                              prob_df: pd.DataFrame,
                              node: Any,
                              xlabel: str = "Time Step",
                              ylabel: str = "Probability",
                              title: Optional[str] = None) -> None:
        """
        Plot a single node’s probability over time.

        Parameters
        ----------
        prob_df: pd.DataFrame
            Indexed by time step, columns = node labels, values = probabilities.

        node: Any
            The node to plot.

        xlabel: str, optional
            Label for the x‐axis.

        ylabel: str, optional
            Label for the y‐axis.

        title: str, optional
            Custom plot title. If None, defaults to
            "Probability vs. Time for Node {node}", or if walk_desc is set,
            "{walk_desc} — Probability vs. Time for Node {node}".
        """
        # Validate input
        if not isinstance(prob_df, pd.DataFrame) or prob_df.empty:
            raise ValueError("prob_df must be a non-empty pandas DataFrame.")
        if node not in prob_df.columns:
            raise ValueError(f"Node {node} not found in prob_df columns.")

        # Determine title
        base_title = f"Probability vs. Time for Node {node}"
        if title:
            plot_title = title
        elif getattr(self, "walk_desc", None):
            plot_title = f"{self.walk_desc} — {base_title}"
        else:
            plot_title = base_title

        # Plot
        series = prob_df[node]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, marker='o', linestyle='-')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
