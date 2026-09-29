"""
plot_data.py

Reusable plotting helper for DQN training curves (e.g. CartPole).
Builds a seaborn lineplot with a shaded confidence-interval band, matching
the style used in the lecture slides ("Basic deep Q learning on Cartpole").

Usage:
    from plot_data import plot_data

    # Directly from a numpy array of shape (n_runs, n_episodes):
    plot_data(all_rewards)

    # Or with multiple conditions to compare, as a list of DataFrames:
    plot_data([df_condition_a, df_condition_b])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_data(
    data,
    y: str = "accumulated_reward",
    x: str = "Episode",
    ci: int = 95,
    estimator: str = "mean",
    condition_name: str = "cartpole_dqn",
    title: str = "Basic deep Q learning on Cartpole",
    ylabel: str = "Accumulated Reward",
    save_path: str | None = None,
    **kwargs,
) -> None:
    """
    Plot mean +/- confidence-interval training curves, one line per 'Condition'.

    Parameters
    ----------
    data : np.ndarray | pd.DataFrame | list[pd.DataFrame]
        - np.ndarray of shape (n_runs, n_episodes): treated as a single
          condition, e.g. the `all_rewards` array produced by train_cartpole.
        - pd.DataFrame: must already be long-form, with columns [x, y] and
          optionally "Condition".
        - list[pd.DataFrame]: concatenated row-wise first (one DataFrame per
          run or condition), then plotted with one line per "Condition".
    y, x : str
        Column names for the y- and x-axis of the (possibly newly built)
        long-form DataFrame.
    ci : int
        Width of the bootstrapped confidence interval band (e.g. 95 -> 95% CI).
    estimator : str
        Central-tendency statistic for the line, e.g. "mean" or "median".
    condition_name : str
        Label used for the single condition when `data` is a raw numpy array.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        If given, the figure is saved to this path (dpi=150) before showing.
    **kwargs :
        Forwarded to seaborn.lineplot (e.g. linewidth, palette).
    """
    # --- Build a long-form DataFrame regardless of input type ---
    if isinstance(data, np.ndarray):
        n_runs, n_episodes = data.shape
        episodes = np.arange(1, n_episodes + 1)
        df = pd.DataFrame(
            {
                x: np.tile(episodes, n_runs),
                y: data.flatten(),
                "Condition": f"({n_runs}x){condition_name}",
            }
        )
    elif isinstance(data, list):
        df = pd.concat(data, ignore_index=True, axis=0)
    else:
        df = data

    if "Condition" not in df.columns:
        df = df.copy()
        df["Condition"] = condition_name

    # --- Plot ---
    sns.set(style="darkgrid", font_scale=1.3)
    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue="Condition",
        estimator=estimator,
        errorbar=("ci", ci),
        **kwargs,
    )

    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Condition", loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()
