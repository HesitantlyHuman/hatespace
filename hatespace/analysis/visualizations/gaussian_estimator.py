from typing import List
import numpy as np
import seaborn
import matplotlib.pyplot as plt


def _gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def _get_kernel_values(
    x_values: np.ndarray,
    archetypes: np.ndarray,
    kernel_resolution: int = 1000,
    kernel_std: float = 5.0,
) -> List[List[float]]:
    smoothed_points = []
    kernel = lambda kernel_position, point_positions: _gaussian(
        point_positions, kernel_position, sig=kernel_std
    )
    smoothed_x_values = np.linspace(min(x_values), max(x_values), num=kernel_resolution)
    for kernel_position in smoothed_x_values:
        kernel_value_at_points = kernel(
            kernel_position=kernel_position, point_positions=x_values
        )
        local_point_density = np.sum(kernel_value_at_points)
        archetype_values = np.sum(
            archetypes * np.expand_dims(kernel_value_at_points, axis=1), axis=0
        )
        smoothed_points.append(archetype_values / local_point_density)
    return smoothed_x_values, np.array(smoothed_points)


def _plot_archetypes(
    x_positions: np.ndarray,
    archetype_values: np.ndarray,
    archetype_names: List[str],
    title: str,
    axis_title: str,
):
    archetype_values = np.copy(archetype_values)
    num_archetypes = len(archetype_values[0])
    lines = [archetype_values[:, archetype] for archetype in range(num_archetypes)]
    fig, ax = plt.subplots(figsize=(12, 8))
    handles = plt.stackplot(x_positions, lines)
    if archetype_names is None:
        archetype_names = [i for i in range(len(handles))]
    if not axis_title is None:
        ax.set_xlabel(axis_title)
    plt.legend(handles, archetype_names, loc="upper left")
    plt.title(title)
    ax.set_ylabel("Average Archetype Proportion")
    return fig, ax


def softmax_kde_plot(
    x_values: List[float],
    point_archetypes: List[List[float]],
    kernel_size: float = 1.0,
    kernel_resolution: int = 1000,
    title: str = None,
    labels: List[str] = None,
    axis_title: str = None,
) -> None:
    seaborn.set_theme(
        style="darkgrid"
    )  # TODO: change this back after, so that we don't have random side effects
    if not isinstance(x_values, np.ndarray):
        x_values = np.array(x_values)
    if not isinstance(point_archetypes, np.ndarray):
        point_archetypes = np.array(point_archetypes)
    smoothed_x_values, smoothed_archetype_values = _get_kernel_values(
        x_values=x_values,
        archetypes=point_archetypes,
        kernel_resolution=kernel_resolution,
        kernel_std=kernel_size,
    )
    return _plot_archetypes(
        smoothed_x_values,
        smoothed_archetype_values,
        archetype_names=labels,
        title=title,
        axis_title=axis_title,
    )
