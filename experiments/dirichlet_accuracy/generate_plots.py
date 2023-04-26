import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hatespace

parser = argparse.ArgumentParser(description="Plot sampled dirichlet accuracy trials.")
parser.add_argument(
    "--data",
    type=str,
    help="location of trial csv data",
    default="experiments/dirichlet_accuracy/trial_data.csv",
)
parser.add_argument(
    "--out",
    type=str,
    help="location to save plots",
    default="experiments/dirichlet_accuracy/plots",
)
args = parser.parse_args()

dataframe = pd.read_csv(args.data)

sns.set_context("paper")
with sns.axes_style("white"):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    ax.set_title("Mean Error by Number of Samples")
    sns.lineplot(
        data=dataframe[dataframe["dimensions"] == 16][dataframe["batch_size"] == 32],
        x="num_sampled",
        y="error",
        ax=ax,
    )

with sns.axes_style("darkgrid"):
    inset_ax = plt.axes([0.44, 0.36, 0.4, 0.4])
    sns.lineplot(
        data=dataframe[dataframe["dimensions"] == 16][dataframe["batch_size"] == 32],
        x="num_sampled",
        y="error",
        ax=inset_ax,
    )
    inset_ax.set_title("Zoom")
    inset_ax.set_xlim([-5, 150])

figure_name = "num_samples.png"
plt.savefig(os.path.join(args.out, figure_name))

sns.set_context("paper")
with sns.axes_style("white"):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    ax.set_title("Mean Error by Batch Size")
    sns.lineplot(data=dataframe, x="batch_size", y="error", ax=ax)

figure_name = "batch_size.png"
plt.savefig(os.path.join(args.out, figure_name))

sns.set_context("paper")
with sns.axes_style("white"):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    ax.set_title("Mean Error by Dirichlet Dimensionality")
    sns.lineplot(data=dataframe, x="dimensions", y="error", ax=ax)

figure_name = "dimensions.png"
plt.savefig(os.path.join(args.out, figure_name))

sns.set_context("paper")
with sns.axes_style("white"):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    ax.set_title("Mean Error by Dirichlet Alpha")
    sns.lineplot(data=dataframe, x="alpha", y="error", ax=ax)

figure_name = "alpha.png"
plt.savefig(os.path.join(args.out, figure_name))
