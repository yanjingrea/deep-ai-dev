import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from constants.utils import set_plot_format
from launch_weekend.cls_sales_rate_regressor import LaunchSalesModel

set_plot_format(plt=plt)

import os
from os.path import dirname, realpath

OUTPUT_DIR = dirname(realpath(__file__)) + os.sep + 'output' + os.sep

# main_model = LaunchSalesModel(min_stock=50).evaluate(test_min_year=2010)
main_model = LaunchSalesModel(min_stock=50)
data = main_model.data[main_model.data['launch_year'] >= 2020].copy()
data = data.convert_dtypes()

ALL_COLUMNS = main_model.final_x_cols
n_all_columns = len(ALL_COLUMNS)
n_samples = 4
n_groups = n_all_columns // 4

idx_list = np.arange(0, n_all_columns + n_samples, n_samples)

for idx_idx, idx_num in enumerate(idx_list):

    if idx_idx == 0:
        continue

    else:
        COLUMNS = ALL_COLUMNS[idx_list[idx_idx - 1]: idx_num]

    COLORS = ["#386cb0", "#fdb462", "#7fc97f"]
    LABELS = ['Low', 'Middle', 'High']

    data['sales_rate'] = data['sales'] / data['num_of_units']

    data['sales_rate_category'], qcut_label = pd.qcut(
        data['sales_rate'],
        3,
        labels=LABELS,
        retbins=True
    )

    # A layout of 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 8), sharex="col", tight_layout=True)

    for i in range(len(COLUMNS)):
        for j in range(len(COLUMNS)):
            if i > j:
                for label, color in zip(LABELS, COLORS):
                    label_data = data[data["sales_rate_category"] == label].dropna(subset=[COLUMNS[i], COLUMNS[j]])
                    axes[i, j].scatter(
                        COLUMNS[j],
                        COLUMNS[i],
                        data=label_data,
                        color=color,
                        alpha=0.5,
                    )

                axes[i, j].set_xlabel(COLUMNS[j].replace('_', ' '))
                axes[i, j].set_ylabel(COLUMNS[i].replace('_', ' '))

            # If this is the main diagonal, add histograms
            if i == j:
                for label, color in zip(LABELS, COLORS):
                    label_data = data[data["sales_rate_category"] == label].dropna(subset=[COLUMNS[j]])
                    axes[i, j].hist(
                        COLUMNS[j],
                        data=label_data,
                        bins=15,
                        alpha=0.5
                    )

                axes[i, j].set_xlabel(COLUMNS[j].replace('_', ' '))

    for i in range(len(COLUMNS)):
        for j in range(len(COLUMNS)):
            # If on the upper triangle
            if i < j:
                axes[i, j].remove()

    # Create handles for lines ------------------------------------------------------------

        next_label = [qcut_label[i] for i in np.arange(1, len(qcut_label))] + ['']

        handles = [
            Line2D(
                [], [], label=f'[{label: .2f}, {next_label[idx]: .2f})',
                lw=0,
                marker="o",
                markersize=10,
                markerfacecolor=COLORS[idx],
            )
            for idx, label in enumerate(qcut_label[:-1])
        ]

        # Append a handle for the line
        handles += [Line2D([], [], label="y ~ x", color="#696969", lw=2)]

        # Add legend -----------------------------------------------------
        legend = fig.legend(
            handles=handles,
            bbox_to_anchor=[0.5, 0.9],  # Located in the top-mid of the figure.
            fontsize=10,
            handletextpad=0.6,  # Space between text and marker/line
            handlelength=1.4,
            columnspacing=1.4,
            loc="center",
            ncol=6,
            frameon=False
        )

        # Set transparency -----------------------------------------------
        # Iterate through first five handles and set transparency
        for i in range(len(qcut_label)):
            handle = legend.legend_handles[i]
            handle.set_alpha(0.5)


    plt.savefig(OUTPUT_DIR + f'triangle {idx_idx}.png', dpi=300)
print()