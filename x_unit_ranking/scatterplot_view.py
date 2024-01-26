import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

from constants.utils import set_plot_format

set_plot_format(plt)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1] * amount, c[2])


def scatter_plot_view(
        x,
        y,
        label_series,
        title,
        ax=None,
        fig=None
):
    label = np.sort(label_series.unique())
    label_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
    category_codes = pd.Categorical(label_series).codes
    colors = np.array(label_colors)[category_codes]
    edgecolors = [adjust_lightness(color, 0.6) for color in colors]

    # Scatter plot ------------------------------------------------------------
    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        rand_jitter(x), y, color=colors, edgecolors=edgecolors,
        s=80, alpha=0.5, zorder=10,
    )

    # Regression -----------------------------------------------------
    # * scikit-learn asks 2-dimensional arrays for X, that's why the reshape
    X = np.array(x).reshape(-1, 1)
    y = y
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)
    x_pred = np.linspace(x.min(), x.max(), num=200).reshape(-1, 1)
    y_pred = linear_regressor.predict(x_pred)
    ax.plot(x_pred, y_pred, color="#696969", lw=2)
    ax.plot(x_pred, x_pred, color="#DCD7C1", lw=2, linestyle='--')

    # Layout Setting -----------------------------------------------------
    plt.rcParams.update({"font.size": "12"})
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(axis="y")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlabel(f'Actual Percentile Ranking')
    ax.set_ylabel(f'Predictive Percentile Ranking')

    # Create handles for lines ------------------------------------------------------------
    handles = [
        Line2D(
            [], [], label=label,
            lw=0,  # there's no line added, just the marker
            marker="o",  # circle marker
            markersize=10,
            markerfacecolor=label_colors[idx],  # marker fill color
        )
        for idx, label in enumerate(label)
    ]

    # Append a handle for the line
    handles += [Line2D([], [], label="y ~ x", color="#696969", lw=2)]

    # Add legend -----------------------------------------------------
    legend = fig.legend(
        handles=handles,
        bbox_to_anchor=[0.5, 0.95],  # Located in the top-mid of the figure.
        fontsize=12,
        handletextpad=0.6,  # Space between text and marker/line
        handlelength=1.4,
        columnspacing=1.4,
        loc="center",
        ncol=6,
        frameon=False
    )

    # Set transparency -----------------------------------------------
    # Iterate through first five handles and set transparency
    for i in range(len(legend.legendHandles)):
        handle = legend.legendHandles[i]
        handle.set_alpha(0.5)

    plt.suptitle(title)

    return fig, ax
