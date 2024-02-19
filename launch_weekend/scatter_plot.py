import numpy as np
import pandas as pd

from adjustText import adjust_text
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


def scatter_plot_with_reg_and_label(
    data,
    x_col,
    y_col,
    label_col,
    n_bins,
    ax=None,
    fig=None
):
    q = np.arange(0, 1 + 1 / n_bins, 1 / n_bins)

    x = data[x_col]
    y = data[y_col]

    # Colors Setting ------------------------------------------------------------

    data['label'], qcut_label = pd.qcut(data[label_col], q=q, retbins=True)
    qcut_label = qcut_label.astype(int)[:-1]
    label_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
    category_codes = pd.Categorical(data["label"]).codes
    colors = np.array(label_colors)[category_codes]
    edgecolors = [adjust_lightness(color, 0.6) for color in colors]

    # Scatter plot ------------------------------------------------------------
    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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
    # ax.plot(x_pred, x_pred, color="#DCD7C1", lw=2, linestyle='--')

    # Layout Setting -----------------------------------------------------
    plt.rcParams.update({"font.size": "10"})
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(axis="y")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    # ax.set_title(title)

    # Create handles for lines ------------------------------------------------------------
    next_label = [qcut_label[i] for i in np.arange(1, len(qcut_label))] + ['']

    handles = [
        Line2D(
            [], [], label=label if label_col != 'transaction_year' else f'[{label}, {next_label[idx]})',
            lw=0,  # there's no line added, just the marker
            marker="o",  # circle marker
            markersize=10,
            markerfacecolor=label_colors[idx],  # marker fill color
        )
        for idx, label in enumerate(qcut_label)
    ]

    # Append a handle for the line
    handles += [Line2D([], [], label="y ~ x", color="#696969", lw=2)]

    # Add legend -----------------------------------------------------
    legend = ax.legend(
        handles=handles,
        bbox_to_anchor=[0.5, 1],  # Located in the top-mid of the figure.
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
    for i in range(5):
        handle = legend.legendHandles[i]
        handle.set_alpha(0.5)

    # Specify countries ----------------------------------------------
    local_y_pred = linear_regressor.predict(X)

    project_highlight = np.array([])
    pos = []

    for i, position in zip(
            [y[y > x * 1.2].index, y[y < x * 0.8].index],
            ['under-estimate', 'over-estimate']
    ):

        temp = data['project_display_name'].loc[i]

        if temp.empty:
            continue

        if len(temp) >= 5:
            ps = temp.sample(5).values
        else:
            ps = temp.values

        project_highlight = np.append(project_highlight, ps)
        pos += [position]

    # # Add labels -----------------------------------------------------
    # texts = []
    #
    # for idx, project in enumerate(data['project_display_name']):
    #     if project in project_highlight:
    #         px, py = x.iloc[idx] + 0.1, y.iloc[idx] + 0.15
    #         texts.append(ax.text(px, py, project, fontsize=10))
    #
    # adjust_text(
    #     texts,
    #     expand=(0.1, 0.15),
    #     arrowprops=dict(arrowstyle="-", lw=0.5),
    #     ax=ax
    # )

    return fig, ax
