import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from launch_weekend.cls_sales_rate_regressor import LaunchSalesModel

main_model = LaunchSalesModel(min_stock=50)
data = main_model.data[main_model.data['launch_year'] >= 2020].copy()
data = data.convert_dtypes()
data['sales_rate'] = data['sales'] / data['num_of_units']

data['sales_rate_category'], qcut_label = pd.qcut(
    data['sales_rate'],
    5,
    retbins=True
)


# we generate a color palette with Seaborn.color_palette()
pal = sns.color_palette(
    palette='coolwarm',
    n_colors=12
)

# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with
# 'palette'
g = sns.FacetGrid(
    data,
    row='sales_rate_category',
    hue='sales_rate_category',
    aspect=15,
    height=0.75,
    palette=pal
)

# then we add the densities kdeplots for each month
g.map(
    sns.histplot, 'nearby_num_of_remaining_units',
    bw_adjust=1,
    clip_on=False,
    fill=True,
    alpha=1,
    linewidth=1.5
)

# here we add a white line that represents the contour of each kdeplot
g.map(
    sns.histplot, 'nearby_num_of_remaining_units',
    bw_adjust=1,
    clip_on=False,
    color="w",
    lw=2
)

# here we add a horizontal line for each plot
g.map(
    plt.axhline,
    y=0,
    lw=2,
    clip_on=False
)

# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(
        -15,
        0.02,
        month_dict[i + 1],
        fontweight='bold', fontsize=15,
        color=ax.lines[-1].get_color()
    )

# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=-0.3)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel('Temperature in degree Celsius', fontweight='bold', fontsize=15)
g.fig.suptitle(
    'Daily average temperature in Seattle per month',
    ha='right',
    fontsize=20,
    fontweight=20
)