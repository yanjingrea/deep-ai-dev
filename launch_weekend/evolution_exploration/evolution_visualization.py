import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from constants.redshift import query_data
from constants.utils import COLOR_SCALE, get_output_dir
from launch_weekend.scatter_plot import scatter_plot_with_reg_and_label

output_dir = get_output_dir(__file__)

with open(
        'evolution_data.sql',
        'r'
) as sql_file:
    sql_script = sql_file.read()

data = query_data(sql_script)

data = data.sort_values('launch_date')
comparable_projects = data['ref_projects'].unique()

initial_projects = []
clusters_dict = {}
sources = []

for idx, row in data.iterrows():

    project = row['project_display_name']
    ref_project = row['ref_projects']

    if (ref_project not in comparable_projects) or (ref_project is None):
        initial_projects += [project]
        clusters_dict[project] = [project]
        sources += [project]

    else:

        for k, v in clusters_dict.items():

            if ref_project in v:
                clusters_dict[k] += [project]
                sources += [k]
                # break

data['launch_year'] = data['launch_date'].apply(lambda a: int(a[:4]))
data['launch_date'] = pd.to_datetime(data['launch_date'])
data['ref_launch_date'] = pd.to_datetime(data['ref_launch_date'])
data['initial_projects'] = sources

test_data = data[(data['launch_year'] >= 2020) & (~data['average_launch_psf'].isna())]

y_pred = []
for idx, row in test_data.iterrows():

    init_proj = row['initial_projects']

    time_mask = (data['launch_date'] < row['launch_date'])
    group_mask = (data['initial_projects'] == init_proj)
    region_mask = (data['region_group'] == row['region_group'])

    clusters_data = data[time_mask & group_mask]

    if clusters_data.empty:
        clusters_data = data[time_mask & region_mask]

        if clusters_data.empty:
            clusters_data = data[time_mask]

    if False:
        if len(clusters_data) <= 3:
            clusters_data = data[time_mask & region_mask]

            if len(clusters_data) <= 3:
                clusters_data = data[time_mask]

        clusters_data = clusters_data.dropna(
         subset=['sales', 'average_launch_psf', 'num_of_units', 'launch_year']
        )

        Q = clusters_data['sales']
        P = clusters_data['average_launch_psf']
        S = clusters_data['num_of_units']
        T = clusters_data['launch_year']

        X = pd.DataFrame(
            dict(
                P=P,
                S=S
            )
        )

        y = np.log(Q)

        model = LinearRegression(fit_intercept=True).fit(X, y)

        pred_sales = np.exp(
            model.predict(
                pd.DataFrame(
                    dict(
                        P=[row['average_launch_psf']],
                        S=row['num_of_units']
                    )
                )
            )
        )

        pred_sales = np.clip(pred_sales, 0, row['num_of_units'])
        pred_sales_rates = pred_sales/row['num_of_units']
        y_pred += [pred_sales_rates[0]]
    else:
        pred_sales_rates = clusters_data['sales_rate'].mean()
        # pred_sales_rates = np.average(
        #     clusters_data['sales_rate'],
        #     weights=np.log(clusters_data['num_of_units']
        #     )
        # )
        y_pred += [pred_sales_rates]


test_data['pred_sales_rate'] = y_pred
test_data['pred_sales'] = test_data['pred_sales_rate'] * test_data['num_of_units']

pred = test_data['pred_sales'].values
true = test_data['sales'].values


def calculate_error(q_pred, q_true):

    error_to_sales = pd.Series(q_pred[q_true != 0] / q_true[q_true != 0] - 1).abs()
    print(f'mean absolute percentage error: {error_to_sales.mean() * 100 :.2f}%')
    print(f'median absolute percentage error: {error_to_sales.median() * 100 :.2f}%')

    error_to_stock = pd.Series(np.abs(q_pred - q_true) / test_data['num_of_units'])
    print(f'mean absolute percentage of stock error: {error_to_stock.mean() * 100 :.2f}%')
    print(f'median absolute percentage of stock error: {error_to_stock.median() * 100 :.2f}%')

    interval = np.append(np.arange(0.025, 0.125, 0.025), 0.2)

    sample_size = len(q_pred)

    if sample_size > 0:

        print('Error compared to Sales:')
        for t in interval:
            correct_rate = len(error_to_sales[error_to_sales <= t]) / sample_size
            print(f'correct rate (error <= {t * 100: .0f}%): {correct_rate * 100: .2f}%')

        print(f'-' * 20)
        print('Error compared to Stock:')
        for t in interval:
            correct_rate = len(error_to_stock[error_to_stock <= t]) / sample_size
            print(f'correct rate (error <= {t * 100: .0f}%): {correct_rate * 100: .2f}%')


calculate_error(pred, true)

fig, ax = scatter_plot_with_reg_and_label(
    data=test_data.reset_index(),
    x_col='sales_rate',
    y_col='pred_sales_rate',
    label_col='num_of_units',
    n_bins=5
)

degree_45 = np.linspace(0, 1 + 1 / 50, 50)

ax.plot(
    degree_45,
    degree_45,
    color='red',
    alpha=0.5,
    linestyle='dashed'
)

ax.fill_between(
    x=degree_45,
    y1=degree_45 + 0.1,
    y2=degree_45 - 0.1,
    color='red',
    alpha=0.2
)

colors = {
    y: c
    for y, c in zip(
        np.sort(data['launch_year'].unique()),
        COLOR_SCALE
    )
}


def plot_clusters(dataset):

    x = dataset['average_launch_psf']
    y = dataset['sales_rate']

    if len(x) == 1:
        return None

    local_color = dataset['launch_year'].apply(lambda a: colors[a])

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        x,
        y,
        edgecolors=COLOR_SCALE[-1],
        s=80,
        color=local_color,
        alpha=0.5,
        zorder=10
    )

    for local_idx, local_row in dataset.iterrows():

        ref_x = local_row['ref_launch_psf']
        ref_y = local_row['ref_sales_rate']

        proj_x = local_row['launch_psf']
        proj_y = local_row['sales_rate']

        ax.text(
            proj_x * 1.001,
            proj_y * 1.001,
            local_row['project_display_name']  # + f'\nprice: {proj_x: .0f}' + f'\nsales rate: {proj_y * 100: .1f}%'
        )

        if local_row['ref_projects'] is None:
            continue

        adjust_coef = 0.9

        dx = (proj_x - ref_x) * adjust_coef
        dy = (proj_y - ref_y) * adjust_coef

        ax.arrow(
            x=ref_x,
            y=ref_y,
            dx=dx,
            dy=dy,
            color=colors[local_row['launch_year']],
            alpha=0.8,
            head_width=0.01,  # arrow head width
            head_length=1.5
            # arrow head length
        )

    fig.savefig(
        output_dir + 'sequence_scatter_' + dataset['initial_projects'].iloc[0] + '.png', dpi=300
    )

    print()

# data.groupby('initial_projects').apply(plot_clusters)
