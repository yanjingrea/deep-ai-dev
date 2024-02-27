import pickle
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DeepAI_weekly_report.scr_get_paths import dev_res_dir, report_dir, td
from constants.utils import set_plot_format

set_plot_format(plt)

test_results = pd.DataFrame()
for group in ['ec', 'condo']:

    test_results_des = dev_res_dir + f'{group}_test_results.plk'
    temp_test_results = pickle.load(open(test_results_des, 'rb'))
    test_results = pd.concat([test_results, temp_test_results], ignore_index=True)

proj_level_results = test_results.groupby(['project_name', 'launching_period'])[
    [
        'num_of_units',
        'num_of_remaining_units',
        'sales',
        'pred_sales'
    ]
].sum().reset_index()
proj_level_results['num_of_bedrooms'] = 'all'

metrics_df = pd.concat([test_results, proj_level_results])
metrics_df['error_to_sales'] = metrics_df['pred_sales'] / metrics_df['sales'] - 1
metrics_df['error_to_stock'] = (
        (metrics_df['pred_sales'] - metrics_df['sales']) / metrics_df['num_of_units']
)
metrics_df['period_label'] = metrics_df['launching_period'].apply(
    lambda a: 'first' if a <= 3 else 'rest'
)

n_sample = metrics_df.groupby(['num_of_bedrooms'])['project_name'].count()

GREY10 = "#1a1a1a"
GREY30 = "#4d4d4d"
GREY40 = "#666666"
GREY50 = "#7f7f7f"
GREY60 = "#999999"
GREY75 = "#bfbfbf"
GREY91 = "#e8e8e8"
GREY98 = "#fafafa"

COLOR_SCALE = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    GREY50
]


def calculate_correct_rate(metric: Literal['error_to_sales', 'error_to_stock']):
    correct_rate_data = pd.DataFrame()

    if metric == 'error_to_sales':
        q = 0.975
        fix = np.arange(0.2, 1.2, 0.2)
    else:
        q = 1
        fix = np.arange(0.02, 0.12, 0.02)

    series = metrics_df[metric]

    std = series[series <= series.quantile(q)].std()

    percents = np.append(fix, std)

    for temp_idx, confidence_interval in enumerate(percents):

        n_correct = metrics_df[
            (metrics_df[metric] <= confidence_interval)
        ].groupby(['num_of_bedrooms'])['project_name'].count()

        series_name = f'error <= {int(confidence_interval * 100)}%' \
            if temp_idx != len(percents) - 1 \
            else f'error <= {confidence_interval * 100: .1f}% (std)'

        correct_rate = (n_correct / n_sample).rename(series_name)

        correct_rate_data = pd.concat([correct_rate_data, correct_rate], axis=1)

    correct_rate_data['num_of_bedrooms'] = correct_rate_data.index.to_series().apply(
        lambda a: f'{int(a) if isinstance(a, float) else "all"}-bedroom'
    )

    return correct_rate_data.reset_index(drop=True), percents


def add_label(x_value, y_value, fontsize, ax, pad):

    if isinstance(x_value, str):
        xy_x = x_value
    else:
        xy_x = x_value + pad

    ax.annotate(
        f"{y_value * 100 :.1f}%",
        xy=(xy_x, y_value + pad),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=12
    )


def plot_correct_rate(
    metric: Literal['error_to_sales', 'error_to_stock'],
):
    correct_rate_data, percents = calculate_correct_rate(metric)

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, row in correct_rate_data.iterrows():

        x = percents
        y = row.iloc[:-1].values

        color = COLOR_SCALE[int(idx)]

        ax.plot(x, y, color=color, lw=2.4, zorder=10, label=row['num_of_bedrooms'])
        ax.scatter(x, y, fc="w", ec=color, s=60, lw=2.4, zorder=12)

        add_label(x[0], y[0], 10, ax, pad=0.008)
        add_label(x[-1], y[-1], 10, ax, pad=-0.008)

    ax.set_xticks(percents)
    ax.set_xticklabels(
        correct_rate_data.columns[:-1],
        fontsize=9
    )
    ax.tick_params(bottom=False)
    ax.legend()

    title = f'{metric} error distribution summary'
    ax.set_title(title)
    report_path = f'{title.replace(" ", "_")}.png'
    plt.savefig(report_dir + report_path, dpi=300)

    return title.replace(" ", "_")


def save_historical_data(
    sum_table,
    data_path
):

    his_path = (
        f'output/{data_path}_historical_error.csv'
    )
    his_data = pd.read_csv(his_path, header=0)
    # his_data['report date'] = pd.to_datetime(his_data['report date'], dayfirst=True).dropna()
    # his_data['report date'] = pd.to_datetime(his_data['report date']).dropna()

    report_date = f'{pd.to_datetime(td).date()}'
    to_record_row = [report_date]
    for pct in ['mean', '25%', '50%', '75%']:
        to_record_row += sum_table.loc[pct][0:].apply(lambda s: s.replace(r'\%', '%')).reset_index(
            drop=True
        ).to_list()

    if his_data['report date'].iloc[-1] == report_date:
        his_data.iloc[-1, :] = to_record_row
    else:
        his_data.loc[-1] = to_record_row
    his_data.reset_index(inplace=True, drop=True)

    if td.isoweekday() == 5:
        his_data.to_csv(his_path, index=False)

    x = his_data['report date']

    his_slide_cols = {}

    for idx, error_type in enumerate(['error to sales', 'error to stock']):

        fig, ax = plt.subplots(figsize=(8, 6))

        for pct, l, color in zip(
                ['', ' 50th'],
                ['mean', 'median'],
                COLOR_SCALE[:2]
        ):

            col = error_type + pct

            y = his_data[col].str[:-1].astype(float)

            ax.plot(x, y, color=color, lw=2.4, zorder=10, label=l)
            ax.scatter(x, y, fc="w", ec=color, s=60, lw=2.4, zorder=12)

            for i in his_data.index:
                ax.annotate(
                    his_data[col][i],
                    xy=(x[i], y[i] + 0.08),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    zorder=12
                )

        ax.set_xticks(x)
        title = f"historical error of {error_type}"
        ax.set_title(title)

        report_path = f'{data_path} {title}.png'.replace(' ', '_')
        plt.legend()
        plt.savefig(report_dir + report_path, dpi=300)

        his_slide_cols[error_type] = report_path

    return his_slide_cols
