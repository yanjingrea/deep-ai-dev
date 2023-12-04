import os
from dataclasses import dataclass
from datetime import datetime
from os.path import dirname, realpath
from typing import Literal

import seaborn as sns
from matplotlib import pyplot as plt

from constants.utils import NatureD, NatureL
from demand_model.dev_project_level.scr_training import *

# -------------------------------------------------

td = datetime.today()
dev_dir = dirname(realpath(__file__)) + f'/output/dev_project_level/{td.date()}/'
dev_figure_dir = dev_dir + f'figures/'
dev_data_dir = dev_dir + f'data/'
dev_3d_dir = dev_dir + f'3d/'

report_dir = f'/Users/wuyanjing/PycharmProjects/presentation/src/images/{td.date()}/'

for mode, directory_path in zip(
        ['dev_figures', 'dev_table', 'report', 'dev_3d'],
        [dev_figure_dir, dev_data_dir, report_dir, dev_3d_dir]
):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"{mode} directory created: {directory_path}")
        except OSError as error:
            print(f"Error creating {mode} directory: {error}")
    else:
        print(f"locate {mode} directory: {directory_path}")


# -------------------------------------------------
@dataclass
class PathsCollections:
    project_name: str
    paths: str


def get_report_results(
        project_name,
        image_paths,
        test_results=None
):
    adjusted_project_data = get_adjusted_project_data(
        project_name
    ).copy().reset_index(drop=True)

    if adjusted_project_data.empty:
        return None, None

    linear_model, adjusted_training_data = demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data.dw_project_id.iloc[0]
    )

    if test_results is not None:
        adjusted_project_data['pred_sales'] = linear_model.predict(adjusted_project_data)

        test_results = pd.concat(
            [
                test_results,
                adjusted_project_data[
                    [
                        'project_name',
                        'proj_num_of_units',
                        'launching_period',
                        'num_of_remaining_units',
                        quantity,
                        'pred_sales'
                    ]
                ]
            ],
            ignore_index=True
        )

    adjusted_training_data.to_csv(
        dev_data_dir + f'{project_name}.csv'
    )

    adjusted_project_data = pd.concat(
        [
            adjusted_project_data,
            demand_model.prepare_forecast_demand_curve_data(adjusted_project_data)
        ],
        ignore_index=True
    )

    def normalize_date(date):
        return str(date.year) + ' ' + date.month_name()[:3]

    adjusted_project_data['date_to_display'] = adjusted_project_data.apply(
        lambda row: f"from {normalize_date(row['transaction_month'])} "
                    f"to {normalize_date(row['transaction_month_end'])}",
        axis=1
    )

    price_range = (
        adjusted_project_data[price].min() * 0.8,
        adjusted_project_data[price].max() * 1.2
    )

    def plot_first_last_curve(mode: Literal['dev', 'report'], image_paths):

        fig, ax = plt.subplots(figsize=(8, 6))

        for idx, row in adjusted_project_data.iterrows():

            if (idx != 0) and (idx != len(adjusted_project_data) - 1):
                continue

            temp_period = row['launching_period']
            temp_display_date = row['date_to_display']

            fig, ax = linear_model.extract_2d_demand_curve(
                adjusted_project_data.iloc[[idx]],
                launching_period=temp_period,
                price_range=price_range,
                fig_format='plt'
            ).plot(
                fig=fig,
                ax=ax,
                color=NatureD['blue'] if idx == 0 else NatureL['blue'],
                label=f'Demand Curve \n{temp_display_date}'
            )

        common_params = dict(
            xmin=price_range[0],
            xmax=price_range[1],
            alpha=0.7,
            linestyles='--',
        )
        ax.hlines(
            y=adjusted_project_data['proj_num_of_units'].iloc[-1],
            **common_params,
            colors='grey',
            label='proj_num_of_units'.replace('_', ' ')
        )

        scatter_params = dict(
            s=80,
            alpha=0.5,
            ax=ax
        )

        if adjusted_project_data[quantity].mean() > 0:
            ax.hlines(
                y=adjusted_project_data['num_of_remaining_units'].iloc[-1],
                **common_params,
                colors='lightgrey',
                label='num_of_remaining_units'.replace('_', ' ')
            )

            scatterplot_data = adjusted_project_data.iloc[:-1]
            scatter_params['hue'] = scatterplot_data['date_to_display']
        else:
            scatterplot_data = pd.DataFrame()

        if mode == 'dev':
            scatterplot_data = pd.concat(
                [scatterplot_data, adjusted_training_data],
                ignore_index=True
            )
            scatter_params['hue'] = scatterplot_data['project_name']

        if not scatterplot_data.empty:
            sns.scatterplot(
                x=scatterplot_data[price],
                y=scatterplot_data[quantity],
                **scatter_params
            )

        plt.legend()
        if mode == 'dev':
            manual_label = ''
            title = f'{project_name}'
            ax.set_title(f'{title}\n{manual_label}')
            plt.savefig(dev_figure_dir + f'dev-{title}{manual_label}.png', dpi=300)
            plt.close()

        else:
            title = f'{project_name}'
            ax.set_title(f'{title}')
            report_path = title.replace('-', '_').replace(' ', '_')
            plt.savefig(report_dir + f"{report_path}.png", dpi=300)
            plt.close()

            image_paths += [
                PathsCollections(
                    project_name=project_name,
                    paths=report_path
                )
            ]

        return fig, ax

    for mode in ['dev', 'report']:
        plot_first_last_curve(mode, image_paths)

    return test_results, image_paths
