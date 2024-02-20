from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from constants.utils import NatureD, NatureL
from demand_curve_main.scr_coef import query_adjust_coef
from demand_curve_condo.scr_common_training import comparable_demand_model, bedroom_data
from DeepAI_weekly_report.test.func_helper_function import (
    dev_figure_dir, report_dir,
    normalize_bed_num
)
from DeepAI_weekly_report.test.cls_paths_collections import PathsCollections

manual_data = pd.read_csv(
    'manual_input/manual_input.csv'
)

manual_data['transaction_month'] = pd.to_datetime(manual_data['transaction_month'])
manual_data = bedroom_data.calculate_launching_period(manual_data)

# manual_data['price'] = manual_data['price'] / query_adjust_coef(manual_data)

target_projects = ['Lentoria', 'The Hill @ One North']

image_paths = []
for idx in np.arange(len(manual_data)):

    if manual_data['project_name'].iloc[idx] not in target_projects:
        continue

    adjusted_project_data = manual_data.iloc[[idx]].copy().reset_index()

    coef = query_adjust_coef(adjusted_project_data)

    rebased_project_data = adjusted_project_data.copy()
    rebased_project_data['price'] = rebased_project_data['price'] / coef

    project_name = adjusted_project_data.project_name.iloc[0]
    num_of_bedroom = adjusted_project_data.num_of_bedrooms.iloc[0]

    include_ids = None
    max_launching_period = None
    exclude_ids = None


    if project_name == 'The Arcady at Boon Keng':
        if num_of_bedroom in [3, 4]:
            include_ids = [
                # '9a9ccc9291a11467ffe12dd8607950ee',
                'b8846c85fc359a8072addec452d2e016'
            ]

            if num_of_bedroom == 4:
                max_launching_period = 5
            else:
                max_launching_period = 12
        else:
            include_ids = [
                '9a9ccc9291a11467ffe12dd8607950ee',
                'b8846c85fc359a8072addec452d2e016'
            ]

            if num_of_bedroom == 2:
                max_launching_period = 3
            else:
                max_launching_period = None

    if project_name == 'Lentoria':

        if num_of_bedroom == 1:
            max_launching_period = 1
            include_ids = [
                '5db64f51ff8b9b30a2270dd44cc15845',
                'c3532fd0b2e9e84d9ad9f36c8f675230',
            ]
        else:
            max_launching_period = 3

    elif project_name == 'The Hill @ One North':

        if num_of_bedroom == 2:
            max_launching_period = 6
            exclude_ids = ['6c7a8743f25477e921c882cd04bc0470']
        elif num_of_bedroom == 3:
            max_launching_period = 1
            exclude_ids = [
                '6c7a8743f25477e921c882cd04bc0470',
                # '2abe67cd206094d32053d93764cc9a65'
            ]
        else:
            max_launching_period = 3



    linear_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data['dw_project_id'].iloc[0],
        project_data=rebased_project_data,
        num_of_bedroom=num_of_bedroom,
        include_ids=include_ids,
        max_launching_period=max_launching_period,
        exclude_ids=exclude_ids
    )


    def normalize_date(date):
        return str(date.year) + ' ' + date.month_name()[:3]


    adjusted_project_data['date_to_display'] = adjusted_project_data.apply(
        lambda row: f"from {normalize_date(row['transaction_month'])} "
                    f"to {normalize_date(row['transaction_month_end'])}",
        axis=1
    )

    price_range = (
        adjusted_project_data['price'].min() * 0.8,
        adjusted_project_data['price'].max() * 1.2
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
            y=adjusted_project_data['num_of_units'].iloc[-1],
            **common_params,
            colors='grey',
            label='num_of_units'.replace('_', ' ')
        )

        scatter_params = dict(
            s=80,
            alpha=0.5,
            ax=ax
        )

        if mode == 'dev':
            scatterplot_data = adjusted_training_data
            scatter_params['hue'] = scatterplot_data['project_name']

            if not scatterplot_data.empty:
                sns.scatterplot(
                    x=scatterplot_data['price'],
                    y=scatterplot_data['sales'],
                    **scatter_params
                )

        plt.legend()
        if mode == 'dev':
            manual_label = ''
            title = f'{project_name} {num_of_bedroom}-bedroom'
            ax.set_title(f'{title}\n{manual_label}')
            plt.savefig(dev_figure_dir + f'dev-{title}{manual_label}.png', dpi=300)
            plt.close()

        else:
            title = f'{project_name} {num_of_bedroom}-bedroom'
            ax.set_title(f'{title}')
            report_path = title.replace('-', '_').replace(' ', '_')
            plt.savefig(report_dir + f"{report_path}.png", dpi=300)
            plt.close()

            image_paths += [
                PathsCollections(
                    project_name=project_name,
                    num_of_bedrooms=normalize_bed_num(num_of_bedroom),
                    paths=report_path
                )
            ]

        return fig, ax


    for mode in ['dev', 'report']:
        plot_first_last_curve(mode, image_paths)
