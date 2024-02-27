from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from constants.utils import NatureD, NatureL
from demand_curve_main.scr_coef import query_adjust_coef
from demand_curve_condo.scr_common_training import comparable_demand_model, bedroom_data
from DeepAI_weekly_report.test.func_helper_function import (
    dev_figure_dir, report_dir, dev_data_dir,
    normalize_bed_num
)
from DeepAI_weekly_report.test.cls_paths_collections import PathsCollections

manual_data = pd.read_csv(
    'manual_input/manual_input.csv'
)

manual_data['transaction_month'] = pd.to_datetime(manual_data['transaction_month'])
manual_data = bedroom_data.calculate_launching_period(manual_data)

# manual_data['price'] = manual_data['price'] / query_adjust_coef(manual_data)

target_projects = [
    'Lentoria',
    'The Hill @ One North'
]

temp_comparable_demand_model = comparable_demand_model.copy()
temp_comparable_demand_model.__setattr__('features', ['price'])

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
            # include_ids = [
            #     '5db64f51ff8b9b30a2270dd44cc15845',  # Lentor Morden
            #     'c3532fd0b2e9e84d9ad9f36c8f675230',  # Lentor Hill Residences
            # ]

            exclude_ids = [
                'c3532fd0b2e9e84d9ad9f36c8f675230',
                '5db64f51ff8b9b30a2270dd44cc15845',
                # '7a0eaa8196d9676a189aa4a7fbabc7e5',
                # 'dc092f3963693f4ebea31bb85ec8cf25',
                # '7d1c38c1beac255f481b37ca3328eabb'  # the-botany-at-dairy-farm
            ]

        elif num_of_bedroom == 2:
            max_launching_period = 1
            # include_ids = [
            #     '7d1c38c1beac255f481b37ca3328eabb'  # the-botany-at-dairy-farm
            #     '5db64f51ff8b9b30a2270dd44cc15845',  # Lentor Morden
            #     'c3532fd0b2e9e84d9ad9f36c8f675230',  # Lentor Hill Residences
            # ]
            exclude_ids = [
                # '3c78a223221a2aca191612502c67f7a2' # pasir-ris-8
                'f3d8b3586d5439ab5df454bac2381bc6'  # the-watergardens-at-canberra
            ]

        elif num_of_bedroom == 3:
            max_launching_period = 1

            exclude_ids = [
                'c3532fd0b2e9e84d9ad9f36c8f675230',  # Lentor Hill Residences
            ]

        elif num_of_bedroom == 4:
            include_ids = [
                '5db64f51ff8b9b30a2270dd44cc15845',  # Lentor Morden
                'c3532fd0b2e9e84d9ad9f36c8f675230',  # Lentor Hill Residences
            ]

            # exclude_ids = [
            #     'f3d8b3586d5439ab5df454bac2381bc6',
            #     'dd7046f546db1ba2253870b151d89f61',
            #     '7d1c38c1beac255f481b37ca3328eabb'
            # ]
            max_launching_period = 3
        else:
            max_launching_period = 3

    elif project_name == 'The Hill @ One North':

        if num_of_bedroom == 2:
            max_launching_period = 1
            # exclude_ids = [
            #     '6c7a8743f25477e921c882cd04bc0470',
            #     '3c78a223221a2aca191612502c67f7a2',
            #     'c3532fd0b2e9e84d9ad9f36c8f675230',
            # ]
        elif num_of_bedroom == 3:
            max_launching_period = 1
            # exclude_ids = [
            #     '6c7a8743f25477e921c882cd04bc0470',
            #     '2abe67cd206094d32053d93764cc9a65',
            #     'd021f3b74360d1019a5eabdc507e0a58'
            # ]
        else:
            max_launching_period = 3
            exclude_ids = [
                '6c7a8743f25477e921c882cd04bc0470',
                '2abe67cd206094d32053d93764cc9a65',
                'd021f3b74360d1019a5eabdc507e0a58'
            ]

    if project_name in ['Lentoria', 'The Hill @ One North']:
        cdm = temp_comparable_demand_model
    else:
        cdm = comparable_demand_model

    linear_model, adjusted_training_data = cdm.fit_project_room_demand_model(
        project_id=adjusted_project_data['dw_project_id'].iloc[0],
        project_data=rebased_project_data,
        num_of_bedroom=num_of_bedroom,
        include_ids=include_ids,
        max_launching_period=max_launching_period,
        exclude_ids=exclude_ids
    )

    adjusted_training_data.to_csv(
        dev_data_dir + f'{project_name} {num_of_bedroom}-bedroom.csv', index=False
    )


    def normalize_date(date):
        return str(date.year) + ' ' + date.month_name()[:3]


    adjusted_project_data['date_to_display'] = adjusted_project_data.apply(
        lambda row: f"from {normalize_date(row['transaction_month'])} "
                    f"to {normalize_date(row['transaction_month_end'])}",
        axis=1
    )

    if False:
        adjusted_project_data['nth_launch'] = adjusted_training_data['nth_launch'].max() + 1
        cy = 2027 if project_name != 'The Hill @ One North' else 2026
        lc = 0 if project_name != 'The Hill @ One North' else 6
        adjusted_project_data['completion_year'] = cy
        adjusted_project_data['num_of_nearby_launched_condo_proj_1000m'] = lc

    # price_range = (
    #     adjusted_project_data['price'].min() * 0.8,
    #     adjusted_project_data['price'].max() * 1.2
    # )

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
