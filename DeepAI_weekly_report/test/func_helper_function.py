import pickle
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from DeepAI_weekly_report.test.cls_paths_collections import PathsCollections
from constants.utils import NatureD, NatureL, print_in_green_bg
from demand_curve_main.cls_comparable_data import BaseCMData
from demand_curve_main.cls_comparable_model import ComparableDemandModel

from demand_curve_main.scr_coef import query_adjust_coef
from DeepAI_weekly_report.scr_get_paths import (
    dev_data_dir, td, dev_figure_dir, report_dir, dev_res_dir
)


# -------------------------------------------------


def get_rebased_project_data(
    *,
    demand_model: ComparableDemandModel,
    project_name,
    num_of_bedroom
):
    available_projects = demand_model.available_projects
    available_projects.set_index(['project_name', 'num_of_bedrooms'], inplace=True)

    try:
        trans_status = available_projects.loc[(project_name, num_of_bedroom), 'with_trans']
    except KeyError:
        return pd.DataFrame()

    if trans_status:
        data_source = demand_model.data
    else:
        data_source = demand_model.forecasting_data

    project_data = data_source[
        (data_source['project_name'] == project_name) &
        (data_source['num_of_bedrooms'] == num_of_bedroom)
        ]

    return project_data


def get_adjusted_project_data(
    *,
    demand_model: ComparableDemandModel,
    project_name,
    num_of_bedroom
):
    rebased_project_data = get_rebased_project_data(
        demand_model=demand_model,
        project_name=project_name,
        num_of_bedroom=num_of_bedroom
    )

    if rebased_project_data.empty:
        return rebased_project_data

    coef_to_multiply = query_adjust_coef(rebased_project_data)

    adjusted_project_data = rebased_project_data.copy()
    adjusted_project_data['price'] = adjusted_project_data['price'] * coef_to_multiply

    return adjusted_project_data


normalize_bed_num = lambda bedroom: str(int(bedroom)) if bedroom != -1 else 'all'


def get_single_project_report_results(
    *,
    data_model: BaseCMData,
    demand_model: ComparableDemandModel,
    project_name,
    num_of_bedroom,
    image_paths,
    test_results=None
):
    adjusted_project_data = get_adjusted_project_data(
        demand_model=demand_model,
        project_name=project_name,
        num_of_bedroom=num_of_bedroom
    ).copy().reset_index(drop=True)

    if adjusted_project_data.empty:
        return None, image_paths

    linear_model, adjusted_training_data = demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data.dw_project_id.iloc[0],
        num_of_bedroom=num_of_bedroom
    )

    if linear_model is None:
        print_in_green_bg(f'Failed to get a demand curve for {project_name} {num_of_bedroom}-bedrooms.')

        return test_results, image_paths

    if test_results is not None:
        adjusted_project_data['pred_sales'] = linear_model.predict(adjusted_project_data)

        test_results = pd.concat(
            [
                test_results,
                adjusted_project_data[
                    [
                        'project_name',
                        'num_of_bedrooms',
                        'num_of_units',
                        'launching_period',
                        'num_of_remaining_units',
                        demand_model.quantity,
                        'pred_sales'
                    ]
                ]
            ],
            ignore_index=True
        )

    adjusted_training_data.to_csv(
        dev_data_dir + f'{project_name} {int(num_of_bedroom)}-bedroom.csv'
    )

    image_paths = model_to_demand_curve(
        data_model=data_model,
        linear_model=linear_model,
        adjusted_project_data=adjusted_project_data,
        adjusted_training_data=adjusted_training_data,
        image_paths=image_paths
    )

    return test_results, image_paths


def model_to_demand_curve(
    *,
    data_model: BaseCMData,
    linear_model,
    adjusted_project_data,
    adjusted_training_data,
    image_paths,
    concat=True
):

    project_name = adjusted_project_data['project_name'].iloc[0]
    num_of_bedroom = adjusted_project_data['num_of_bedrooms'].iloc[0]

    last_date = adjusted_project_data['transaction_month_end'].iloc[-1]

    if concat:

        if last_date > pd.to_datetime(td.replace(day=1)):

            adjusted_project_data = pd.concat(
                [
                    adjusted_project_data,
                    data_model.prepare_forecast_demand_curve_data(adjusted_project_data)
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
        adjusted_project_data[data_model.price].min() * 0.8,
        adjusted_project_data[data_model.price].max() * 1.2
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

        if adjusted_project_data[data_model.quantity].mean() > 0:
            ax.hlines(
                y=adjusted_project_data['num_of_remaining_units'].iloc[-1],
                **common_params,
                colors='lightgrey',
                label='num_of_remaining_units'.replace('_', ' ')
            )

            scatterplot_data = adjusted_project_data.iloc[:-1]

            n_display = 5
            if len(scatterplot_data) >= n_display:
                scatterplot_data = scatterplot_data.sample(n_display)

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
                x=scatterplot_data[data_model.price],
                y=scatterplot_data[data_model.quantity],
                **scatter_params
            )

        plt.legend()
        if mode == 'dev':
            manual_label = ''
            title = f'{project_name} {normalize_bed_num(num_of_bedroom) + "-bedroom"}'
            ax.set_title(f'{title}\n{manual_label}')
            plt.savefig(dev_figure_dir + f'dev-{title}{manual_label}.png', dpi=300)
            plt.close()

        else:
            title = f'{project_name} {normalize_bed_num(num_of_bedroom) + "-bedroom"}'
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

    return image_paths


def weekly_test_demand_model(
    *,
    data_model: BaseCMData,
    demand_model: ComparableDemandModel,
    image_paths: list,
    min_stock=10,
    min_proj_size=50,
    label: Literal['condo', 'ec'],
    test_projects=None
) -> pd.DataFrame:

    data = demand_model.data
    forecast_data = demand_model.forecasting_data

    if test_projects is None:
        max_period = data.groupby('dw_project_id')['launching_period'].transform('max')
        test_data = data[
            (data['num_of_remaining_units'] - data[demand_model.quantity] >= min_stock) &
            (data['launching_period'] == max_period) &
            (data['proj_num_of_units'] >= min_proj_size)
            ]
    else:
        test_data = data[data.isin(test_projects)]

    # forecast data
    # --------------------------------------------------------------------------------------------

    manual_input_curves = pd.DataFrame(image_paths)
    manual_input_projects = manual_input_curves['project_name'].unique()

    if forecast_data is not None:
        for idx in np.arange(len(forecast_data)):
            temp_row = forecast_data.iloc[[idx]].copy()
            temp_name = temp_row.project_name.iloc[0]
            temp_bed = temp_row.num_of_bedrooms.iloc[0]

            if temp_name in manual_input_projects:
                continue

            _, image_paths = get_single_project_report_results(
                data_model=data_model,
                demand_model=demand_model,
                project_name=temp_name,
                num_of_bedroom=temp_bed,
                image_paths=image_paths
            )

    unique_test_data = test_data[
        ['dw_project_id', 'project_name', 'num_of_bedrooms']].drop_duplicates().sort_values(
        by=['project_name', 'num_of_bedrooms']
    ).reset_index()

    test_results = pd.DataFrame()
    for idx, row in unique_test_data.iterrows():

        temp_name = row.project_name
        temp_bed = row.num_of_bedrooms

        test_results, image_paths = get_single_project_report_results(
            data_model=data_model,
            demand_model=demand_model,
            project_name=temp_name,
            num_of_bedroom=temp_bed,
            image_paths=image_paths,
            test_results=test_results
        )

    test_results_des = dev_res_dir + f'{label}_test_results.plk'
    pickle.dump(test_results, open(test_results_des, 'wb'))

    paths_df = pd.DataFrame(image_paths)

    return paths_df
