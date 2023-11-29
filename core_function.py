import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib import cm
from plotly.subplots import make_subplots

from constants.utils import NatureD
from demand_model.comparable_curve_model import ComparableDemandModel

comparable_demand_model = ComparableDemandModel()

available_projects = comparable_demand_model.available_projects
available_projects.set_index(['project_name', 'num_of_bedrooms'], inplace=True)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity


def get_rebased_project_data(
        project_name,
        num_of_bedroom
):
    try:
        trans_status = available_projects.loc[(project_name, num_of_bedroom), 'with_trans']
    except KeyError:
        return pd.DataFrame()

    if trans_status:
        data_source = comparable_demand_model.data
    else:
        data_source = comparable_demand_model.forecasting_data

    project_data = data_source[
        (data_source['project_name'] == project_name) &
        (data_source['num_of_bedrooms'] == num_of_bedroom)
        ]

    return project_data


def get_adjusted_project_data(
        project_name,
        num_of_bedroom
):
    rebased_project_data = get_rebased_project_data(
        project_name,
        num_of_bedroom
    )

    if rebased_project_data.empty:
        return rebased_project_data

    coef_to_multiply = comparable_demand_model.query_adjust_coef(rebased_project_data)

    adjusted_project_data = rebased_project_data.copy()
    adjusted_project_data[price] = adjusted_project_data[price] * coef_to_multiply

    return adjusted_project_data


# todo: currently just min period
def plot_2d_demand_curve(
        project_name,
        num_of_bedroom
):
    adjusted_project_data = get_adjusted_project_data(
        project_name,
        num_of_bedroom
    ).copy().reset_index()

    if adjusted_project_data.empty:
        return None

    adjusted_project_data = pd.concat(
        [
            adjusted_project_data,
            comparable_demand_model.prepare_forecast_demand_curve_data(adjusted_project_data)
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

    linear_model = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data.dw_project_id.iloc[0],
        num_of_bedroom=num_of_bedroom
    )

    color_scale = px.colors.qualitative.Set2
    color_map = {
        category: color_scale[i % len(color_scale)]
        for i, category in enumerate(adjusted_project_data.index)
    }

    facecolors = adjusted_project_data.index.map(color_map)

    fig = make_subplots()

    for idx, row in adjusted_project_data.iterrows():

        temp_period = row['launching_period']
        temp_display_date = row['date_to_display']

        params = dict(
            name=f'demand curve\n {temp_display_date} {"(forecast)" if idx == len(adjusted_project_data) - 1 else ""}',
            color=facecolors[idx]
        )

        if (idx > 0) and (idx < len(adjusted_project_data) - 1):
            params['visible'] = 'legendonly'

        fig = linear_model.extract_2d_demand_curve(
            adjusted_project_data.iloc[[idx]],
            launching_period=temp_period,
            price_range=price_range
        ).plot(
            fig=fig,
            **params
        )

    if adjusted_project_data[quantity].mean() != 0:

        scatter_data = adjusted_project_data.iloc[:-1]

        fig.add_trace(
            go.Scatter(
                x=scatter_data[price],
                y=scatter_data[quantity],
                mode='markers',
                marker_color='grey',
                marker=dict(
                    opacity=0.3,
                    size=16
                ),
                text=scatter_data['date_to_display'],
                textposition="top right",
                name='transactions'
            )
        )
        fig.update_traces(textposition="top right")
        fig.update_layout(
            title=f'{project_name} {int(num_of_bedroom)}-bedrooms',
            autosize=False,
            width=1200,
            height=600
        )


    return fig


def display_project(project):
    return project


# todo: decide what to display
def display_project_summary(project, df):
    if project == ALL_PROJECTS_NAMES:
        return ""
    else:
        df = (
            df.loc[lambda x: x.state == project]
            .reset_index()
            .assign(
                total_migration=lambda x: x.inbound_migration
                                          + x.outbound_migration
                                          + x.within_state_migration
            )
        )

        total_migration = df.total_migration[0]

        inbound_migration = df.inbound_migration[0]
        inbound_pct_total = f"{round(inbound_migration / total_migration * 100, 1)}%"
        outbound_migration = df.outbound_migration[0]
        outbound_pct_total = f"{round(outbound_migration / total_migration * 100, 1)}%"
        within_state_migration = df.within_state_migration[0]
        within_state_pct_total = (
            f"{round(within_state_migration / total_migration * 100, 1)}%"
        )

        return f"""
        **Inbound Migration Total (%):** 

        {"{:,}".format(inbound_migration)} ({inbound_pct_total})

        **Outbound Migration Total (%):** 

        {"{:,}".format(outbound_migration)} ({outbound_pct_total}) 

        **Within State Migration Total (%):** 

        {"{:,}".format(within_state_migration)} ({within_state_pct_total})
        """
