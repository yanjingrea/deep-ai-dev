import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots

from demand_model.scr_common_training import *


# -------------------------------------------------

def get_linear_model_and_training_data(
        project_name,
        num_of_bedroom
):
    adjusted_project_data = get_adjusted_project_data(
        project_name,
        num_of_bedroom
    ).copy().reset_index()

    if adjusted_project_data.empty:
        return None

    linear_model, training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data.dw_project_id.iloc[0],
        num_of_bedroom=num_of_bedroom
    )

    return linear_model, training_data


def plot_2d_demand_curve(
        project_name,
        num_of_bedroom,
        dev_mode=False
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
            comparable_demand_model.prepare_forecast_demand_curve_data(
                adjusted_project_data
            )
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

    linear_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
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

        if row['num_of_remaining_units'] <= 0:
            continue

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

        if dev_mode:
            scatter_data = pd.concat(
                [adjusted_project_data.iloc[:-1], adjusted_training_data],
                ignore_index=True
            )

            colors_list = px.colors.qualitative.Set1 + px.colors.qualitative.Light24_r
            training_projects = adjusted_training_data['project_name'].unique()
            colors_map = pd.Series(
                index=training_projects,
                data=colors_list[:len(training_projects)]
            )

            marker_color = scatter_data['project_name'].apply(lambda p: colors_map.loc[p])

        else:
            scatter_data = adjusted_project_data.iloc[:-1]
            marker_color = 'grey'

        fig.add_trace(
            go.Scatter(
                x=scatter_data[price],
                y=scatter_data[quantity],
                mode='markers',
                marker_color=marker_color,
                marker=dict(
                    opacity=0.5,
                    size=16
                ),
                hovertext=scatter_data['project_name'],
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
