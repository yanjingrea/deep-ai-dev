import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots

from demand_model.scr_common_training import *


# -------------------------------------------------

def plot_ui_demand_curve(
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

    linear_model, _ = comparable_demand_model.fit_project_room_demand_model(
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
