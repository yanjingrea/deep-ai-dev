import plotly.graph_objects as go

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
    trans_status = available_projects.loc[(project_name, num_of_bedroom), 'with_trans']
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
    ).copy()

    if adjusted_project_data.empty:
        return None

    linear_model = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data.dw_project_id.iloc[0],
        num_of_bedroom=num_of_bedroom
    )
    min_period = adjusted_project_data['launching_period'].min()
    min_period_data = adjusted_project_data.iloc[[0]].copy()

    fig = linear_model.extract_2d_demand_curve(
        min_period_data,
        launching_period=min_period
    ).plot(name=f'demand curve')

    fig.update_layout()

    if adjusted_project_data[quantity].mean() != 0:
        colors = adjusted_project_data['launching_period'].apply(
            lambda a: NatureD['red'] if a == min_period else NatureD['orange']
        )

        date_to_display = adjusted_project_data.apply(
            lambda row: row['transaction_month'].month_name,
            axis=1
        )

        fig.add_trace(
            go.Scatter(
                x=adjusted_project_data[price],
                y=adjusted_project_data[quantity],
                mode='markers',
                marker_color=colors,
                text=adjusted_project_data['transaction_month'].apply(lambda a: a.date()),
                marker=dict(
                    opacity=0.6,
                    size=16
                ),
                name='transactions'
            )
        )
        fig.update_traces(textposition="bottom right")
        fig.update_layout(
            title=f'{project_name} {int(num_of_bedroom)}-bedrooms'
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
