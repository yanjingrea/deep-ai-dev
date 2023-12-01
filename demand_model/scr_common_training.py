import pandas as pd
from demand_model.cls_comparable_curve_model import ComparableDemandModel

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