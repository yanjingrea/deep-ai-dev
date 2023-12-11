import pandas as pd
from demand_model.dev_project_level.cls_project_demand_model import ProjectDemandModel

demand_model = ProjectDemandModel()
demand_model.__setattr__('max_year_gap', 10)

available_projects = demand_model.available_projects
available_projects.set_index(['project_name'], inplace=True)

price = demand_model.price
quantity = demand_model.quantity


def get_rebased_project_data(project_name):
    try:
        trans_status = available_projects.loc[project_name, 'with_trans']
    except KeyError:
        return pd.DataFrame()

    if trans_status:
        data_source = demand_model.data
    else:
        data_source = demand_model.forecasting_data

    project_data = data_source[
        (data_source['project_name'] == project_name)
    ]

    return project_data


def get_adjusted_project_data(project_name):
    rebased_project_data = get_rebased_project_data(project_name)

    if rebased_project_data.empty:
        return rebased_project_data

    coef_to_multiply = demand_model.query_adjust_coef(rebased_project_data)

    adjusted_project_data = rebased_project_data.copy()
    adjusted_project_data[price] = adjusted_project_data[price] * coef_to_multiply

    return adjusted_project_data
