import pandas as pd

from demand_curve_hybrid.cls_cm_data import BaseCMData
from demand_curve_hybrid.cls_comparable_model import *

project_data = BaseCMData(aggregate_level='project')
bedroom_data = BaseCMData(aggregate_level='bedrooms')

training_data = pd.concat([project_data.data, bedroom_data.data], ignore_index=True).sort_values(
    by=['dw_project_id', 'num_of_bedrooms', 'transaction_month']
)

forecasting_data = pd.concat([project_data.forecasting_data, bedroom_data.forecasting_data], ignore_index=True).sort_values(
    by=['dw_project_id', 'num_of_bedrooms', 'transaction_month']
)

comparable_demand_model = ComparableDemandModel(
    data=training_data,
    forecasting_data=forecasting_data
)

available_projects = comparable_demand_model.available_projects
available_projects.set_index(['project_name', 'num_of_bedrooms'], inplace=True)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity
