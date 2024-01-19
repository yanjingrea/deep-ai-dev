import pandas as pd

from demand_curve_hybrid.cls_cm_data import BaseCMData
from demand_curve_hybrid.cls_comparable_model import *

bedroom_data = BaseCMData(aggregate_level='bedrooms')

# if include bedroom type data
if False:
    project_data = BaseCMData(aggregate_level='project')
    training_data = pd.concat([project_data.data, bedroom_data.data], ignore_index=True)
    forecasting_data = pd.concat(
        [
            project_data.forecasting_data,
            bedroom_data.forecasting_data
        ], ignore_index=True
    )
else:
    training_data = bedroom_data.data
    forecasting_data = bedroom_data.forecasting_data

sequence = ['dw_project_id', 'num_of_bedrooms', 'transaction_month']
training_data = training_data.sort_values(by=sequence)
forecasting_data = forecasting_data.sort_values(by=sequence)

comparable_demand_model = ComparableDemandModel(
    data=training_data,
    forecasting_data=forecasting_data
)

available_projects = comparable_demand_model.available_projects
available_projects.set_index(['project_name', 'num_of_bedrooms'], inplace=True)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity