from demand_curve_condo.cls_cm_data import CondoCMData
from demand_curve_main.cls_comparable_model import *



bedroom_data = CondoCMData(aggregate_level='bedrooms')
sequence = ['dw_project_id', 'num_of_bedrooms', 'transaction_month']

training_data = bedroom_data.data.sort_values(by=sequence)
forecasting_data = bedroom_data.forecasting_data.sort_values(by=sequence)

comparable_demand_model = ComparableDemandModel(
    data=training_data,
    forecasting_data=forecasting_data
)

available_projects = comparable_demand_model.available_projects
available_projects.set_index(['project_name', 'num_of_bedrooms'], inplace=True)

price = comparable_demand_model.price
quantity = comparable_demand_model.quantity