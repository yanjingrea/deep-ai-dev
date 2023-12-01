import numpy as np
from demand_model.weekly_report.func_test_function import *

min_stock = 10
min_proj_size = 50
test_projects = None

data = comparable_demand_model.data.copy()
forecast_data = comparable_demand_model.forecasting_data.copy()

if test_projects is None:
    max_period = data.groupby('dw_project_id')['launching_period'].transform('max')
    test_data = data[
        (data['num_of_remaining_units'] - data[quantity] >= min_stock) &
        (data['launching_period'] == max_period)
        ]
else:
    test_data = data[data.isin(test_projects)]

image_paths = []

for idx in np.arange(len(forecast_data)):
    temp_row = forecast_data.iloc[[idx]].copy()
    temp_name = temp_row.project_name.iloc[0]
    temp_bed = temp_row.num_of_bedrooms.iloc[0]

    _, image_paths = get_report_results(
        temp_name,
        temp_bed,
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

    test_results, image_paths = get_report_results(
        temp_name,
        temp_bed,
        image_paths=image_paths,
        test_results=test_results
    )


print(test_results)
