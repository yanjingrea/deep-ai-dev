import numpy as np
import pandas as pd

from demand_curve_hybrid.weekly_report.func_helper_function import *
from demand_curve_hybrid.scr_common_training import *
from demand_curve_hybrid.weekly_report.scr_get_paths import dev_res_dir

min_stock = 10
min_proj_size = 50
test_projects = None

data = training_data.copy()
forecast_data = forecasting_data.copy()

if test_projects is None:
    max_period = data.groupby('dw_project_id')['launching_period'].transform('max')
    test_data = data[
        (data['num_of_remaining_units'] - data[quantity] >= min_stock) &
        (data['launching_period'] == max_period)
        ]
else:
    test_data = data[data.isin(test_projects)]

image_paths = []

# forecast data
# --------------------------------------------------------------------------------------------
for idx in np.arange(len(forecast_data)):
    temp_row = forecast_data.iloc[[idx]].copy()
    temp_name = temp_row.project_name.iloc[0]
    temp_bed = temp_row.num_of_bedrooms.iloc[0]

    _, image_paths = get_report_results(
        temp_name,
        temp_bed,
        image_paths=image_paths
    )

# manual input data
# --------------------------------------------------------------------------------------------
manual_data = pd.read_csv(
    '/Users/wuyanjing/PycharmProjects/app/demand_curve_hybrid/local/manual_input.csv'
)
manual_data['transaction_month'] = pd.to_datetime(manual_data['transaction_month'])
manual_data = bedroom_data.calculate_launching_period(manual_data)

for idx in np.arange(len(manual_data)):

    adjusted_project_data = manual_data.iloc[[idx]].copy().reset_index()

    coef = query_adjust_coef(adjusted_project_data)

    rebased_project_data = adjusted_project_data.copy()
    rebased_project_data['price'] = rebased_project_data['price'] / coef

    project_name = adjusted_project_data.project_name.iloc[0]
    num_of_bedroom = adjusted_project_data.num_of_bedrooms.iloc[0]

    if project_name in comparable_demand_model.available_projects:
        continue

    if project_name == 'The Arcady at Boon Keng':
        if num_of_bedroom in [3, 4]:
            include_ids = [
                # '9a9ccc9291a11467ffe12dd8607950ee',
                'b8846c85fc359a8072addec452d2e016'
            ]

            if num_of_bedroom == 4:
                max_launching_period = 5
            else:
                max_launching_period = 12
        else:
            include_ids = [
                '9a9ccc9291a11467ffe12dd8607950ee',
                'b8846c85fc359a8072addec452d2e016'
            ]

            if num_of_bedroom == 2:
                max_launching_period = 3
            else:
                max_launching_period = None
    else:
        include_ids = None
        max_launching_period = None

    linear_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data['dw_project_id'].iloc[0],
        project_data=rebased_project_data,
        num_of_bedroom=num_of_bedroom,
        include_ids=include_ids,
        max_launching_period=max_launching_period
    )

    image_paths = model_to_demand_curve(
        linear_model,
        adjusted_project_data,
        adjusted_training_data,
        image_paths
    )

# back test data
# --------------------------------------------------------------------------------------------
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

test_results_des = dev_res_dir + 'test_results.plk'
pickle.dump(test_results, open(test_results_des, 'wb'))


paths_df = pd.DataFrame(image_paths)
image_paths_des = dev_res_dir + 'paths_df.plk'
pickle.dump(paths_df, open(image_paths_des, 'wb'))

if False:
    test_results['error_to_sales'] = np.abs(test_results['pred_sales'] - test_results['sales']) / test_results['sales']
    test_results['error_to_stock'] = np.abs(test_results['pred_sales'] - test_results['sales']) / test_results[
        'num_of_units']
    n_sample = test_results.groupby(['num_of_bedrooms'])['project_name'].count()


    def calculate_correct_rate(metric: Literal['error_to_sales', 'error_to_stock']):
        correct_rate_data = pd.DataFrame()

        std = test_results[metric].std()
        percents = np.append(np.arange(0.02, 0.12, 0.02), std)

        for temp_idx, confidence_interval in enumerate(percents):

            n_correct = test_results[
                (test_results[metric] <= confidence_interval)
            ].groupby(['num_of_bedrooms'])['project_name'].count()

            series_name = f'correct_rate_{confidence_interval:.2f}' if temp_idx != len(
                percents
            ) - 1 else f'correct_rate_std'

            correct_rate = (n_correct / n_sample).rename(series_name)

            if correct_rate_data.empty:
                correct_rate_data = correct_rate
            else:
                correct_rate_data = pd.concat([correct_rate_data, correct_rate], axis=1)

        return correct_rate_data


    to_stock_correct_rate = calculate_correct_rate('error_to_stock')
    to_sales_correct_rate = calculate_correct_rate('error_to_sales')
print(test_results)
