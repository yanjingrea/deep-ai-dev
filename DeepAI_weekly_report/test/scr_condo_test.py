"""
Overview:
* This Python script is dedicated to generating demand curves for specified condo projects and bedroom types.
* The output includes graphical representations of demand curves, which are stored as image paths for reporting purposes.
* The script also performs weekly tests on the demand model and saves the results for further analysis.
"""

from DeepAI_weekly_report.test.func_helper_function import *
from demand_curve_condo.scr_common_training import *
from demand_curve_main.scr_coef import query_adjust_coef


min_stock = 10
min_proj_size = 50
test_projects = None

image_paths = []

# manual input data
# --------------------------------------------------------------------------------------------
manual_data = pd.read_csv(
    '/Users/wuyanjing/PycharmProjects/app/demand_curve_condo/local/manual_input.csv'
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
        data_model=bedroom_data,
        linear_model=linear_model,
        adjusted_project_data=adjusted_project_data,
        adjusted_training_data=adjusted_training_data,
        image_paths=image_paths,
        concat=False
    )

condo_paths_df = weekly_test_demand_model(
    data_model=bedroom_data,
    demand_model=comparable_demand_model,
    image_paths=image_paths,
    min_stock=min_stock,
    min_proj_size=min_proj_size,
    test_projects=test_projects,
    label='condo'
)

image_paths_des = dev_res_dir + f'condo_paths_df.plk'
pickle.dump(condo_paths_df, open(image_paths_des, 'wb'))
