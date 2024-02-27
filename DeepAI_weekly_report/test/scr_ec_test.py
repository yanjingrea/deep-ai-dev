from DeepAI_weekly_report.test.func_helper_function import *
from demand_curve_ec.scr_common_training import comparable_demand_model, bedroom_data
from demand_curve_main.scr_coef import query_adjust_coef
from demand_curve_ec.manual_input import *


min_stock = 10
min_proj_size = 50
test_projects = None

image_paths = []

# manual input data
# --------------------------------------------------------------------------------------------
manual_data = pd.read_csv(
    '/Users/wuyanjing/PycharmProjects/deep-ai-dev/demand_curve_condo/manual_input/manual_input.csv'
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

    if num_of_bedroom == 3:
        # continue
        max_launching_period = 3
        include_ids = None
        exclude_ids = None

    elif num_of_bedroom == 4:
        # continue
        max_launching_period = 12
        include_ids = None
        exclude_ids = [
            # '5fa4cfb70b0238c6f81ffea1377e9e45',
            'd51b263c3d0e13d4d8368a5d08d02087'
        ]

    else:
        # continue
        max_launching_period = 12
        include_ids = None
        exclude_ids = ['d51b263c3d0e13d4d8368a5d08d02087']

    linear_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
        project_id=adjusted_project_data['dw_project_id'].iloc[0],
        project_data=rebased_project_data,
        num_of_bedroom=num_of_bedroom,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
        max_launching_period=max_launching_period,
        # price_range=(1450, 1650),
        coefficient_range=(-8, -3)
    )

    if linear_model is None:
        continue


    image_paths = model_to_demand_curve(
        data_model=bedroom_data,
        linear_model=linear_model,
        adjusted_project_data=adjusted_project_data,
        adjusted_training_data=adjusted_training_data,
        image_paths=image_paths,
        concat=False
    )

ec_paths_df = weekly_test_demand_model(
    data_model=bedroom_data,
    demand_model=comparable_demand_model,
    image_paths=image_paths,
    min_stock=min_stock,
    min_proj_size=min_proj_size,
    test_projects=test_projects,
    label='ec'
)

image_paths_des = dev_res_dir + 'ec_paths_df.plk'
pickle.dump(ec_paths_df, open(image_paths_des, 'wb'))

