import pickle

from demand_curve_live.the_arden.scr_get_paths import model_dir


project_name = 'The Arden'
launching_period = 1
bedrooms_list = [2, 3, 4]

from datetime import datetime
today = datetime.today().date()
models_path = model_dir + f'{project_name} {today}'.replace(' ', '_')

if True:
    linear_models = pickle.load(open(models_path, 'rb'))
else:
    linear_models = {}

    for num_of_bedroom in bedrooms_list:
        adjusted_project_data = get_adjusted_project_data(
            project_name,
            num_of_bedroom
        )

        project_id = adjusted_project_data.dw_project_id.iloc[0]

        price_range = (
            adjusted_project_data[price].min() * 0.8,
            adjusted_project_data[price].max() * 1.2
        )

        linear_model, _ = comparable_demand_model.fit_project_room_demand_model(
            project_id,
            num_of_bedroom,
            exclude_ids=[project_id]
        )

        linear_models[num_of_bedroom] = linear_model

    pickle.dump(
        linear_models, open(models_path, 'wb')
    )


