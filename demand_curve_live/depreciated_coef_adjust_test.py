import numpy as np
import pandas as pd

from demand_model_utils.cls_linear_demand_model import BaseLinearDemandModel
from demand_curve_sep.cls_plt_demand_curve import PltDemandCurve
from demand_curve_sep.scr_common_training import (
    price,
    get_adjusted_project_data,
    comparable_demand_model,
    get_rebased_project_data
)
from optimization.project_config import ProjectConfig
from src.main.land_config import LandConstraints, TLandLot, ConfigGenerationError

"""
Shift Curve Validation Test

This script performs a test to validate whether shifting the demand curve by applying a coefficient (coef) has the 
same effect as adjusting the prices of the training data using the same coefficient and then refitting the model.

Author: Wu Yanjing
Date: Dec 14 2023
"""


def process_config(
    cfg: ProjectConfig,
    num_of_bedrooms,
):

    launch_year_month = pd.to_datetime(f'{cfg.launching_year}-{cfg.launching_month:02d}-01')
    config_data = pd.DataFrame(
        {
            'project_name': [cfg.project_name],
            'num_of_bedrooms': num_of_bedrooms,
            'launch_year_month': launch_year_month,
            'transaction_month': launch_year_month,
            'launching_period': np.nan,
            'sales': np.nan,
            'price': np.nan,
            'num_of_units': cfg.get_units_count(num_of_bedrooms),
            'num_of_remaining_units': cfg.get_units_count(num_of_bedrooms),
            'proj_num_of_units': sum(cfg.total_unit_count),
            'tenure': 1 if cfg.tenure == 'freehold' else 0,
            'floor_area_sqm': cfg.avg_unit_size_per_bed(num_of_bedrooms),
            'proj_max_floor': cfg.max_floor,
        }
    )

    return config_data


land_constraints = LandConstraints.from_location(TLandLot('MK10-01653L'))

initial_config = ProjectConfig(
    project_name='The Arden',
    launching_year=2023,
    launching_month=8,
    tenure='leasehold',

    total_unit_count=(0, 45, 30, 30, 0),
    avg_unit_size=(0, 64.8, 101.0, 124.7),
    max_floor=5,
    num_of_stacks=3,

    completion_year=2025,
    is_top10_developer=1
)

unit_size_ranges = {
    2: (60, 95),
    3: (95, 120),
    4: (120, 160)
}
config_params = LandConstraints.TConfigParams(
    n_buildings=3,
    unit_size_ranges=unit_size_ranges,
    max_stacks=30
)


def get_random_project_config(project_config_params):

    n_attempts = 10

    for n in range(n_attempts):

        try:

            bc = land_constraints.gen_random_config(
                project_config_params
            ).aggregated(initial_config.static_config)

            pc_dict = {}
            for k, v in bc.__dict__.items():

                if k == 'postal_code':
                    pc_dict['project_name'] = v
                else:
                    if isinstance(v, tuple):
                        v = tuple(v)

                    pc_dict[k] = v

            return ProjectConfig(**pc_dict)

        except ConfigGenerationError:

            continue

    print(f'Unable to generate qualified config after {n_attempts} attempts.')


random_config = get_random_project_config(
    project_config_params=config_params
)

project_name = 'The Arden'
num_of_bedrooms = 2
rebased_project_data = get_rebased_project_data(
    project_name,
    num_of_bedrooms
)

adjusted_project_data = get_adjusted_project_data(
    project_name,
    num_of_bedrooms
)

project_id = adjusted_project_data.dw_project_id.iloc[0]

price_range = (
    adjusted_project_data[price].min() * 0.8,
    adjusted_project_data[price].max() * 1.2
)

manual_min = rebased_project_data.price.min()
manual_max = rebased_project_data.price.max()

if num_of_bedrooms == 2:

    manual_price_range = (
        manual_min,
        manual_max
    )

else:
    manual_price_range = (
        manual_min / 0.9 * 0.85,
        manual_max / 1.1 * 1.2
    )

old_model, adjusted_training_data = comparable_demand_model.fit_project_room_demand_model(
    project_id,
    num_of_bedrooms,
    price_range=manual_price_range,
    exclude_ids=[project_id],
    # include_ids=['7a0eaa8196d9676a189aa4a7fbabc7e5']
)

row = adjusted_project_data.iloc[[0]]
old_curve = old_model.extract_2d_demand_curve(
    row,
    launching_period=1,
    price_range=price_range,
    fig_format='plt',
)

new_project_data = process_config(random_config, num_of_bedrooms)
old_coef = comparable_demand_model.query_adjust_coef(rebased_project_data)
new_coef = comparable_demand_model.query_adjust_coef(new_project_data)

index_to_multiply = 1 / old_coef * new_coef

new_training_data = adjusted_training_data.copy()
new_training_data['price'] = new_training_data['price'] * index_to_multiply

new_model = BaseLinearDemandModel(
    quantity='sales',
    price='price',
    features=comparable_demand_model.features
).fit(new_training_data)

new_curve = new_model.extract_2d_demand_curve(
    row,
    launching_period=1,
    price_range=(price_range[0] * index_to_multiply, price_range[1] * index_to_multiply),
    fig_format='plt',
)
old_curve_transformed = PltDemandCurve(
    P=old_curve.P * index_to_multiply,
    Q=old_curve.Q
)

fig, ax = old_curve.plot(color='red')
fig, ax = old_curve_transformed.plot(fig, ax, color='yellow')
fig, ax = new_curve.plot(fig, ax)

print()
