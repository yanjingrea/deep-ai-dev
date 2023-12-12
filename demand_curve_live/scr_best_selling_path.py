import numpy as np
import pandas as pd

from constants.utils import print_in_green_bg
from demand_curve_sep.scr_common_training import (
    price,
    quantity,
    get_adjusted_project_data,
    comparable_demand_model
)

from demand_curve_live.scr_get_paths import *
from demand_curve_live.scr_get_model import linear_models as models
from optimization.revenue_optimization import PathRevenue, PathsCollection, customize_optimization

project_name = 'The Arden'
launching_period = 1
max_launching_period = 12

bedrooms_list = [2, 3, 4]
price_ranges = {
    2: (1600, 1930),
    3: (1550, 1850),
    4: (1600, 1900)
}

initial_paths = {
    2: (1771,)*max_launching_period,
    3: (1744,)*max_launching_period,
    4: (1744, )*max_launching_period
}


temp_paths = {}
for num_of_bedrooms in bedrooms_list:
    adjusted_project_data = get_adjusted_project_data(project_name, num_of_bedrooms)
    data_row = adjusted_project_data.iloc[[0]]
    index_to_multiply = comparable_demand_model.query_time_rebase_index(adjusted_project_data)

    def get_projects_best_path(
            path_length=8,
            price_range=(1600, 1900),
            max_growth_psf=None,
            max_growth_rate=0.02,
    ):

        def path_generator(current_path, temperature):
            lower_bound = np.random.uniform(*price_range, size=1)

            if max_growth_psf:
                upper_bound = lower_bound + max_growth_psf
            else:
                upper_bound = min(
                    lower_bound * (1 + max_growth_rate) ** path_length,
                    max(*price_range)
                )

            return tuple(
                np.sort(
                    np.random.uniform(lower_bound, upper_bound, size=path_length)
                )
            )

        def revenue_calculator(psf_path, full_output=False, discount_rate=0.025):

            valid_psf_path = np.array([])
            valid_quantity_path = np.array([])
            floor_area_sqft = data_row['floor_area_sqm'].iloc[0] * 10.76
            stock = data_row['num_of_units'].iloc[0]

            for idx, p in enumerate(psf_path):
                t = 1+idx*3
                remaining_units = int(stock - valid_quantity_path.sum())

                data = data_row.copy()
                data['price'] = p
                data['launching_period'] = t
                data['num_of_remaining_units'] = remaining_units
                q = int(models[num_of_bedrooms].predict(data).iloc[0])

                valid_psf_path = np.append(valid_psf_path, p * index_to_multiply)
                valid_quantity_path = np.append(valid_quantity_path, q)

                if q == remaining_units:
                    break

            valid_period_path = np.arange(1, len(valid_quantity_path) + 1)

            revenue = valid_quantity_path * valid_psf_path * floor_area_sqft
            discounted_revenue = revenue / (1 + discount_rate) ** valid_period_path

            results = PathRevenue(
                **{
                    'bed_num': num_of_bedrooms,
                    'quantity_path': valid_quantity_path.astype(int),
                    'psf_path': valid_psf_path,
                    'price_path': valid_psf_path * floor_area_sqft,
                    'revenue_path': revenue,
                    'discounted_revenue_path': discounted_revenue,
                    'total_revenue': np.nansum(revenue),
                    'discounted_total_revenue': np.nansum(discounted_revenue)
                }
            )

            if full_output:
                return results

            return results.discounted_total_revenue

        suggestion_path = customize_optimization(
            initial_state=(price_range[0],) * path_length,
            state_generator=path_generator,
            revenue_calculator=revenue_calculator
        )

        res = revenue_calculator(suggestion_path, full_output=True)

        if sum(res.quantity_path) != data_row['num_of_units'].iloc[0]:
            print_in_green_bg(f'{num_of_bedrooms}-bed: fail to sell out.')

        return res

    temp_paths[num_of_bedrooms] = get_projects_best_path(
        path_length=max_launching_period,
        price_range=price_ranges[num_of_bedrooms],
        max_growth_psf=150
    )

suggested_paths = PathsCollection(paths=temp_paths)

fig, ax = suggested_paths.plot()

project_tr = suggested_paths.revenue
project_tr_mill = project_tr / 10 ** 6

fig_name = f"Best Selling Path {project_name}\n" \
           f"Total revenue {project_tr_mill: g} millions"

fig.suptitle(fig_name)

file_name = f"best selling path {project_name} {max_launching_period} periods {project_tr_mill: g}m"
fig.savefig(figure_dir + file_name.replace(' ', '_') + '.png', format='png')

res = suggested_paths.to_dataframe()
res.to_csv(table_dir + file_name.replace(' ', '_') + '.csv', index=False)

print()

if False:
    def process_config(cfg: ProjectConfig, num_of_remaining_units):
        res = adjusted_project_data.copy()
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
                'num_of_remaining_units': num_of_remaining_units,
                'proj_num_of_units': sum(cfg.total_unit_count),
                'tenure': 1 if cfg.tenure == 'freehold' else 0,
                'floor_area_sqm': cfg.avg_unit_size_per_bed(num_of_bedrooms),
                'proj_max_floor': cfg.max_floor,
            }
        )

        for k, v in config_data.items():
            res[k] = v

        return res
