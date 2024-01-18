import numpy as np
from demand_curve_live.le_quest.func_helper import (
    models,
    calculate_actual_revenue,
    LocalBestPath,
    LocalBestPaths
)
from demand_curve_live.le_quest.scr_get_paths import table_dir
from optimization.cls_base_simulation_results import ProjectSalesPaths, PathGeneratorParams
from optimization.project_config import ProjectConfig
from src.utils.cm_timer import cm_timer

initial_config = ProjectConfig(
    project_name='Le Quest',
    launching_year=2017,
    launching_month=8,
    tenure='leasehold',

    total_unit_count=(132, 144, 192, 48, 0),
    avg_unit_size=(45.02, 63.20, 88.52, 115.04, 0),
    max_floor=16,
    num_of_stacks=43,

    completion_year=2020,
    is_top10_developer=1
)

if True:
    price_ranges = {
        1: (1350, 1475),
        2: (1375, 1525),
        3: (1350, 1450),
        4: (1325, 1425),
    }

else:
    path_params = {
        1: {
            1: PathGeneratorParams(path_length=9, price_range=(1700, 1800)),
            2: PathGeneratorParams(path_length=8, price_range=(1800, 1880))
        },
        2: {
            1: PathGeneratorParams(path_length=9, price_range=(1700, 1800)),
            2: PathGeneratorParams(path_length=8, price_range=(1880, 1880))
        },
        3: {
            1: PathGeneratorParams(path_length=9, price_range=(1665, 1750)),
            2: PathGeneratorParams(path_length=8, price_range=(1750, 1850))
        },
        4: {
            1: PathGeneratorParams(path_length=9, price_range=(1640, 1750)),
            2: PathGeneratorParams(path_length=8, price_range=(1750, 1865))
        },
        5: {
            1: PathGeneratorParams(path_length=9, price_range=(1550, 1625)),
            2: PathGeneratorParams(path_length=8, price_range=(1625, 1725))
        }
    }

path_length = 13

with cm_timer(fmt=f'getting the best selling paths ' + '{:g} s'):

    if False:
        suggestion_paths = ProjectSalesPaths(
            {
                bed_num: LocalBestPath(
                    num_of_bedrooms=bed_num,
                    demand_model=models[bed_num],
                    initial_config=initial_config
                ).get_projects_best_path(
                    initial_config,
                    price_range=price_ranges[bed_num],
                    path_length=path_length,
                    discount_rate=0.0125
                )
                for bed_num in np.arange(1, 6)
            }
        )

    paths_model = LocalBestPaths(
        demand_models=models,
        initial_config=initial_config
    )

    suggestion_paths = paths_model.get_best_selling_paths(
        price_ranges=price_ranges,
        path_lengths=path_length,
    )

true_paths = {}
for bed_num in models.keys():

    true_bed_revenue = calculate_actual_revenue(bed_num)
    pred_bed_revenue = suggestion_paths.paths[bed_num]

    n_trans = true_bed_revenue.quantity_path.sum()
    actual_stock = {
        1: 132,
        2: 144,
        3: 192,
        4: 48
    }.get(bed_num, lambda a: a)

    n_missing_trans = actual_stock - n_trans
    avg_amount = true_bed_revenue.total_revenue / n_trans

    actual_revenue = avg_amount * actual_stock
    actual_discounted_revenue = true_bed_revenue.discounted_total_revenue + avg_amount * n_missing_trans / (
            1 + 0.025) ** 8

    true_bed_revenue.__setattr__('total_revenue', actual_revenue)
    true_bed_revenue.__setattr__('discounted_total_revenue', actual_discounted_revenue)

    quantity_diff = pred_bed_revenue.quantity_path.sum() - actual_stock
    gain = pred_bed_revenue.total_revenue / actual_revenue - 1

    print(
        f"""
        {bed_num}-bedroom
        actual price path:          {true_bed_revenue.psf_path.astype(int)}
        actual quantity path:       {true_bed_revenue.quantity_path}
        predictive price path:      {pred_bed_revenue.psf_path.astype(int)}
        predictive quantity path:   {pred_bed_revenue.quantity_path}
        cumulative quantity error:  {quantity_diff}

        actual total revenue:           {actual_revenue / 10 ** 6: .2f} million dollars
        predictive total revenue:       {pred_bed_revenue.total_revenue / 10 ** 6: .2f} million dollars
        relative gain:   {gain * 100: .2f}%
        """
    )

    true_paths[bed_num] = true_bed_revenue

actual_paths = ProjectSalesPaths(true_paths)
actual_paths_revenue = 547264867.391927
# total_profit = suggestion_paths.revenue / actual_paths.revenue - 1
total_profit = suggestion_paths.revenue / actual_paths_revenue - 1
project_tr_mill = suggestion_paths.revenue / 10 ** 6

print(
    f"""
    the whole project
    actual total revenue:               {actual_paths_revenue / 10 ** 6: .2f} million dollars
    predictive total revenue:           {project_tr_mill: .2f} million dollars
    gain of total revenue:              {total_profit * 100: .2f}%
    """
)

file_name = f"best selling path {initial_config.project_name} {path_length} periods {project_tr_mill:.2f}m"
res = suggestion_paths.detailed_dataframe()
res.to_csv(table_dir + file_name.replace(' ', '_') + '.csv', index=False)

res2 = suggestion_paths.summarized_dataframe()
res2.to_csv(table_dir + "sum_" + file_name.replace(' ', '_') + '.csv', index=False)

print()
