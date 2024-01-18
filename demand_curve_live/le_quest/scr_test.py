import numpy as np

from demand_curve_live.le_quest.func_helper import (
    models,
    calculate_actual_revenue,
    fill_nominal_data,
    LocalBestPath
)
from demand_curve_live.le_quest.scr_get_paths import table_dir
from optimization.cls_base_simulation_results import ProjectSalesPaths
from optimization.project_config import ProjectConfig

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

pred_paths = {}
for bed_num in np.arange(1, 5):

    bed_data = fill_nominal_data(bed_num, size_rebased=False)

    path_model = LocalBestPath(
        num_of_bedrooms=bed_num,
        demand_model=models[bed_num],
        initial_config=initial_config
    )

    pred_path = path_model.calculate_total_revenue(
        initial_config,
        psf_path=bed_data['price'].values,
        full_output=True,
        discount_rate=0.025
    )
    pred_paths[bed_num] = pred_path

if True:
    true_paths = {}
    for bed_num in np.arange(1, 5):

        true_bed_revenue = calculate_actual_revenue(bed_num)
        pred_bed_revenue = pred_paths[bed_num]

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
                1 + 0.025) ** 4

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
    predictive_paths = ProjectSalesPaths(pred_paths)

    total_error = predictive_paths.revenue / actual_paths.revenue - 1
    project_tr_mill = predictive_paths.revenue / 10 ** 6

    print(
        f"""
            the whole project
            actual total revenue:  {actual_paths.revenue / 10 ** 6: .2f} million dollars
            predictive total revenue: {project_tr_mill: .2f} million dollars
            error of total revenue: {total_error * 100: .2f}%
        """
    )

if False:
    for bed_num in np.arange(1, 6):

        pred_bed_revenue = pred_paths[bed_num]
        actual_revenue = {
            1: 87587888.88888888,
            2: 135059000,
            3: 248396250,
            4: 78800000
        }.get(bed_num, lambda a: a)

        actual_stock = {
            1: 133,
            2: 144,
            3: 192,
            4: 48
        }.get(bed_num, lambda a: a)

        quantity_diff = pred_bed_revenue.quantity_path.sum() - actual_stock
        gain = pred_bed_revenue.total_revenue / actual_revenue - 1

        print(
            f"""
                        {bed_num}-bedroom
                        cumulative quantity error:      {quantity_diff}
                        actual total revenue:           {actual_revenue / 10 ** 6: .2f} million dollars
                        predictive total revenue:       {pred_bed_revenue.total_revenue / 10 ** 6: .2f} million dollars
                        relative gain:                  {gain * 100: .2f}%
                        """
        )

    predictive_paths = ProjectSalesPaths(pred_paths)
    actual_paths_revenue = 549843138.8888888

    total_error = predictive_paths.revenue / actual_paths_revenue - 1
    project_tr_mill = predictive_paths.revenue / 10 ** 6

    print(
        f"""
            the whole project
            actual total revenue:  {actual_paths_revenue / 10 ** 6: .2f} million dollars
            predictive total revenue: {project_tr_mill: .2f} million dollars
            error of total revenue: {total_error * 100: .2f}%
        """
    )

file_name = f"back test {initial_config.project_name} {project_tr_mill:.2f}m"
res = predictive_paths.detailed_dataframe()
res.to_csv(table_dir + file_name.replace(' ', '_') + '.csv', index=False)

res2 = predictive_paths.summarized_dataframe()
res2.to_csv(table_dir + "sum_" + file_name.replace(' ', '_') + '.csv', index=False)

print()
