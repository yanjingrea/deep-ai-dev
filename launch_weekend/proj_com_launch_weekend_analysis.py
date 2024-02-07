from typing import Union

import numpy as np
import pandas as pd

from constants.redshift import query_data
from launch_weekend.scatter_plot import scatter_plot_with_reg_and_label

raw_data = query_data(
    f"""
    with
        base_launch_data as (
                                select
                                    dw_project_id,
                                    min(transaction_date) as launch_date
                                from (
                                         select
                                             *,
                                                     row_number()
                                                     over (partition by dw_property_id order by transaction_date desc) as seq
                                         from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                              join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                   using (dw_project_id)
                                         where a.property_type_group = 'Condo'
                                           and transaction_sub_type = 'new sale'
                                           and property_type != 'ec'
                                     ) as "*2"
                                where seq = 1
                                group by 1
                            )
        ,
        base_sales as (
                          select
                              a.dw_project_id,
                              num_of_bedrooms,
                              (
                                  select
                                      non_landed_index
                                  from developer_tool.sg_gov_residential_index
                                  order by quarter_index desc
                                  limit 1
                              ) as current_index,
                              count(dw_property_id) as sales,
                              avg(unit_price_psf / c.non_landed_index * current_index) as average_launch_psf,
                              avg(transaction_amount / c.non_landed_index * current_index) as average_launch_price
                          from base_launch_data a
                               left join data_science.ui_master_sg_transactions_view_filled_features_condo b
                                         on a.dw_project_id = b.dw_project_id
                                             and b.transaction_date::date <= dateadd(day, 7, launch_date::date)
                               join developer_tool.sg_gov_residential_index c
                                    on b.transaction_quarter_index = c.quarter_index
                          group by 1, 2
                      )
            ,
        base as (
                    select
                        dw_project_id,
                        project_launch_month,
                        num_of_bedrooms,
                        case
                            when num_of_bedrooms = 0 then project_units_zero_rm
                            when num_of_bedrooms = 1 then project_units_one_rm
                            when num_of_bedrooms = 2 then project_units_two_rm
                            when num_of_bedrooms = 3 then project_units_three_rm
                            when num_of_bedrooms = 4 then project_units_four_rm
                            when num_of_bedrooms = 5 then project_units_five_rm
                            end
                            as num_of_units,
                        case when sales is null then 0 else sales end as sales,
                        average_launch_psf,
                        neighborhood
                    from (select distinct dw_project_id, num_of_bedrooms from data_science.ui_master_sg_properties_view_filled_static_features_condo) as a
                        left join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                            using (dw_project_id)
                         left join base_sales c
                              using (dw_project_id, num_of_bedrooms)
                    where left(project_launch_month, 4)::int >= 2015
                )
            ,
        base_comparable as (
                               select distinct
                                   p_base.dw_project_id as dw_project_id,
                                   p_base.num_of_bedrooms,
                                   p_base.sales as sales,
                                   p_base.average_launch_psf as average_launch_psf,
                                   p_base.num_of_units as num_of_units,
                                   p_base.sales / p_base.num_of_units as sales_rate,
                                   c_base.dw_project_id as ref_project_id,
                                   c_base.sales as ref_sales,
                                   c_base.average_launch_psf as ref_average_launch_psf,
                                   c_base.num_of_units as ref_num_of_units,
                                   c_base.sales / c_base.num_of_units as ref_sales_rate
    --                                        row_number()
    --                                        over (partition by p_base.dw_project_id, p_base.num_of_bedrooms order by p.comparison_order) as similarity_order
                               from base as p_base
                                    left join ui_app.project_comparables_prod_sg p
                                              on p.project_dwid = p_base.dw_project_id
    --                                               and comparable_type = 'similar-resale'
                                    left join base as c_base on
                                           p.comparable_project_dwid = c_base.dw_project_id
                                       and p_base.num_of_bedrooms = c_base.num_of_bedrooms
                                       and p_base.project_launch_month >= c_base.project_launch_month
                                       and p_base.num_of_units > 0
                                       and c_base.num_of_units > 0
                                       and p_base.num_of_bedrooms = c_base.num_of_bedrooms
                                       and abs(p_base.average_launch_psf / c_base.average_launch_psf - 1) <= 0.15
                               order by 1, num_of_bedrooms, p.comparison_order
                           ),
        base_static as (
                           select
                               *
                           from (
                                    select
                                        project_dwid as dw_project_id,
                                        project_display_name,
                                        launch_date,
                                        tenure_type,
                                        tenure_int,
                                        latitude,
                                        longitude,
                                        building_count,
                                        unit_count,
                                        residential_unit_count,
                                        commercial_unit_count,
                                        max_floor_count,
                                        project_age,
                                        land_max_gfa,
                                        land_size_sqft
                                    from ui_app.project_summary_prod_sg
                                    where property_group = 'condo'
                                      and property_type != 'ec'
                                      and launch_date is not null
                                ) a
                                left join (
                                              select
                                                  dw_project_id,
                                                  project_units_zero_rm,
                                                  project_units_one_rm,
                                                  project_units_two_rm,
                                                  project_units_three_rm,
                                                  project_units_four_rm,
                                                  project_units_five_rm,
                                                  project_avg_size_of_zero_rm,
                                                  project_avg_size_of_one_rm,
                                                  project_avg_size_of_two_rm,
                                                  project_avg_size_of_three_rm,
                                                  project_avg_size_of_four_rm,
                                                  project_avg_size_of_five_rm,
                                                  region,
                                                  zone,
                                                  neighborhood,
                                                  district
                                              from data_science.ui_master_sg_project_geo_view_filled_features_condo
                                          ) as b
                                          using (dw_project_id)
    
                       ),
        base_geo as (
                        select
                            dw_project_id,
                            max(km_to_sg_cbd) as km_to_sg_cbd,
                            max(num_of_bus_stops) as num_of_bus_stops,
                            max(meters_to_mrt) as meters_to_mrt,
                            max(num_of_good_schools) as num_of_good_schools
                        from data_science.ui_master_sg_properties_view_filled_static_features_condo p
                             join data_science.ui_master_sg_building_view_filled_features_condo b
                                  using (dw_building_id)
                        group by 1
                    )
    select
        num_of_bedrooms,
        a.*,
        left(launch_date, 4)::int as launch_year,
        sales,
        average_launch_psf,
        num_of_units,
        sales_rate,
        ref_project_id,
        ref_sales,
        ref_average_launch_psf,
        ref_num_of_units,
        ref_sales_rate,
    --     similarity_order,
        km_to_sg_cbd,
        num_of_bus_stops,
        meters_to_mrt,
        num_of_good_schools
    from base_static a
         left join base_comparable b
                   using (dw_project_id)
         left join base_geo c
                   using (dw_project_id)
    order by launch_date desc, project_display_name, num_of_bedrooms
    """
)

y_col = 'sales'
x_cols = [
    # 'dw_project_id', 'project_display_name',
    # 'launch_date',
    'num_of_bedrooms',
    'tenure_type',
    'tenure_int',
    'latitude',
    'longitude',
    'building_count',
    # 'unit_count',
    'residential_unit_count',
    # 'commercial_unit_count',
    'max_floor_count',
    # 'project_age',
    'land_max_gfa',
    'land_size_sqft',
    'project_units_zero_rm',
    'project_units_one_rm',
    'project_units_two_rm',
    'project_units_three_rm',
    'project_units_four_rm',
    'project_units_five_rm',
    'project_avg_size_of_zero_rm',
    'project_avg_size_of_one_rm',
    'project_avg_size_of_two_rm',
    'project_avg_size_of_three_rm',
    'project_avg_size_of_four_rm',
    'project_avg_size_of_five_rm',
    # 'region', 'zone', 'neighborhood', 'district',
    'launch_year',
    # 'dw_project_id',
    # 'num_of_bedrooms',
    # 'sales',
    'average_launch_psf',
    'num_of_units',
    'sales_rate',
    # 'ref_project_id',
    'ref_sales',
    'ref_average_launch_psf',
    'ref_num_of_units',
    'ref_sales_rate',
    'similarity_order',
    'km_to_sg_cbd',
    'num_of_bus_stops',
    'meters_to_mrt',
    'num_of_good_schools'
]

raw_data = raw_data.dropna(subset=[y_col])
raw_data = raw_data[raw_data['residential_unit_count'] > 150]

for bed_text in [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five'
]:

    raw_data[f'project_avg_size_of_{bed_text}_rm'] = raw_data[f'project_avg_size_of_{bed_text}_rm'].replace(-1, 0)

raw_data['meters_to_mrt'] = raw_data['meters_to_mrt'].fillna(3000)

project_key = 'project_display_name'
unique_projects = raw_data[project_key].unique()
num_col = raw_data[x_cols]._get_numeric_data().columns
cat_col = list(set(x_cols).difference(num_col))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    [
        # ("numerical", StandardScaler(), num_col),
        ("categorical", OneHotEncoder(sparse_output=False), cat_col),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough"
)

values = preprocessor.fit_transform(raw_data[[project_key, y_col] + x_cols])

data = pd.DataFrame(
    data=values,
    columns=preprocessor.get_feature_names_out()
)

final_x_cols = data.columns.difference([project_key, y_col])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor


def fit_and_test_regressor(
    *,
    train_projects: Union[list, set, np.ndarray],
    test_projects: Union[list, set, np.ndarray]
):

    train = data[data[project_key].isin(train_projects)].copy()
    test = data[data[project_key].isin(test_projects)].copy()

    test = test[test[y_col] != 0].copy()

    model = XGBRegressor(random_state=42)
    model.fit(train[final_x_cols].astype(float), train[y_col].astype(int))

    print(f'-' * 20)
    print('Features Importance:')
    features_importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=False)
    print(features_importance)

    raw_pred_y = model.predict(test[final_x_cols].astype(float))

    test[y_pred_col] = np.clip(raw_pred_y, 0, test['num_of_units'])

    test_projects_data = test.groupby(
        [project_key, 'num_of_bedrooms'],
        as_index=False
    )[[y_col, y_pred_col, 'num_of_units']].mean()
    test_projects_data = test_projects_data.groupby(project_key, as_index=False)[
        [y_col, y_pred_col, 'num_of_units']].sum()
    test_projects_data = test_projects_data.convert_dtypes()

    test_projects_data['sales_rate'] = test_projects_data[y_col] / test_projects_data['num_of_units']
    test_projects_data['pred_sales_rate'] = test_projects_data[y_pred_col] / test_projects_data['num_of_units']

    return test_projects_data


# train_projects, test_projects = train_test_split(unique_projects, test_size=0.15, random_state=42)

n_unique_projects = len(unique_projects)
n_groups = 10
n_samples = n_unique_projects // n_groups
idx_list = np.arange(0, n_unique_projects + n_samples, n_samples)

y_pred_col = 'pred_' + y_col

combined_test_data = pd.DataFrame()
for i, idx_num in enumerate(idx_list):

    if i == 0:
        continue

    test_p = unique_projects[idx_list[i - 1]: idx_num]
    train_p = set(unique_projects).difference(test_p)

    res = fit_and_test_regressor(train_projects=train_p, test_projects=test_p)

    combined_test_data = pd.concat([combined_test_data, res], ignore_index=True)

y_pred = combined_test_data[y_pred_col].values
y_true = combined_test_data[y_col].values

mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
print(f'mean_absolute_percentage_error: {mape * 100 :.2f}%')

error = pd.Series(y_pred / y_true - 1)
abs_error = error.abs()

sample_size = len(combined_test_data)

print('Error compared to Sales:')
for t in np.arange(0.02, 0.12, 0.02):
    correct_rate = len(abs_error[abs_error <= t]) / sample_size
    print(f'correct rate in confidence level {t * 100: .0f}%: {correct_rate * 100: .2f}%')

print(f'-' * 20)
print('Error compared to Stock:')
error_to_stock = np.abs(y_pred - y_true) / combined_test_data['num_of_units'].values
for t in np.arange(0.02, 0.12, 0.02):
    correct_rate = len(error_to_stock[error_to_stock <= t]) / sample_size
    print(f'correct rate in confidence level {t * 100: .0f}%: {correct_rate * 100: .2f}%')

fig, ax = scatter_plot_with_reg_and_label(
    data=combined_test_data,
    x_col='sales_rate',
    y_col='pred_sales_rate',
    label_col='num_of_units',
    n_bins=5
)

degree_45 = np.linspace(0, 1 + 1 / 50, 50)

ax.plot(
    degree_45,
    degree_45,
    color='red',
    alpha=0.5,
    linestyle='dashed'
)

ax.fill_between(
    x=degree_45,
    y1=degree_45+0.1,
    y2=degree_45-0.1,
    color='red',
    alpha=0.2
)


print()
# ---------------------------------------------------------------


if False:
    from catboost import CatBoostRegressor
    from catboost import Pool

    model = CatBoostRegressor(random_state=42, verbose=False)
    train, test = train_test_split(raw_data, test_size=0.15, random_state=42)
    test = test[test[y_col] != 0].copy()

    construct_pool_data = lambda df: Pool(
        data=df[x_cols].values,
        label=df[y_col].values,
        feature_names=x_cols,
        cat_features=cat_col
    )

    model.fit(construct_pool_data(train))
    y_pred = model.predict(construct_pool_data(test))
    y_true = test[y_col]

    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    print(f'mean_absolute_percentage_error: {mape * 100 :.2f}%')
