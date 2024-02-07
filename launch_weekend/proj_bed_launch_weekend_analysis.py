import numpy as np
import pandas as pd

from constants.redshift import query_data

with open(
        'proj_bed_sales_rate.sql',
        'r'
) as sql_file:
    sql_script = sql_file.read()

raw_data = query_data(sql_script)

y_col = 'sales_quantity'
x_cols = [
    # 'dw_project_id', 'project_display_name',
    'num_of_bedrooms',
    'num_of_units',
    'launch_year',
    'tenure',
    'latitude',
    'longitude',
    'building_count',
    'unit_count',
    'residential_unit_count',
    'commercial_unit_count',
    'average_launch_psf',
    'average_launch_price',
    'max_floor_count',
    'project_age',
    'land_max_gfa',
    'land_size_sqft',
    'project_units_zero_rm',
    'project_units_one_rm', 'project_units_two_rm',
    'project_units_three_rm', 'project_units_four_rm',
    'project_units_five_rm', 'project_avg_size_of_zero_rm',
    'project_avg_size_of_one_rm', 'project_avg_size_of_two_rm',
    'project_avg_size_of_three_rm', 'project_avg_size_of_four_rm',
    'project_avg_size_of_five_rm',
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

train_projects, test_projects = train_test_split(unique_projects, test_size=0.15, random_state=42)

train = data[data[project_key].isin(train_projects)].copy()
test = data[data[project_key].isin(test_projects)].copy()

test = test[test[y_col] != 0].copy()

model = XGBRegressor(random_state=42)
model.fit(train[final_x_cols].astype(float), train[y_col].astype(int))


y_pred_col = 'pred'+y_col
test[y_pred_col] = model.predict(test[final_x_cols].astype(float))

test_projects_data = test.groupby(project_key)[[y_col, y_pred_col, 'num_of_units']].sum()

y_pred = test_projects_data[y_pred_col].values
y_true = test_projects_data[y_col].values

mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
print(f'mean_absolute_percentage_error: {mape * 100 :.2f}%')

error = pd.Series(y_pred / y_true - 1)
abs_error = error.abs()

sample_size = len(test_projects_data)

print('Error compared to Sales:')
for t in np.arange(0.02, 0.12, 0.02):
    correct_rate = len(abs_error[abs_error <= t]) / sample_size
    print(f'correct rate in confidence level {t * 100: .0f}%: {correct_rate * 100: .2f}%')

print(f'-' * 20)
print('Error compared to Stock:')
error_to_stock = np.abs(y_pred - y_true) / test_projects_data['num_of_units'].values
for t in np.arange(0.02, 0.12, 0.02):
    correct_rate = len(error_to_stock[error_to_stock <= t]) / sample_size
    print(f'correct rate in confidence level {t * 100: .0f}%: {correct_rate * 100: .2f}%')

print(f'-' * 20)
print('Features Importance:')
features_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)

print(features_importance)

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
