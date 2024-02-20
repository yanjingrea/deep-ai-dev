import numpy as np
import pandas as pd
import shap

from catboost import CatBoostRegressor, Pool
from matplotlib import pyplot as plt

from constants.redshift import query_data
from constants.utils import get_output_dir

output_dir = get_output_dir(__file__) + 'live/'

seed = 123

# manual input
hold_on_projects = ['Lentoria', 'The Hill @ One North']
hold_on_price = [2050, 2500]
comparable_projects = [
    'Lentor Modern',
    'Lentor Hills Residences',
    'Hillock Green',
    'Amo Residence',
    'One-North Eden',
]

# set up
# ---------------------------------------
project_key = 'project_display_name'

min_stock = 75
n_groups = 10
train_min_year = 2019
test_min_year = 2021

sales_rate_col = y_col = 'sales_rate'
pred_sales_rate_col = pred_y_col = 'pred_' + y_col
sales_col = 'sales'
pred_sales_col = 'pred_' + sales_col

first_layer = pd.read_csv('first_layer_label.csv')
first_layer.set_index('project_display_name', inplace=True)
# -----------------------------------------------
# query data
with open(
        'regression_data.sql',
        'r'
) as sql_file:

    sql_script = sql_file.read()
    data = query_data(sql_script)

with open(
        'classification_comparable.sql',
        'r'
) as sql_file:
    sql_script = sql_file.read()
comparable_data = query_data(sql_script)

comp_features = ['sales', 'price_psf', 'residential_unit_count', 'meters_to_mrt']

comp_rename_dict = {
    i: "ref_" + i
    for i in comp_features
}
comp_rename_dict['project_display_name'] = 'ref_project'
# 'project_display_name': 'ref_project',

first_comp = comparable_data[comparable_data['rank'] == 1].copy()
first_comp = first_comp.merge(
    data[['project_display_name'] + comp_features].rename(columns=comp_rename_dict),
    on='ref_project'
)
data = data.merge(first_comp, on='project_display_name')

# -----------------------------------------------
# processing
data.loc[data['project_display_name'] == 'The Linq @ Beauty World', 'residential_unit_count'] = 120
data.loc[data['project_display_name'] == 'Royal Hallmark', 'residential_unit_count'] = 32

data['launch_year'] = data['activity_date'].str[:4].astype(int)
data['sales_rate'] = data['sales'] / data['residential_unit_count']
data['meters_to_mrt'] = data['meters_to_mrt'].fillna(3000)

for bed_text in [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five'
]:

    data[f'project_avg_size_of_{bed_text}_rm'] = data[f'project_avg_size_of_{bed_text}_rm'].replace(
        -1, 0
    )

data = data[data['residential_unit_count'] >= min_stock].copy()


def try_except_nan(a):

    try:
        res = first_layer['first_layer_label'].loc[a]
    except KeyError:
        res = np.nan

    return res


data['first_layer_label'] = data['project_display_name'].apply(try_except_nan)

features = [
    'num_of_bedrooms',
    # 'first_layer_label',
    'launch_year',
    'price_psf',
    'tenure_type',
    'tenure_int',
    # 'latitude',
    # 'longitude',
    'building_count',
    # 'unit_count',
    'num_of_units',
    'residential_unit_count',
    # 'commercial_unit_count',
    'max_floor_count',
    # 'project_age',
    # 'land_max_gfa',
    'land_size_sqft',
    # 'project_zero_rm_percentage',
    'project_one_rm_percentage',
    'project_two_rm_percentage',
    'project_three_rm_percentage',
    'project_four_rm_percentage',
    'project_five_rm_percentage',
    # 'project_avg_size_of_zero_rm',
    'project_avg_size_of_one_rm',
    'project_avg_size_of_two_rm', 'project_avg_size_of_three_rm',
    'project_avg_size_of_four_rm', 'project_avg_size_of_five_rm',
    'region',
    # 'zone',
    'neighborhood',
    'district',
    'km_to_sg_cbd',
    'num_of_bus_stops',
    'num_of_mrt',
    'meters_to_mrt',
    'num_of_good_schools',
    'num_of_remaining_units_neighborhood',
    'cumulative_num_of_launched_units',
    'cumulative_units_sold_neighborhood',
    'rolling_num_of_available_units_neighborhood',
    'rolling_num_of_launched_projects_neighborhood',
    'rolling_num_of_launched_units_neighborhood',
    'num_of_comparables',
    'ref_sales',
    'ref_price_psf',
    'ref_residential_unit_count',
    'ref_meters_to_mrt'
]

y_col = 'sales_rate'
X_cols = features

num_cols = data[X_cols]._get_numeric_data().columns
cat_cols = list(set(X_cols).difference(num_cols))

# helper func
construct_pool = lambda dataset: Pool(
    data=dataset[X_cols].values,
    label=dataset[y_col].values,
    feature_names=X_cols,
    cat_features=cat_cols
)

test_p = hold_on_projects
train_p = set(data['project_display_name'].unique()).difference(test_p)


def predict(*, to_test_data, first_label):

    train = to_test_data[to_test_data[project_key].isin(train_p)].copy()
    test = to_test_data[to_test_data[project_key].isin(test_p)].copy()

    if test.empty:
        return None

    train = train.dropna(subset=[y_col] + cat_cols)
    train_pool = construct_pool(train)
    test_pool = construct_pool(test)

    # model
    model = CatBoostRegressor(random_state=seed).fit(
        train_pool,
        verbose=False,

    )

    train[pred_y_col] = np.clip(
        model.predict(train_pool),
        0,
        test['num_of_units'] if y_col == 'sales' else 1
    )

    raw_pred_y = model.predict(test_pool)

    if first_label is True:
        raw_pred_y *= 1.5

    test[pred_y_col] = np.clip(
        raw_pred_y,
        0,
        test['num_of_units'] if y_col == 'sales' else 1
    )

    if y_col == 'sales_rate':
        test[sales_col] = test[sales_rate_col] * test['num_of_units']
        test[pred_sales_col] = test[pred_sales_rate_col] * test['num_of_units']

        train[sales_col] = train[sales_rate_col] * train['num_of_units']
        train[pred_sales_col] = train[pred_sales_rate_col] * train['num_of_units']

    def plot_waterfall(*, waterfall_pool, waterfall_data):

        explainer = shap.Explainer(
            model,
            feature_names=X_cols
        )

        shap_values = explainer(
            waterfall_pool
        )
        shap_values.__setattr__('data', waterfall_data[X_cols].values)

        for idx in waterfall_data.index:

            ax = shap.plots.waterfall(
                shap_values[idx],
                max_display=20,
                show=False
            )

            fig = ax.figure

            fig.set_size_inches(16, 8)
            fig.set_tight_layout(tight=True)

            p = waterfall_data['project_display_name'].iloc[idx]
            bed = waterfall_data['num_of_bedrooms'].iloc[idx]

            title = f'{sales_col} prediction ' + p + f' {bed}-bedroom'

            ax.set_title(
                title +
                f'\nactual_sales: {waterfall_data[sales_col].iloc[idx]: .0f}' +
                f'\npredictive_sales: {waterfall_data[pred_sales_col].iloc[idx]: .0f}'
            )

            plt.savefig(
                output_dir +
                f'{title.replace(" ", "_")}.png',
                dpi=300
            )

            plt.close(fig)

        plot_data = pd.concat(
            [
                test,
                train[train['project_display_name'].isin(comparable_projects)]
            ],
            ignore_index=True
        )

        plot_pool = construct_pool(plot_data)

        plot_waterfall(waterfall_pool=plot_pool, waterfall_data=plot_data)

        features_importance = pd.Series(
            model.feature_importances_,
            index=model.feature_names_
        )

        print(f'-' * 20)
        print('Features Importance:')
        print(features_importance)

    plot_data = pd.concat(
        [
            test,
            train[train['project_display_name'].isin(comparable_projects)]
        ],
        ignore_index=True
    )

    plot_pool = construct_pool(plot_data)

    plot_waterfall(waterfall_pool=plot_pool, waterfall_data=plot_data)

    return test


combined_test_data = pd.DataFrame()
for l in [True, False]:

    print(f'-' * 40)
    print(f'test group with label {l}')

    label_set = data[data['first_layer_label'].apply(lambda a: True if a is l else False)]

    combined_test_data = pd.concat(
        [
            combined_test_data,
            predict(to_test_data=label_set, first_label=l)
        ]
    )
