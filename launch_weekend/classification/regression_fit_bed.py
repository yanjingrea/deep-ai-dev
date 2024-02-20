import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from catboost import CatBoostRegressor, Pool

from constants.redshift import query_data
from launch_weekend.scatter_plot import scatter_plot_with_reg_and_label

seed = 123

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
# data_dum = pd.get_dummies(data[features+['zone', 'label']], columns=['zone'])

# helper func
construct_pool = lambda dataset: Pool(
    data=dataset[X_cols].values,
    label=dataset[y_col].values,
    feature_names=X_cols,
    cat_features=cat_cols
)


def iter_test(*, to_test_data, first_label):

    test_projects = to_test_data[project_key].unique()
    n_unique_projects = len(test_projects)
    n_samples = n_unique_projects // n_groups
    idx_list = np.arange(0, n_unique_projects + n_samples, n_samples)

    avg_features_importance = pd.DataFrame()
    res = pd.DataFrame()

    for i, idx_num in enumerate(idx_list):

        if i == 0:
            continue

        test_p = test_projects[idx_list[i - 1]: idx_num]
        train_p = set(test_projects).difference(test_p)

        train = data[data[project_key].isin(train_p)].copy()
        test = data[data[project_key].isin(test_p)].copy()

        train = train.dropna(subset=[y_col] + cat_cols)
        train_pool = construct_pool(train)
        test_pool = construct_pool(test)

        # model
        # model = XGBClassifier(random_state=seed).fit(X=train[X_cols], y=train[y_col])
        model = CatBoostRegressor(random_state=seed).fit(
            train_pool,
            verbose=False,

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

        if False:

            explainer = shap.Explainer(
                model,
                feature_names=X_cols
            )
            shap_values = explainer(
                test_pool
            )
            shap_values.__setattr__('data', test[X_cols].values)

            for idx in y_pred.index:

                ax = shap.plots.waterfall(
                    shap_values[idx],
                    max_display=20,
                    show=False
                )

                fig = ax.figure

                fig.set_size_inches(16, 8)
                fig.set_tight_layout(tight=True)

                launch_weekend.classification.training_data.data = test[X_cols].iloc[idx].values

                p = test['project_display_name'].iloc[idx]

                title = f'{y_col[0].replace(" ", "_")} prediction ' + p

                ax.set_title(
                    title +
                    f'\nactual_{y_col[0]}: {y_true.iloc[idx][0]}' +
                    f'\npredictive_{y_col[0]}: {y_pred.iloc[idx]}'
                )

                plt.savefig(
                    output_dir +
                    f'{title.replace(" ", "_")}.png',
                    dpi=300
                )

                plt.close(fig)

        features_importance = pd.Series(
            model.feature_importances_,
            index=model.feature_names_
        )

        avg_features_importance = pd.concat(
            [
                avg_features_importance,
                features_importance
            ], axis=1
        )

        res = pd.concat([test, res])

    print(f'-' * 20)
    print('Features Importance:')
    avg_features_importance = avg_features_importance.mean(axis=1).sort_values(ascending=False)
    print(avg_features_importance)

    return res


def calculate_error(y_pred, y_true):

    error_to_sales = pd.Series(y_pred[y_true != 0] / y_true[y_true != 0] - 1).abs()
    print(f'mean absolute percentage error: {error_to_sales.mean() * 100 :.2f}%')
    print(f'median absolute percentage error: {error_to_sales.median() * 100 :.2f}%')

    error_to_stock = pd.Series(np.abs(y_pred - y_true) / combined_test_data['num_of_units'])
    print(f'mean absolute percentage of stock error: {error_to_stock.mean() * 100 :.2f}%')
    print(f'median absolute percentage of stock error: {error_to_stock.median() * 100 :.2f}%')

    interval = np.append(np.arange(0.025, 0.125, 0.025), 0.2)

    sample_size = len(y_pred)

    if sample_size > 0:

        print('Error compared to Sales:')
        for t in interval:
            correct_rate = len(error_to_sales[error_to_sales <= t]) / sample_size
            print(f'correct rate (error <= {t * 100: .0f}%): {correct_rate * 100: .2f}%')

        print(f'-' * 20)
        print('Error compared to Stock:')
        for t in interval:
            correct_rate = len(error_to_stock[error_to_stock <= t]) / sample_size
            print(f'correct rate (error <= {t * 100: .0f}%): {correct_rate * 100: .2f}%')


combined_test_data = pd.DataFrame()
for l in [True, False]:

    print(f'-' * 40)
    print(f'test group with label {l}')

    label_set = data[data['first_layer_label'].apply(lambda a: True if a is l else False)]

    combined_test_data = pd.concat(
        [
            combined_test_data,
            iter_test(to_test_data=label_set, first_label=l)
        ]
    )

pred = combined_test_data[pred_sales_col].values
true = combined_test_data[sales_col].values
calculate_error(pred, true)

fig, ax = scatter_plot_with_reg_and_label(
    data=combined_test_data,
    x_col=sales_rate_col,
    y_col=pred_sales_rate_col,
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
    y1=degree_45 + 0.1,
    y2=degree_45 - 0.1,
    color='red',
    alpha=0.2
)

ax.set_title(f'Launch Weekend Sales Rate Prediction')

print()
