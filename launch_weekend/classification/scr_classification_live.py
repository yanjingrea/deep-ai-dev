import pandas as pd
import shap
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier, Pool

from constants.utils import get_output_dir
from launch_weekend.classification.training_data import data

output_dir = get_output_dir(__file__) + 'live/'

hold_on_projects = ['Lentoria', 'The Hill @ One North']
hold_on_price = [2050, 2500]
comparable_projects = [
    'Lentor Modern',
    'Lentor Hills Residences',
    'Hillock Green',
    'Amo Residence',
    'One-North Eden',
]

project_key = 'project_display_name'

n_groups = 10
train_min_year = 2019

data = data.dropna(subset=['activity_date'])

threshold = data['sales_rate'].quantile(0.75)

data = data[data['residential_unit_count'] >= 75].copy()
data = data[data['launch_year'] >= train_min_year].copy()
data['label'] = data['sales_rate'].apply(lambda a: True if a > threshold else False)


features = [
    'launch_year',
    'price_psf',
    'tenure_type',
    'tenure_int',
    # 'latitude',
    # 'longitude',
    'building_count',
    # 'unit_count',
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
    'zone',
    # 'neighborhood',
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

y_col = ['label']
X_cols = features

num_cols = data[X_cols]._get_numeric_data().columns
cat_cols = list(set(X_cols).difference(num_cols))


construct_pool = lambda dataset: Pool(
    data=dataset[X_cols].values,
    label=dataset[y_col].values,
    feature_names=X_cols,
    cat_features=cat_cols
)

seed = 123

# train test split
test_p = hold_on_projects
train_p = set(data['project_display_name'].unique()).difference(test_p)

train = data[data[project_key].isin(train_p)].copy()
train = train.dropna(subset=y_col + cat_cols)
test = data[data[project_key].isin(test_p)].copy()

for proj, price in zip(hold_on_projects, hold_on_price):
    test.loc[test['project_display_name'] == hold_on_projects, 'price_psf'] = hold_on_price

train_pool = construct_pool(train)
test_pool = construct_pool(test)

# model

model = CatBoostClassifier(random_state=seed).fit(
    train_pool,
    verbose=False
)

y_true = test[y_col]
y_pred = model.predict(test_pool)
y_pred = pd.Series(y_pred).apply(lambda a: False if a == 'False' else True)

test['first_layer_label'] = y_pred

if True:

    explainer = shap.Explainer(
        model,
        feature_names=X_cols
    )


    def plot_waterfall(*, waterfall_pool, waterfall_data):

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

            title = f'{y_col[0].replace(" ", "_")} prediction ' + p

            ax.set_title(
                title +
                f'\nactual_{y_col[0]}: {waterfall_data["label"].iloc[idx]}' +
                f'\npredictive_{y_col[0]}: {waterfall_data["first_layer_label"].iloc[idx]}'
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


full_pool = construct_pool(data)
data['first_layer_label'] = pd.Series(model.predict(full_pool)).apply(lambda a: False if a == 'False' else True)

first_layer = data[['project_display_name', 'first_layer_label']]
first_layer.to_csv('first_layer_label.csv', index=False)
