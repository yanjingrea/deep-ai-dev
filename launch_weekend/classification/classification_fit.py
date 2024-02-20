import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import shap

from constants.utils import get_output_dir
from launch_weekend.classification.training_data import data
from launch_weekend.scatter_plot import scatter_plot_with_reg_and_label
output_dir = get_output_dir(__file__)




project_key = 'project_display_name'

n_groups = 10
train_min_year = 2021
test_min_year = 2021


data = data.dropna(subset=['sales_rate', 'launch_date'])

threshold = data['sales_rate'].quantile(0.75)

data = data[data['residential_unit_count'] >= 75].copy()
data = data[data['launch_year'] >= train_min_year].copy()
data['label'] = data['sales_rate'].apply(lambda a: True if a > threshold else False)
# data['label'] = pd.qcut(data['sales_rate'], 5, np.arange(1,6))


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
# data_dum = pd.get_dummies(data[features+['zone', 'label']], columns=['zone'])


construct_pool = lambda dataset: Pool(
    data=dataset[X_cols].values,
    label=dataset[y_col].values,
    feature_names=X_cols,
    cat_features=cat_cols
)

seed = 123

# train test split
# train, test = train_test_split(data, test_size=0.1, random_state=seed)

combined_test_data = data[data['launch_year'] >= test_min_year]

test_projects = combined_test_data[project_key].unique()
n_unique_projects = len(test_projects)
n_samples = n_unique_projects // n_groups
idx_list = np.arange(0, n_unique_projects + n_samples, n_samples)

features_importance = pd.DataFrame()
combined_y_true = np.array([])
combined_y_pred = np.array([])

for i, idx_num in enumerate(idx_list):

    if i == 0:
        continue

    test_p = test_projects[idx_list[i - 1]: idx_num]
    train_p = set(test_projects).difference(test_p)

    train = data[data[project_key].isin(train_p)].copy()
    test = data[data[project_key].isin(test_p)].copy()

    train = train.dropna(subset=y_col + cat_cols)
    train_pool = construct_pool(train)
    test_pool = construct_pool(test)

    # model
    # model = XGBClassifier(random_state=seed).fit(X=train[X_cols], y=train[y_col])
    model = CatBoostClassifier(random_state=seed).fit(
        train_pool,
        verbose=False
    )

    y_true = test[y_col]
    y_pred = model.predict(test_pool)
    y_pred = pd.Series(y_pred).apply(lambda a: False if a == 'False' else True)

    if False:

        explainer = shap.Explainer(
            model,
            feature_names=X_cols
        )
        shap_values = explainer(
            test_pool
        )
        shap_values.__setattr__('data',test[X_cols].values)

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

    combined_y_true = np.append(combined_y_true, y_true)
    combined_y_pred = np.append(combined_y_pred, y_pred)

    temp_features_importance = pd.Series(
        data=model.feature_importances_,
        index=model.feature_names_
    )

    features_importance = pd.concat(
        [
            features_importance,
            temp_features_importance
        ],
        axis=1
    )


score = accuracy_score(
    y_true=combined_y_true,
    y_pred=combined_y_pred
)

print(f'accuracy score: {score * 100 :.2f}%')
print(features_importance.mean(axis=1).sort_values(ascending=False) * 100)

disp = ConfusionMatrixDisplay.from_predictions(
    combined_y_true, combined_y_pred,
    cmap=plt.cm.Blues
)

title = f'model: {seed} accuracy score: {score * 100 :.2f}%'

ax = disp.ax_
ax.set_title(title)
plt.savefig(output_dir + f'model {seed}.png', dpi=300)


full_pool = construct_pool(data)
data['first_layer_label'] = pd.Series(model.predict(full_pool)).apply(lambda a: False if a == 'False' else True)

first_layer = data[['project_display_name', 'first_layer_label']]
first_layer.to_csv('first_layer_label.csv', index=False)

if False:

    data['launch_date'] = pd.to_datetime(data['activity_date'])
    data = data.sort_values('launch_date')
    comparable_projects = data['project_display_name'].unique()

    initial_projects = []
    clusters_dict = {}
    sources = []

    for idx, row in data.iterrows():

        project = row['project_display_name']
        ref_project = row['ref_project']

        if (ref_project not in comparable_projects) or (ref_project is None):
            initial_projects += [project]
            clusters_dict[project] = [project]
            sources += [project]

        else:

            for k, v in clusters_dict.items():

                attached = False

                if ref_project in v:
                    clusters_dict[k] += [project]
                    sources += [k]
                    # break
                    attached = True

            # if not attached:
            #     clusters_dict[project] = [project]


    data['launch_year'] = data['launch_date'].dt.year
    data['initial_projects'] = sources

    test_data = data[(data['launch_year'] >= 2021) & (~data['price_psf'].isna())].copy()

    y_pred = []
    for idx, row in test_data.iterrows():

        init_proj = row['initial_projects']


        label_mask = (data['first_layer_label'] == row['first_layer_label'])
        time_mask = (data['launch_date'] < row['launch_date'])
        group_mask = (data['initial_projects'] == init_proj)
        region_mask = (data['zone'] == row['zone'])

        clusters_data = data[label_mask & time_mask & group_mask]

        if clusters_data.empty:
            clusters_data = data[label_mask & time_mask & region_mask]

            if clusters_data.empty:
                clusters_data = data[label_mask & time_mask]

        if False:
            if len(clusters_data) <= 3:
                clusters_data = data[time_mask & region_mask]

                if len(clusters_data) <= 3:
                    clusters_data = data[time_mask]

            clusters_data = clusters_data.dropna(
             subset=['sales', 'average_launch_psf', 'num_of_units', 'launch_year']
            )

            Q = clusters_data['sales']
            P = clusters_data['average_launch_psf']
            S = clusters_data['num_of_units']
            T = clusters_data['launch_year']

            X = pd.DataFrame(
                dict(
                    P=P,
                    S=S
                )
            )

            y = np.log(Q)

            model = LinearRegression(fit_intercept=True).fit(X, y)

            pred_sales = np.exp(
                model.predict(
                    pd.DataFrame(
                        dict(
                            P=[row['average_launch_psf']],
                            S=row['num_of_units']
                        )
                    )
                )
            )

            pred_sales = np.clip(pred_sales, 0, row['num_of_units'])
            pred_sales_rates = pred_sales/row['num_of_units']
            y_pred += [pred_sales_rates[0]]
        else:
            pred_sales_rates = clusters_data['sales_rate'].mean()
            # pred_sales_rates = np.average(
            #     clusters_data['sales_rate'],
            #     weights=np.log(clusters_data['num_of_units']
            #     )
            # )
            y_pred += [pred_sales_rates]


    test_data['num_of_units'] = test_data['residential_unit_count']
    test_data['pred_sales_rate'] = y_pred
    test_data['pred_sales'] = test_data['pred_sales_rate'] * test_data['num_of_units']

    pred = test_data['pred_sales'].values
    true = test_data['sales'].values


    def calculate_error(q_pred, q_true):

        error_to_sales = pd.Series(q_pred[q_true != 0] / q_true[q_true != 0] - 1).abs()
        print(f'mean absolute percentage error: {error_to_sales.mean() * 100 :.2f}%')
        print(f'median absolute percentage error: {error_to_sales.median() * 100 :.2f}%')

        error_to_stock = pd.Series(np.abs(q_pred - q_true) / test_data['num_of_units'])
        print(f'mean absolute percentage of stock error: {error_to_stock.mean() * 100 :.2f}%')
        print(f'median absolute percentage of stock error: {error_to_stock.median() * 100 :.2f}%')

        interval = np.append(np.arange(0.025, 0.125, 0.025), 0.2)

        sample_size = len(q_pred)

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


    calculate_error(pred, true)

    fig, ax = scatter_plot_with_reg_and_label(
        data=test_data.reset_index(),
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
        y1=degree_45 + 0.1,
        y2=degree_45 - 0.1,
        color='red',
        alpha=0.2
    )


print()
