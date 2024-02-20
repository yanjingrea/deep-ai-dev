from dataclasses import dataclass
from typing import Optional, Union, Literal, Any

import pandas as pd
import shap
from matplotlib import pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
from statsmodels.tools.typing import ArrayLike

from xgboost import XGBRegressor

from constants.redshift import query_data

from launch_weekend.scatter_plot import scatter_plot_with_reg_and_label

ListLike = Union[pd.DataFrame, pd.Series, np.ndarray, list, set]


@dataclass
class EvolutionModel:

    y_col: Literal['sales', 'sales_rate']

    min_year: Optional[int] = 2010
    min_stock: Optional[int] = 50

    sales_col: Optional[str] = 'sales'
    initial_x_cols = [
        'launch_year',
        'average_launch_psf',
        'num_of_units',
        # 'longitude',
        # 'latitude',
        'ref_launch_psf',
        # 'ref_num_of_units',
        'ref_sales_rate',
        # 'distance',
        'days_gap',
        'initial_projects',
        # 'project_zero_rm_percentage',
        'project_one_rm_percentage',
        # 'project_two_rm_percentage',
        # 'project_three_rm_percentage',
        # 'project_four_rm_percentage',
        'project_five_rm_percentage',
        # 'km_to_sg_cbd',
        # 'num_of_bus_stops',
        'meters_to_mrt',
        # 'num_of_good_schools',
        'nearby_average_psf',
        'nearby_num_of_remaining_units',
        # 'latest_half_year_nearby_launched_units',
        # 'latest_half_year_nearby_sold_units',
        # 'latest_half_year_nearby_average_psf'
    ]

    sales_rate_col = 'sales_rate'
    pred_sales_col = 'pred_' + sales_col
    pred_sales_rate_col = 'pred_' + sales_rate_col

    project_key = 'project_display_name'

    def query_raw_data(self):

        with open(
                'evolution_training_data.sql',
                'r'
        ) as sql_file:
            sql_script = sql_file.read()

        raw_data = query_data(sql_script)

        raw_data['launch_year'] = raw_data['launch_date'].apply(lambda a: int(a[:4]))
        raw_data = raw_data[raw_data['launch_year'] >= self.min_year].copy()

        return raw_data

    def process_raw_data(self):
        raw_data = self.query_raw_data()

        raw_data = raw_data.dropna(subset=[self.y_col])
        raw_data = raw_data[
            (raw_data['residential_unit_count'] > self.min_stock)
        ]

        def locate_source_projects(dataset):

            dataset = dataset.sort_values('launch_date')
            comparable_projects = dataset['project_display_name'].unique()

            initial_projects = []
            clusters_dict = {}
            sources = []

            for idx, row in dataset.iterrows():

                project = row['project_display_name']
                ref_project = row['ref_projects']

                if (ref_project not in comparable_projects) or (ref_project is None):
                    initial_projects += [project]
                    clusters_dict[project] = [project]
                    sources += [project]

                else:

                    for k, v in clusters_dict.items():

                        if ref_project in v:
                            clusters_dict[k] += [project]
                            sources += [k]
                            break

            dataset['initial_projects'] = sources

            return dataset['initial_projects'].sort_index()

        all_beds = raw_data['num_of_bedrooms'].unique()

        for b in all_beds:
            mask = raw_data['num_of_bedrooms']==b
            raw_data.loc[mask, 'initial_projects'] = locate_source_projects(raw_data.loc[mask, :])

        for bed_text in [
            'zero',
            'one',
            'two',
            'three',
            'four',
            'five'
        ]:

            raw_data[f'project_avg_size_of_{bed_text}_rm'] = raw_data[f'project_avg_size_of_{bed_text}_rm'].replace(
                -1, 0
            )

        raw_data['meters_to_mrt'] = raw_data['meters_to_mrt'].fillna(3000)

        if True:

            num_col = raw_data[self.initial_x_cols]._get_numeric_data().columns
            cat_col = list(set(self.initial_x_cols).difference(num_col))

            preprocessor = ColumnTransformer(
                [
                    # ("numerical", StandardScaler(),
                    #  list(set(num_col).difference(['num_of_bedrooms', 'num_of_units', 'launch_year']))),
                    ("categorical", OneHotEncoder(sparse_output=False), cat_col),
                ],
                verbose_feature_names_out=False,
                remainder="passthrough"
            )

            values = preprocessor.fit_transform(
                raw_data[[self.project_key, self.y_col] + self.initial_x_cols]
            )

            data = pd.DataFrame(
                data=values,
                columns=preprocessor.get_feature_names_out()
            )

        else:
            data = raw_data[[self.project_key, self.y_col] + self.initial_x_cols]

        return data

    def __post_init__(self):

        self.pred_y_col = 'pred_' + self.y_col

        self.data = self.process_raw_data()
        self.final_x_cols = self.data.columns.difference([self.project_key, self.y_col])
        self.features_importance = pd.DataFrame()

    @property
    def unique_projects(self):
        return self.data[self.project_key].unique()

    def fit_and_test_regressor(
        self,
        *,
        data,
        train_projects: ListLike,
        test_projects: ListLike
    ):

        # test = test[~test['ref_average_launch_psf'].isna()].copy()
        # test = test[test[self.sales_col] != 0].copy()

        train = data[data[self.project_key].isin(train_projects)].copy()
        test = data[data[self.project_key].isin(test_projects)].copy()

        if True:

            if False:

                from catboost import CatBoostRegressor

                model = CatBoostRegressor(
                    random_state=123,
                    verbose=False
                ).fit(
                    train[self.final_x_cols].values,
                    y=train[self.y_col].values
                )

                features_importance = pd.Series(
                    model.feature_importances_,
                    index=model.feature_names_
                )

            else:
                model = XGBRegressor(random_state=42).fit(
                    train[self.final_x_cols].astype(float),
                    y=train[self.y_col].astype(float)
                )

                features_importance = pd.Series(
                    model.feature_importances_,
                    index=model.feature_names_in_
                )
                # .sort_values(ascending=False)

        else:

            from sklearn.linear_model import LinearRegression
            train = train.dropna(subset=self.final_x_cols)
            test = test.dropna(subset=self.final_x_cols)

            model = LinearRegression(fit_intercept=True).fit(
                train[self.final_x_cols].astype(float),
                train[self.y_col].astype(float)
            )

            features_importance = pd.Series(
                model.coef_,
                index=model.feature_names_in_
            )

        self.features_importance = pd.concat(
            [self.features_importance,
             features_importance], axis=1
        )

        if test.empty:
            return pd.DataFrame()

        raw_pred_y = model.predict(test[self.final_x_cols].astype(float).values)

        if False:
            self.plot_projects_waterfall(
                model,
                test
            )

        test[self.pred_y_col] = np.clip(
            raw_pred_y,
            0,
            test['num_of_units'] if self.y_col == 'sales' else 1
        )

        if self.y_col == 'sales_rate':
            test[self.sales_col] = test[self.sales_rate_col] * test['num_of_units']
            test[self.pred_sales_col] = test[self.pred_sales_rate_col] * test['num_of_units']

        output_cols = [
            self.sales_col,
            self.pred_sales_col,
            'num_of_units'
        ]

        test_projects_data = test.groupby(
            [self.project_key],
            as_index=False
        )[output_cols].mean().convert_dtypes()

        test_projects_data = test_projects_data.groupby(
            [self.project_key],
            as_index=False
        )[output_cols].sum().convert_dtypes()

        test_projects_data[self.sales_rate_col] = test_projects_data[self.sales_col] / test_projects_data[
            'num_of_units']
        test_projects_data[self.pred_sales_rate_col] = test_projects_data[self.pred_sales_col] / test_projects_data[
            'num_of_units']

        return test_projects_data

    def iter_test(self, *, n_groups=10, test_min_year=None):

        if test_min_year:
            test_projects = self.data[self.data['launch_year'] >= test_min_year][self.project_key].unique()
        else:
            test_projects = self.unique_projects

        n_unique_projects = len(test_projects)
        n_samples = n_unique_projects // n_groups
        idx_list = np.arange(0, n_unique_projects + n_samples, n_samples)

        combined_test_data = pd.DataFrame()

        for i, idx_num in enumerate(idx_list):

            if i == 0:
                continue

            test_p = test_projects[idx_list[i - 1]: idx_num]
            train_p = set(test_projects).difference(test_p)

            common = dict(
                train_projects=train_p,
                test_projects=test_p
            )

            res = self.fit_and_test_regressor(
                data=self.data,
                **common
            )

            combined_test_data = pd.concat([combined_test_data, res], ignore_index=True)

        return combined_test_data

    def plot_projects_waterfall(
        self,
        model,
        test_data
    ):
        plt.rcParams.update({"font.size": "10"})

        test_data = test_data.copy().reset_index()

        X = test_data[self.final_x_cols].astype(float)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        for idx in test_data.index:
            p = test_data['project_display_name'].loc[idx]
            bed = test_data['num_of_bedrooms'].loc[idx]
            actual_sales = test_data[self.y_col].loc[idx]

            ax = shap.plots.waterfall(shap_values[idx], max_display=20, show=False)

            title = f'{self.y_col.replace(" ", "_")} prediction ' + p + f' {bed}-bedroom'
            ax.set_title(title + f'\nactual_{self.y_col}: {int(actual_sales)}')

            plt.savefig(
                f'/Users/wuyanjing/PycharmProjects/app/launch_weekend/output/'
                f'{title.replace(" ", "_")}.png',
                dpi=300
            )

            plt.close()

    def evaluate(self, *, n_groups=10, test_min_year=None):

        combined_test_data = self.iter_test(
            n_groups=n_groups,
            test_min_year=test_min_year
        )

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

        pred = combined_test_data[self.pred_sales_col].values
        true = combined_test_data[self.sales_col].values
        calculate_error(pred, true)

        print(f'-' * 20)
        print('Features Importance:')
        avg_features_importance = self.features_importance.mean(axis=1).sort_values(ascending=False)
        print(avg_features_importance)

        fig, ax = scatter_plot_with_reg_and_label(
            data=combined_test_data,
            x_col=self.sales_rate_col,
            y_col=self.pred_sales_rate_col,
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


if True:
    EvolutionModel(
        min_stock=75,
        min_year=2015,
        y_col='sales_rate'
    ).evaluate(test_min_year=2020)
