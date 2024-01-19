import pickle
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from constants.redshift import query_data
from constants.utils import NatureD, NatureL, set_plot_format
from unit_ranking.CoreRanker import CoreRanker
from unit_ranking.RankerEvaluator import RankerEvaluator, RankerComparative, tables_dir, figures_dir

set_plot_format(plt)


def print_in_green_bg(text):
    print('\x1b[6;30;42m' + text + '\x1b[0m')


@dataclass
class CatAVMRanker(CoreRanker):
    rank_args = dict(ascending=True, method='dense')

    def __init__(
            self,
            target: Literal['label', 'ranking'],
            metrics: Literal['relative', 'absolute'],
            min_year: int,
            include_stack_index: bool
    ):
        super().__init__(target, metrics, min_year=min_year)

        self.model = {
            'label': CatBoostClassifier(silent=True, random_seed=42),
            'ranking': CatBoostRanker(silent=True, random_seed=42)
        }.get(self.target)

        self.ranking_features = [
                                    self.actual_quantum_price,
                                    self.raw_avm_psf,
                                    self.price_difference_col,
                                    # self.ranking_col
                                ] + ['stack_index'] if include_stack_index else []

        self.__post_init__()

    def __post_init__(self):

        self.raw_data = self._get_raw_data()
        self.projects = self.raw_data['project_name'].sort_values().unique()
        self.train_projects, self.test_projects = train_test_split(
            self.projects, test_size=0.3, random_state=42
        )

    @property
    def hedonic_features(self):

        return [
            'floor_area_sqft',
            'address_floor_num',
            'top_floor',
            'latitude',
            'longitude',
        ]

    @property
    def market_features(self):

        return [
            'transaction_year',
            'new_preference_ratio_neighborhood',
            'new_sale_time_neighborhood',
            'rolling_12m_avg_elasticity_neighborhood',
            'compound_sora__1_month',
            'compound_sora__3_month',
            'compound_sora__6_month',
            'compound_sora__9_month',
            'compound_sora_std_dev',
        ]

    @property
    def geo_features(self):

        return [
            'km_to_sg_cbd',
            'num_of_bus_stops',
            'num_of_good_schools'
        ]

    @property
    def control_features(self):
        return self.ranking_features + self.hedonic_features + self.geo_features

    def _get_core_raw_data(self, retrain=False):
        raw_data_path = tables_dir + f'avm_ranker_data_{self.min_year}.plk'

        if not retrain:
            raw_data = pickle.load(open(raw_data_path, 'rb'))
        else:
            from constants.redshift import query_data

            raw_data = query_data(
                self.query_scripts +
                f"""
                select 
                a.*,
                {','.join(self.hedonic_features + self.geo_features)}
                from {self.sql_ranking_table_name} a
                join (
                    select 
                        dw_property_id, 
                        {','.join(self.hedonic_features)}
                    from data_science.ui_master_sg_transactions_view_filled_features_condo
                    where transaction_sub_type = 'new sale'
                )
                    using (dw_property_id)
                join (
                    select 
                        dw_building_id, 
                        {','.join(self.geo_features)}
                    from data_science.ui_master_sg_building_view_filled_features_condo
                ) b
                    using (dw_building_id)
                order by 1, 2, 3
            """
            )
            pickle.dump(raw_data, open(raw_data_path, 'wb'))

        return raw_data

    def _get_raw_data(self, retrain=False, index_mode: Literal['local', 'global'] = None):

        raw_data = self._get_core_raw_data(retrain=retrain)

        if 'stack_index' in self.ranking_features:
            sql_filter = f"""
                        where dw_project_id in ({','.join([i.__repr__() for i in raw_data['dw_project_id']])})
                    """

            index_table = 'sg_condo_stack_index_resale' \
                if index_mode == 'global' else 'sg_condo_local_stack_index_resale'

            stack_index_table = query_data(
                f"""
                select 
                    dw_project_id, 
                    address_stack,
                    coef as stack_index
                from developer_tool.{index_table}
                {sql_filter}
                """
            )

            raw_data = raw_data.merge(
                stack_index_table, on=['dw_project_id', 'address_stack'], how='left'
            )

        raw_data[self.actual_ranking] = raw_data.groupby(
            self.groupby_keys)['days_on_market'].rank(**self.rank_args)

        raw_data[self.absolute_diff] = raw_data[self.actual_psf] - raw_data[self.raw_avm_psf]
        raw_data[self.relative_diff] = raw_data[self.actual_psf] / raw_data[self.raw_avm_psf] - 1
        raw_data[self.actual_quantum_price] = raw_data[self.actual_psf] * raw_data['floor_area_sqft']

        return raw_data

    def fit(self):
        train = self.raw_data[self.raw_data['project_name'].isin(self.train_projects)].copy()
        train_func = {
            'label': self._fit_classifier,
            'ranking': self._fit_ranker,
        }.get(self.target, lambda a: a)

        train_func(train)

        return self

    def _fit_classifier(self, train):
        X = train[self.control_features]
        y = train[self.actual_y]

        self.features_importance = pd.Series(
            self.model.get_feature_importance(),
            self.control_features
        ).sort_values(ascending=False)

        return self

    def _fit_ranker(self, train):
        train['group'] = train['project_name'] + '_' + train['num_of_bedrooms'].astype(str)

        #  to avoid Error: queryIds should be grouped
        train = train.sort_values('group')

        train_pool = Pool(
            data=train[self.control_features],
            label=train[self.actual_y],
            group_id=train['group']
        )

        # https://catboost.ai/en/docs/references/training-parameters/common#monotone_constraints
        cons = ()
        for cf in self.control_features:
            if cf == self.actual_psf or cf == self.actual_quantum_price:
                v = 1
            elif cf == self.raw_avm_psf or cf == self.adj_avm_psf:
                v = 0
            elif cf == self.price_difference_col:
                v = 1
            else:
                v = 0

            cons += (v,)

        self.model.set_params(**{'monotone_constraints': list(cons)}).fit(train_pool)

        self.model.fit(train_pool)

        # print_in_green_bg('Evaluating training data...')
        # predicted_train = self._test_ranker(train)
        # self.evaluate(predicted_train)

        self.features_importance = pd.Series(
            self.model.get_feature_importance(train_pool),
            self.control_features
        ).sort_values(ascending=False)

    def calculate_shap_values(self, subset):

        subset['group'] = subset['project_name'] + '_' + subset['num_of_bedrooms'].astype(str)
        subset = subset.sort_values('group')

        subset_pool = Pool(
            data=subset[self.control_features],
            label=subset[self.actual_y],
            group_id=subset['group']
        )

        shap_array = self.model.get_feature_importance(subset_pool, type='ShapValues')
        shap_df = pd.DataFrame(
            shap_array,
            index=subset.index,
            columns=self.control_features + ['group']
        )

        return subset_pool, shap_df

    def plot_feature_shap_value(
            self,
            subset,
            shap_df,
            feature,
            ax,
            q_percent=None
    ):

        if feature != self.price_difference_col:
            subset[f'shap_{feature}'] = shap_df[feature]
            x_label = feature
        else:
            if self.price_difference_col in self.ranking_features:
                subset[f'shap_{feature}'] = shap_df[feature]
            else:
                subset[f'shap_{feature}'] = shap_df[self.actual_psf] + shap_df[self.raw_avm_psf]
            x_label = 'transaction psf / avm psf - 1'

        subset[f'rank_shap_{feature}'] = subset.groupby(self.groupby_keys)[f'shap_{feature}'].rank(
            **self.rank_args,
            pct=True)

        colors = {
            f'rank_shap_{feature}': NatureL['red'],
            f'{self.actual_ranking}_pct': NatureD['blue']
        }

        subset[f'{self.actual_ranking}_pct'] = subset.groupby(
            self.groupby_keys)['days_on_market'].rank(**self.rank_args, pct=True)

        for y in [f'rank_shap_{feature}', f'{self.actual_ranking}_pct']:

            common_args = dict(
                data=subset,
                x=feature,
                y=y,
                ax=ax,
                color=colors[y],
                label=y
            )

            try:
                sns.regplot(
                    **common_args,
                    order=2,
                    scatter_kws=dict(alpha=0.7),
                    ci=None,

                )

            except np.linalg.LinAlgError:
                sns.scatterplot(
                    **common_args,
                    alpha=0.7,
                )

            if q_percent:
                plt.hlines(
                    xmin=subset[feature].min(),
                    xmax=subset[feature].max(),
                    y=q_percent,
                    colors=NatureD['orange']
                )

            ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel('local rank of shap value')
        ax.set_title(feature)

        return ax

    def plot_partial_shap_values(self, project_name, text=None, bed_num=None, q_percent=None):

        subset = self.raw_data[self.raw_data['project_name'] == project_name].copy()

        if len(subset) < 5:
            return self

        sliced_subset_pool, shap_df = self.calculate_shap_values(subset)

        bed_list = [bed_num] if bed_num else np.arange(1, 5)

        for bed in bed_list:

            bed_subset = subset[subset['num_of_bedrooms'] == bed].copy()

            if len(bed_subset) <= 3:
                continue

            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            for idx, feature in enumerate(self.ranking_features + self.hedonic_features[:2]):
                ax = plt.subplot(2, 3, idx + 1)
                ax = self.plot_feature_shap_value(bed_subset, shap_df, feature, ax, q_percent=q_percent)

                if text:
                    ax.text(
                        0.8,
                        0.8,
                        text,
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes
                    )

            title = f'partial shap values plot {project_name} {bed}-bedroom'
            plt.suptitle(title)
            plt.savefig(figures_dir + f'/partial_shap_value/{title}.png', dpi=300)
            plt.close()

    def test(self, include_train=False):

        if include_train:
            test = self.raw_data
        else:
            test = self.raw_data[self.raw_data['project_name'].isin(self.test_projects)].copy()

        test_func = {
            'label': self._test_classifier,
            'ranking': self._test_ranker,
        }.get(self.target, lambda a: a)

        test = test_func(test)

        return self.evaluate(test, include_train=include_train)

    def _test_ranker(self, test):
        test['group'] = test['project_name'] + '_' + test['num_of_bedrooms'].astype(str)

        #  to avoid Error: queryIds should be grouped
        test = test.sort_values('group')

        test_pool = Pool(
            data=test[self.control_features],
            label=test[self.actual_y],
            group_id=test['group']
        )
        test['pred_score'] = self.model.predict(test_pool)
        test[self.pred_y] = test.groupby(self.groupby_keys)['pred_score'].rank(**self.rank_args)
        score = self.model.score(test_pool)
        print(f'accuracy score {score * 100: .2f}%')

        quantity = test.groupby(self.groupby_keys)[self.actual_label].apply(
            lambda a: len(a[a == True])
        )

        def rank_to_label(rank_num, project_name, num_of_bedrooms):
            if rank_num <= quantity.loc[(project_name, num_of_bedrooms)]:
                return True
            else:
                return False

        test['pred_label'] = test.apply(
            lambda df: rank_to_label(
                df[self.pred_y],
                df['project_name'],
                df['num_of_bedrooms']
            ),
            axis=1
        )

        return test

    def _test_classifier(self, test):
        X = test[self.control_features]
        test[self.pred_y] = self.model.predict(X)
        return test

    def evaluate(self, test, include_train=False):

        evaluator = RankerEvaluator(
            raw_data=test,
            actual_label=self.actual_label,
            pred_label=self.pred_label,
            groupby_keys=self.groupby_keys,
            price_difference_col=self.price_difference_col
        )

        # evaluator.plot_confusion_matrix()
        # evaluator.compare_prediction_and_random()

        fig, ax = evaluator.plot_project_level_u_curve(control_stock=200)
        title = f'project level u curve {"train + test" if include_train else "test"}'

        ax.set_title(title)
        plt.savefig(figures_dir + f'u_curve/{title}.png', dpi=300)

        projects_score_table = evaluator.projects_score_table
        projects_bed_score_table = evaluator.projects_bed_score_table
        mid_big_projects = projects_score_table[projects_score_table['stock'] > 200]

        print(projects_score_table)
        print(f"overall gain {projects_score_table['gain'].mean() * 100 :g}%")
        print(f"mid-big projects gain {mid_big_projects['gain'].mean() * 100 :g}%")

        sorted_table = projects_bed_score_table[projects_bed_score_table['stock'] > 50].sort_values(
            ['gain', 'avg_price_diff'], ascending=[False, True]
        )

        if not include_train:
            interested_rows = pd.concat([sorted_table[:20], sorted_table[-20:]], ignore_index=True)
            for idx, row in interested_rows.iterrows():
                project_name = row.project_name
                self.plot_partial_shap_values(
                    project_name,
                    text=f"score: {row.score * 100:.2f}% \n gain: {row.gain * 100:.2f}%",
                    bed_num=row.num_of_bedrooms,
                    q_percent=row.quantity/row.stock
                )

        return projects_score_table


if True:
    common_params = dict(
        target='ranking', metrics='relative', min_year=2010
    )

    # m1 = CatAVMRanker(**common_params, include_stack_index=False).fit()
    m2 = CatAVMRanker(**common_params, include_stack_index=True).fit()
    # r = RankerComparative(
    #     m1,
    #     m2
    # )
    # r.compare(min_stock=200)
    # st = m2.test(include_train=False)
    ...
    m2.test(include_train=False)
    m2.test(include_train=True)

    ...
