import pickle
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from constants.utils import NatureD, NatureL, set_plot_format
from unit_sequence.RankerEvaluator import RankerEvaluator, tables_dir, figures_dir, plot_random_u_curve

from os.path import dirname

set_plot_format(plt)


def print_in_green_bg(text):
    print('\x1b[6;30;42m' + text + '\x1b[0m')


@dataclass
class RankerParams:
    actual_psf: Optional[str] = 'listing_price_psf'
    actual_quantum_price: Optional[str] = 'listing_price'
    raw_avm_psf: Optional[str] = 'avm_price_psf'

    actual_label: Optional[str] = 'actual_label'
    actual_ranking: Optional[str] = 'actual_ranking'

    pred_label: Optional[str] = 'pred_label'
    pred_ranking: Optional[str] = 'pred_ranking'


@dataclass
class CatBoostAVMRanker:

    target: Literal['label', 'ranking']
    metrics: Literal['relative', 'absolute']
    min_year: Optional[int] = 2015

    actual_psf: Optional[str] = 'listing_price_psf'
    actual_quantum_price: Optional[str] = 'listing_price'
    avm_psf: Optional[str] = 'avm_price_psf'

    actual_label: Optional[str] = 'actual_label'
    actual_ranking: Optional[str] = 'actual_ranking'

    pred_label: Optional[str] = 'pred_label'
    pred_ranking: Optional[str] = 'pred_ranking'

    relative_diff: Optional[str] = 'price_relative_difference'
    absolute_diff: Optional[str] = 'price_absolute_difference'

    days_on_market: Optional[str] = 'days_on_market'

    rank_args = dict(ascending=True, method='dense')
    groupby_keys = ['project_name', 'num_of_bedrooms']

    def __post_init__(self):

        self.raw_data = self._get_raw_data(retrain=True)
        self.price_difference_col = f'price_{self.metrics}_difference'

        self.model = {
            'label': CatBoostClassifier(silent=True, random_seed=42),
            'ranking': CatBoostRanker(silent=True, random_seed=42)
        }.get(self.target)

        self.actual_y = {
            'label': self.actual_label,
            'ranking': self.actual_ranking
        }.get(self.target, lambda a: a)

        self.pred_y = 'pred_' + self.target

        self.ranking_features = [
            self.actual_quantum_price,
            self.avm_psf,
            self.price_difference_col
        ]

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
    def geo_features(self):

        return [
            'km_to_sg_cbd',
            'num_of_bus_stops',
            'num_of_good_schools'
        ]

    @property
    def control_features(self):
        return self.ranking_features + self.hedonic_features + self.geo_features

    def _get_core_raw_data(self, retrain=True):
        raw_data_path = tables_dir + f'avm_ranker_data_{self.min_year}.plk'

        if not retrain:
            raw_data = pickle.load(open(raw_data_path, 'rb'))
        else:
            from constants.redshift import query_data

            raw_data = query_data(
            f"""
            with
                base_final_listing_price as (
                                                with
                                                    base_trans as (
            
                                                                      select
                                                                          *,
                                                                          min(transaction_date::date) over (partition by dw_project_id) as project_launch_date,
                                                                                  row_number()
                                                                                  over (partition by dw_property_id order by transaction_date desc) as seq
                                                                      from data_science.ui_master_sg_transactions_view_filled_features_condo a
                                                                           join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                                using (dw_project_id)
                                                                      where a.property_type_group = 'Condo'
                                                                        and transaction_sub_type = 'new sale'
                                                                  ),
                                                    base_index as (
                                                                      select
                                                                          transaction_month_index,
                                                                          hi_avg_improved
                                                                      from data_science.ui_master_daily_sg_index_sale
                                                                      where property_group = 'condo'
                                                                        and index_area_type = 'country'
                                                                  ),
                                                    base_trans_price as (
                                                                            select
                                                                                dw_project_id,
                                                                                dw_property_id,
                                                                                transaction_date,
                                                                                project_launch_date,
                                                                                        unit_price_psf /
                                                                                        rebase_index.hi_avg_improved *
                                                                                        adjust_index.hi_avg_improved as transaction_price_psf,
                                                                                        unit_price_psf /
                                                                                        rebase_index.hi_avg_improved *
                                                                                        adjust_index.hi_avg_improved as transaction_price,
                                                                                floor_area_sqft
                                                                            from base_trans a
                                                                                 join base_index as rebase_index
                                                                                      using (transaction_month_index)
                                                                                 join data_science.ui_master_sg_project_geo_view_filled_features_condo c
                                                                                      using (dw_project_id)
                                                                                 join base_index as adjust_index
                                                                                      on c.project_launch_month =
                                                                                         adjust_index.transaction_month_index
                                                                                          and seq = 1
                                                                            order by c.project_name, unit
                                                                        ),
                                                    base_listing_price as (
                                                                              select
                                                                                  dw_project_id,
                                                                                  property_dwid as dw_property_id,
                                                                                  developer_price,
                                                                                  row_number() over (partition by property_dwid order by update_date desc) as seq
                                                                              from raw_reference.sg_new_launch_developer_price a
                                                                                   join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                                        on a.project_dwid = b.dw_project_id
                                                                                            and a.update_date <=
                                                                                                to_date(b.project_launch_month, 'YYYYMM')
                                                                                            and developer_price is not null
                                                                          )
                                                select
                                                    dw_property_id,
                                                    transaction_date,
                                                    project_launch_date,
                                                    case
                                                        when developer_price is null
                                                            then transaction_price_psf
                                                        else developer_price / floor_area_sqft
                                                        end as listing_price_psf,
                                                    case
                                                        when developer_price is null
                                                            then transaction_price
                                                        else developer_price
                                                        end as listing_price
                                                from base_trans_price
                                                     full outer join base_listing_price
                                                                     using (dw_property_id)
                                                where project_launch_date >= '2015-01-01'
                                            ),
                base_final_avm_price as (
                                            with
                                                base_avm_price as (
                                                                      select
                                                                          dw_property_id,
                                                                          unit_price_psf as daily_avm_price_psf,
                                                                          update_date,
                                                                          row_number() over (partition by dw_property_id order by update_date desc) as seq
                                                                      from data_science.master_daily_sale_valuation_sg_combined a
                                                                           join data_science.ui_master_sg_properties_view_filled_static_features_condo
                                                                                using (dw_property_id)
                                                                           join data_science.ui_master_sg_project_geo_view_filled_features_condo b
                                                                                using (dw_project_id)
                                                                      where update_method = 'no_comparables_ml'
                                                                        and property_group = 'condo'
                                                                        and (
                                                                                  to_date(update_date, 'YYYYMMDD') <=
                                                                                  to_date(project_launch_month, 'YYYYMM')
                                                                              or dw_project_id = 'f7fea87e1a59592e817cd84c2b416bb4'
                                                                          )
                                                                  ),
                                                base_his_avm_price as (
                                                                          select
                                                                              dw_property_id,
                                                                              xgb_avg_pred_psf
                                                                          from developer_tool.sg_condo_properties_estimate_launch_price
                                                                      )
                                            select
                                                dw_property_id,
                                                case
                                                    when daily_avm_price_psf is null then xgb_avg_pred_psf
                                                    else daily_avm_price_psf end as avm_price_psf,
                                                avm_price_psf * floor_area_sqft as avm_price,
                                                update_date
                                            from data_science.ui_master_sg_properties_view_filled_static_features_condo
                                            left join (
                                                     select
                                                         *
                                                     from base_avm_price
                                                     where seq = 1
                                                 ) as avm_price
                                                using (dw_property_id)
                                            left join base_his_avm_price
                                                using (dw_property_id)
                                            order by 1, 2
                                        ),
                geo_features as (
                                    select
                                        dw_property_id,
                                        case when p.latitude is null then m.latitude else p.latitude end as filled_latitude,
                                        case when p.longitude is null then m.longitude else p.longitude end as filled_longitude,
                                        case
                                            when b.km_to_sg_cbd is null then round(
                                                                                     st_distancesphere(
                                                                                             ST_point(filled_latitude, filled_longitude),
                                                                                             ST_point(103.851463066212, 1.2839332623453799)
                                                                                     )
                                                                             ) / 1000
                                            else b.km_to_sg_cbd end as filled_km_to_sg_cbd,
                                        num_of_bus_stops,
                                        num_of_good_schools
                                    from data_science.ui_master_sg_properties_view_filled_static_features_condo p
                                         join data_science.ui_master_sg_building_view_filled_features_condo b
                                              using (dw_building_id)
                                         join (
                                                  select
                                                      project_dwid as dw_project_id,
                                                      project_display_name,
                                                      latitude,
                                                      longitude
                                                  from masterdata_sg.address
                                                       join masterdata_sg.project
                                                            using (address_dwid)
                                              ) as m
                                              using (dw_project_id)
                                )
            select distinct
                project_display_name as project_name,
                dw_project_id,
                dw_property_id,
                unit,
                address_stack,
                address_floor_num,
                top_floor,
                num_of_bedrooms,
                floor_area_sqft,
                proj_num_of_units,
                case
                    when num_of_bedrooms = 0 then project_units_zero_rm
                    when num_of_bedrooms = 1 then project_units_one_rm
                    when num_of_bedrooms = 2 then project_units_two_rm
                    when num_of_bedrooms = 3 then project_units_three_rm
                    when num_of_bedrooms = 4 then project_units_four_rm
                    when num_of_bedrooms = 5 then project_units_five_rm
                    end
                    as num_of_units,
                transaction_date,
                project_launch_date,
                datediff(day, project_launch_date::date, transaction_date::date) as days_on_market,
                listing_price_psf,
                avm_price_psf,
                listing_price,
                avm_price,
                filled_latitude as latitude,
                filled_longitude as longitude,
                filled_km_to_sg_cbd as km_to_sg_cbd,
                num_of_bus_stops,
                num_of_good_schools
            from data_science.ui_master_sg_properties_view_filled_static_features_condo
                 left join (
                               select
                                   project_dwid as dw_project_id,
                                   project_display_name
                               from ui_app.project_summary_prod_sg
                           ) as c
                           using (dw_project_id)
                 join (select * from  data_science.ui_master_sg_project_geo_view_filled_features_condo where project_launch_month::int > 201800) d
                           using (dw_project_id)
                 left join base_final_listing_price a
                           using (dw_property_id)
                 left join base_final_avm_price b
                           using (dw_property_id)
                 left join geo_features
                           using (dw_property_id)
            order by 1, 2, address_stack, address_floor_num;
            """
            )

        return raw_data

    def _get_raw_data(self, retrain=True):

        raw_data = self._get_core_raw_data(retrain=retrain)

        # todo: manual alert
        mask = raw_data['project_name'] == 'Lumina Grand'

        manual_info = {
            'project_launch_date': pd.to_datetime('2023-01-27'),
            'latitude': 1.361184629001846,
            'longitude': 103.7411523137532,
            'km_to_sg_cbd': 14.972
        }

        for key, value in manual_info.items():
            if raw_data.loc[mask, key].isna().any():
                raw_data.loc[mask, key] = value

        local_data = pd.read_csv(
            dirname(__file__) + f'/manual_input/lumina_grand_20240124.csv'
        )

        local_data.set_index('property_dwid', inplace=True)

        raw_data.loc[mask, 'listing_price'] = raw_data.loc[mask, :].apply(
            lambda row: local_data.loc[row['dw_property_id'], 'price1'],
            axis=1
        )

        raw_data[self.actual_ranking] = raw_data.groupby(self.groupby_keys)['days_on_market'].rank(**self.rank_args)

        raw_data[self.absolute_diff] = raw_data[self.actual_psf] - raw_data[self.avm_psf]
        raw_data[self.relative_diff] = raw_data[self.actual_psf] / raw_data[self.avm_psf] - 1

        raw_data[self.actual_label] = raw_data[self.days_on_market].apply(
            lambda a: True if a <= 14 else False
        )

        for date_col in ['project_launch_date', 'transaction_date']:
            raw_data[date_col] = pd.to_datetime(raw_data[date_col])

        # raw_data[self.actual_quantum_price] = raw_data[self.actual_psf] * raw_data['floor_area_sqft']

        return raw_data

    def fit(self):
        train = self.raw_data[self.raw_data['project_name'].isin(self.train_projects)].copy()
        train_func = {
            'label': self._fit_classifier,
            'ranking': self._fit_ranker,
        }.get(self.target, lambda a: a)

        train_func(train)

        return self

    def _fit_classifier(self, training_data: pd.DataFrame):
        X = training_data[self.control_features]
        y = training_data[self.actual_y]

        self.features_importance = pd.Series(
            self.model.get_feature_importance(),
            self.control_features
        ).sort_values(ascending=False)

        return self

    def construct_pool(self, data: pd.DataFrame, drop_null_label: bool = True) -> Pool:

        # target column in training data can not contain null
        if drop_null_label:
            data = data.dropna(subset=[self.actual_y]).copy()
        else:
            data = data.copy()

        #  to avoid Error: queryIds should be grouped
        data['group'] = data['project_name'] + '_' + data['num_of_bedrooms'].astype(str)
        data = data.sort_values('group')

        # to avoid AttributeError: 'DataFrame' object has no attribute 'iteritems'
        pool = Pool(
            data=data[self.control_features].values,
            label=data[self.actual_y].values,
            group_id=data['group'].values
        )

        return pool

    def _fit_ranker(self, training_data: pd.DataFrame):

        train_pool = self.construct_pool(training_data)

        # https://catboost.ai/en/docs/references/training-parameters/common#monotone_constraints
        cons = ()
        for cf in self.control_features:
            if cf == self.actual_psf or cf == self.actual_quantum_price:
                v = 1
            elif cf == self.avm_psf:
                v = 0
            elif cf == self.price_difference_col:
                v = 1
            else:
                v = 0

            cons += (v,)

        if self.model.is_fitted():
            self.model = CatBoostRanker(silent=True, random_seed=42)

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
                subset[f'shap_{feature}'] = shap_df[self.actual_psf] + shap_df[self.avm_psf]
            x_label = 'transaction psf / avm psf - 1'

        subset[f'rank_shap_{feature}'] = subset.groupby(self.groupby_keys)[f'shap_{feature}'].rank(
            **self.rank_args,
            pct=True
        )

        colors = {
            f'rank_shap_{feature}': NatureL['red'],
            f'{self.actual_ranking}_pct': NatureD['blue']
        }

        subset[f'{self.actual_ranking}_pct'] = subset.groupby(
            self.groupby_keys
        )['days_on_market'].rank(**self.rank_args, pct=True)

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

        test = test.copy()

        test_pool = self.construct_pool(test, drop_null_label=False)
        test['pred_score'] = self.model.predict(test_pool)
        test[self.pred_y] = test.groupby(self.groupby_keys)['pred_score'].rank(**self.rank_args)

        # score = self.model.score(test_pool)
        # print(f'accuracy score {score * 100: .2f}%')

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
                    q_percent=row.quantity / row.stock
                )

        return projects_score_table

    def iteration_test(
        self,
        *,
        min_year=2020,
        min_stock=150,
        n_groups=2,
        highlight_projects: list = None
    ):

        latest_projects = self.raw_data[
            (self.raw_data['project_launch_date'].dt.year >= min_year) &
            (self.raw_data['proj_num_of_units'] >= min_stock)
            ]['project_name'].sort_values().unique()

        n_latest_projects = len(latest_projects)
        n_project_per_group = n_latest_projects // n_groups

        tested_data = pd.DataFrame()

        for i in range(0, n_latest_projects, n_project_per_group):

            temp_test_projects = latest_projects[i:i + n_project_per_group]

            temp_train_data = self.raw_data[~self.raw_data['project_name'].isin(temp_test_projects)]
            temp_test_data = self.raw_data[self.raw_data['project_name'].isin(temp_test_projects)]

            self._fit_ranker(temp_train_data)

            tested_data = pd.concat([tested_data, self._test_ranker(temp_test_data)], ignore_index=True)

        fig, ax = plot_random_u_curve()
        evaluator = RankerEvaluator(
            raw_data=tested_data,
            actual_label=self.actual_label,
            pred_label=self.pred_label,
            groupby_keys=self.groupby_keys,
            price_difference_col=self.price_difference_col
        )

        fig, ax = evaluator.plot_scatter_above_u_curve(
            fig=fig,
            ax=ax,
            control_stock=min_stock,
            highlight_projects=highlight_projects
        )

        title = f'project level u curve'
        ax.set_title(title)
        plt.savefig(figures_dir + f'u_curve/{title}.png', dpi=300)

        projects_score_table = evaluator.projects_score_table
        mid_big_projects = projects_score_table[projects_score_table['stock'] > min_stock]

        print(projects_score_table)
        print(f"overall gain {projects_score_table['gain'].mean() * 100 :g}%")
        print(f"mid-big projects gain {mid_big_projects['gain'].mean() * 100 :g}%")

        projects_score_table.to_csv(
            tables_dir + f'u_curve_projects_score_table.csv', index=False
        )

        for proj in highlight_projects:

            project_data = tested_data[tested_data['project_name'] == proj].copy()
            available_beds = project_data['num_of_bedrooms'].unique()

            for bed in available_beds:
                bed_data = project_data[project_data['num_of_bedrooms'] == bed]

                fig, ax = self.plot_tower_view(
                    project_data=bed_data,
                    values=self.pred_ranking
                )

    def plot_project_tower_view(
        self,
        *,
        project_name,
        feature_name
    ):
        project_data = self.raw_data[self.raw_data[project_name] == project_name].copy()

        available_beds = project_data['num_of_bedrooms'].unique()

        for bed in available_beds:
            bed_data = project_data[project_data['num_of_bedrooms'] == bed]

            fig, ax = self.plot_tower_view(
                project_data=bed_data,
                values=feature_name
            )

            title = f'Selling Sequence {project_name} {bed}-bedroom'
            ax.set_title(title)

            plt.savefig(figures_dir + title + '.png', dpi=300)

    @staticmethod
    def plot_tower_view(
        *,
        project_data,
        values,
        threshold=None,
        fig=None,
        ax=None
    ):

        project_trans_sorted = project_data.sort_values(by='address_stack').fillna(-1)

        pivot_data = project_trans_sorted.pivot_table(
            index=project_trans_sorted['address_floor_num'].astype(int),
            columns=project_trans_sorted['address_stack'],
            values=values,
            aggfunc='mean'
        ).iloc[::-1]
        pivot_data[pivot_data == -1] = np.nan

        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))

        common_params = dict(
            data=pivot_data, ax=ax,
            annot=True, fmt=".0f", annot_kws={"size": 8},
            linewidth=0.5, linecolor='lightgrey'
        )

        if threshold is None:

            cmap = sns.color_palette("coolwarm", as_cmap=True).copy()
            cmap.set_bad(color="lightgrey")
            sns.heatmap(
                **common_params,
                cmap=cmap
            )

        else:

            sold_mask = pivot_data <= threshold
            unsold_mask = ~sold_mask

            cmap_sold = sns.light_palette("Reds", as_cmap=True)
            cmap_unsold = sns.light_palette("Greens", as_cmap=True)

            sns.heatmap(
                **common_params,
                mask=sold_mask,
                cmap=cmap_sold
            )

            sns.heatmap(
                **common_params,
                mask=unsold_mask,
                cmap=cmap_unsold
            )

        plt.ylabel("Address Floor Number")
        plt.xlabel("Bedroom-Stack")

        return fig, ax
