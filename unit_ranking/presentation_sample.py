from dataclasses import dataclass
from os.path import dirname, realpath
from typing import Literal, Union, Optional

import catboost
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import Pool
from catboost.utils import eval_metric
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from constants.redshift import query_data
from constants.utils import print_in_green_bg, set_plot_format

set_plot_format(plt)
figures_dir = dirname(realpath(__file__)) + '/output/figures/tower_view/'


@dataclass
class CARFeatures:
    price_features: Optional[list] = None
    hedonic_features: Optional[list] = None
    geo_features: Optional[list] = None

    @property
    def all_features(self):
        features = []
        for v in self.__dict__.values():

            if v is not None:
                features += v

        return features


@dataclass
class CBM:
    control_features: list
    target: str
    core_model: Union[catboost.CatBoostRegressor, catboost.CatBoostRanker, catboost.CatBoostClassifier]
    model_params = dict(silent=True, random_seed=42)
    features_importance = None

    def construct_pool_data(self, data):
        data['group'] = data['project_name'] + '_' + data['num_of_bedrooms'].astype(str)

        #  to avoid Error: queryIds should be grouped
        data = data.sort_values(['group', 'days_on_market_rank'], ascending=False)

        pool_data = Pool(
            data=data[self.control_features],
            label=data[self.target],
            group_id=data['group']
        )

        return data.index, pool_data

    def fit(self, data):
        _, pool_data = self.construct_pool_data(data)
        self.core_model.set_params(**self.model_params).fit(pool_data)

        self.features_importance = pd.Series(
            self.core_model.get_feature_importance(pool_data),
            self.control_features
        ).sort_values(ascending=False)
        print(f'\nfeature importance: \n{self.features_importance}')

        return self

    def predict(self, data):
        index, pool_data = self.construct_pool_data(data)
        res = pd.Series(
            self.core_model.predict(pool_data),
            index=index
        ).sort_index()

        if (data[self.target].dtypes == bool) & (res.dtype == object):
            res = pd.Series(res).apply(
                lambda a: {
                    'True': True,
                    'False': False,
                }.get(a)
            )

        score = self.core_model.score(pool_data)

        # a replicate of catboost built-in score function
        def ndcg(
                actual_values: pd.Series,
                predicted_values: pd.Series
        ):
            base_log_dcg = lambda a, wa: (a / np.log2(wa + 1)).sum()
            res = (
                    base_log_dcg(actual_values, predicted_values.rank(method='first', ascending=False)) /
                    base_log_dcg(actual_values, actual_values.rank(method='first', ascending=False))
            )

            return res

        data['pred_score'] = res

        groupby_data = data.groupby(['project_name', 'num_of_bedrooms'])
        c_score = groupby_data.apply(
            lambda df: eval_metric(
                df[self.target].values,
                df['pred_score'].values,
                'NDCG',
                group_id=[1, ] * len(df)
            )[0]
        )
        r_score = groupby_data.apply(
            lambda df: ndcg(df[self.target], df['pred_score'])
        )
        diff = (c_score - r_score).mean()
        print_in_green_bg(f'agg built-in score {score * 100: .2f}%')
        print(f'agg replicated score {r_score.mean() * 100: .2f}%')
        print(f'The abs average difference between replicated score and built-in score is {diff :g}')

        return res


@dataclass
class CatAVMRanker:
    features: CARFeatures
    min_year = 2012

    # field to be fitted
    # ------------------------------------------------------------
    features_importance = None
    ranker_layer = None

    @property
    def groupby_keys(self) -> list:
        return ['project_name', 'num_of_bedrooms']

    @property
    def rank_args(self) -> dict:
        return dict(ascending=True, method='first', pct=True)

    def get_data_subset(self, subset: Literal['train', 'test']):

        projects = {
            'train': self.train_projects,
            'test': self.test_projects
        }.get(subset)

        data = self.property_data[self.property_data['project_name'].isin(projects)].copy()

        return data

    def __post_init__(self):
        self.ranker_target = 'days_on_market_rank'
        self.pred_ranker_target = 'pred_' + self.ranker_target

        # -----------------------------------
        self.property_data = self.query_raw_data()
        self.projects = self.property_data['project_name'].sort_values().unique()
        self.train_projects, self.test_projects = train_test_split(
            self.projects, test_size=0.3, random_state=42
        )

    def query_raw_data(self, index_mode=None):

        transaction_price_psf = 'transaction_price_psf'
        avm_price_psf = 'avm_price_psf'
        relative_price_difference = 'relative_price_difference'

        raw_data = query_data(
            f"""
                with base_index as (
                    select 
                    transaction_month_index, 
                    hi_avg_improved
                from data_science.ui_master_daily_sg_index_sale
                where property_group = 'condo'
                    and index_area_type = 'country'
                ), base_transaction as (
                    select
                        *,
                        rebase_index.hi_avg_improved rebase_i,
                        min(transaction_date::date) over (partition by dw_project_id) as project_launch_date,
                        unit_price_psf / rebase_i as adj_unit_price_psf,
                        transaction_amount / rebase_i as adj_transaction_amount
                    from data_science.ui_master_sg_transactions_view_filled_features_condo t
                    left join base_index rebase_index
                        on t.transaction_month_index = rebase_index.transaction_month_index
                    where property_type_group = 'Condo'
                        and transaction_sub_type = 'new sale'
                        and transaction_year >= {self.min_year}
                ), base_launch_round as (
                    select *,
                    launch_month as ura_launch_month,
                    lead(launch_month, 1) over (
                        partition by project_dwid
                        order by launch_month asc
                    ) as ura_next_launch_month,
                    count(*) over (
                        partition by project_dwid
                        order by launch_month rows between unbounded preceding and current row
                    ) as ura_launch_round
                    from developer_tool.project_developer_launch b
                ), base_first_launch_round as (
                    select 
                        a.dw_project_id,
                        case when ura_next_launch_month is null
                            then getdate()
                            else ura_next_launch_month 
                            end as next_launch_month
                    from (select distinct dw_project_id, project_launch_date from base_transaction) a
                    left join (select * from base_launch_round where ura_launch_round = 1) b
                        on a.dw_project_id = b.project_dwid
                ), base_transaction_price as (
                    select distinct
                        d.project_name,
                        b.dw_project_id,
                        b.dw_property_id,
                        b.dw_building_id,
                        b.unit,
                        b.address_stack,
                        transaction_year,
                        datediff(day, project_launch_date::date ,transaction_date::date) + 1 days_on_market,
                        adjust_index.hi_avg_improved adjust_i,
                        adj_unit_price_psf * adjust_i as {transaction_price_psf},
                        xgb_avg_pred_psf as {avm_price_psf}
                    from base_transaction b
                    join data_science.ui_master_sg_project_geo_view_filled_features_condo d
                        using (dw_project_id)
                    join developer_tool.sg_condo_properties_estimate_launch_price
                        using (dw_property_id)
                    join base_first_launch_round c
                        on b.dw_project_id = c.dw_project_id
                        and b.transaction_date < c.next_launch_month
                    left join base_index adjust_index
                        on project_launch_month = adjust_index.transaction_month_index
                    where project_launch_month >= {self.min_year}00
                    order by 1, 2, 3
                )
                select 
                    a.project_name,
                    a.dw_project_id,
                    a.unit,
                    a.address_stack,
                    a.days_on_market,
                    {transaction_price_psf},
                    {avm_price_psf},
                    {','.join(
                self.features.hedonic_features +
                self.features.geo_features
            )}
                    from base_transaction_price a
                    left join data_science.ui_master_sg_project_geo_view_filled_features_condo
                        using (dw_project_id)
                    left join data_science.ui_master_sg_building_view_filled_features_condo b
                        using (dw_building_id)
                    left join data_science.ui_master_sg_properties_view_filled_static_features_condo c
                        using (dw_property_id)    
                    order by 1, 2, 3
                """
        )

        if 'stack_index' in self.features.price_features:
            index_table = 'sg_condo_stack_index_resale' \
                if index_mode == 'global' else 'sg_condo_local_stack_index_resale'

            stack_index_table = query_data(
                f"""
                select 
                    dw_project_id, 
                    address_stack,
                    coef as stack_index
                from developer_tool.{index_table}
                where dw_project_id in ({','.join([i.__repr__() for i in raw_data['dw_project_id']])})
                """
            )

            raw_data = raw_data.merge(stack_index_table, on=['dw_project_id', 'address_stack'], how='left')

        raw_data[self.ranker_target] = \
            raw_data.groupby(self.groupby_keys)['days_on_market'].rank(**self.rank_args)
        raw_data[relative_price_difference] = \
            raw_data[transaction_price_psf] / raw_data[avm_price_psf] - 1

        return raw_data

    def fit_property_level_ranker(self, train):

        self.ranker_layer = CBM(
            control_features=self.features.all_features,
            target=self.ranker_target,
            core_model=catboost.CatBoostRanker()
        ).fit(train)

        return self

    def fit(self):

        train = self.get_data_subset(subset='train').copy()
        print_in_green_bg(f'training the ranker...')
        self.fit_property_level_ranker(train)

        return self

    def test(self, include_train=False):

        if include_train:
            test = self.property_data
        else:
            test = self.get_data_subset('test').copy()

        print_in_green_bg(f'predict the ranks...')
        test['pred_score'] = self.ranker_layer.predict(test)

        groupby_data = test.groupby(self.groupby_keys)
        test[self.pred_ranker_target] = groupby_data['pred_score'].rank(**self.rank_args)
        # test[self.pred_ranker_target] = test.groupby(self.groupby_keys)['pred_score'].rank(
        #     **dict(ascending=True, method='dense', pct=True)
        # )

        abs_rel_diff = groupby_data.apply(
            lambda df: df[self.pred_ranker_target] / df[self.ranker_target] - 1
        )
        print_in_green_bg(f'agg abs relative deviation {abs_rel_diff.mean() * 100 :.2f}%')

        return test


input_features = CARFeatures(
    price_features=[
        'transaction_price_psf',
        'avm_price_psf',
        'relative_price_difference',
        'stack_index'
    ],
    hedonic_features=[
        'floor_area_sqft',
        'address_floor_num',
        'top_floor',
        'bottom_2_floor',
        'num_of_bedrooms'
    ],
    geo_features=[
        'proj_num_of_units',
        'num_of_nearby_completed_condo_proj_200m',
        'num_of_nearby_completed_condo_proj_400m',
        'num_of_nearby_completed_condo_proj_600m',
        'num_of_nearby_completed_condo_proj_800m',
        'num_of_nearby_completed_condo_proj_1000m',
        'num_of_nearby_launched_condo_proj_200m',
        'num_of_nearby_launched_condo_proj_400m',
        'num_of_nearby_launched_condo_proj_600m',
        'num_of_nearby_launched_condo_proj_800m',
        'num_of_nearby_launched_condo_proj_1000m',
        'num_of_nearby_completed_condo_units_1000m',
        'num_of_nearby_launched_condo_units_1000m',
        'num_of_nearby_completed_hdb_units_1000m',
        'project_units_zero_rm',
        'project_units_one_rm',
        'project_units_two_rm',
        'project_units_three_rm',
        'project_units_four_rm',
        'project_units_five_rm',
        'km_to_sg_cbd',
        'num_of_bus_stops',
        'num_of_good_schools',
        'latitude',
        'longitude'
    ],
)

model = CatAVMRanker(
    features=input_features
).fit()

samples = model.test()
filtered_samples = samples[
    (samples['proj_num_of_units'] > 200)
]

sorts = samples.groupby('project_name').apply(
    lambda df: (df['pred_days_on_market_rank'] / df['days_on_market_rank'] - 1).abs().mean()
).sort_values()


def plot_tower_view(
        project_trans,
        values,
        ax=None,
        fig=None,

):
    project_trans_sorted = project_trans.sort_values(by='address_stack').fillna(-1)

    pivot_data = project_trans_sorted.pivot_table(
        index=project_trans_sorted['address_floor_num'].astype(int),
        columns=project_trans_sorted['address_stack'],
        values=values,
        aggfunc='mean'
    ).iloc[::-1]
    pivot_data[pivot_data == -1] = np.nan

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    cmap = sns.color_palette("coolwarm", as_cmap=True).copy()
    cmap.set_bad(color="lightgrey")
    sns.heatmap(
        pivot_data, ax=ax,
        cmap=cmap, annot=True, fmt=".0f", annot_kws={"size": 8},
        linewidth=0.5, linecolor='lightgrey'
    )
    title = f'{values.replace("_", " ")} heatmap'
    ax.set_title(title)
    plt.ylabel("Address Floor Number")
    plt.xlabel("Bedroom-Stack")

    return fig, ax


def compare_tower_views(
        project_trans,
        num_of_bedrooms
):
    project_trans = project_trans[project_trans['num_of_bedrooms'] == num_of_bedrooms].copy()

    rank_cols = [model.ranker_target, model.pred_ranker_target]

    for rank_col in rank_cols:
        project_trans[rank_col] = project_trans[rank_col] * 100

    project_trans['diff'] = project_trans[model.pred_ranker_target] - project_trans[model.ranker_target]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for idx, v in enumerate(rank_cols + ['diff']):
        ax = axs[idx]

        fig, ax = plot_tower_view(
            project_trans,
            values=v,
            ax=ax,
            fig=fig,
        )

    project_name = project_trans['project_name'].iloc[0]

    _, pool = model.ranker_layer.construct_pool_data(project_trans)
    c_score = model.ranker_layer.core_model.score(pool)

    plt.suptitle(f'{project_name} {num_of_bedrooms}-bedroom {c_score * 100: .2f}%')
    plt.savefig(figures_dir + f'ranks tower view {project_name} {num_of_bedrooms}-bedroom.png', dpi=300)


target_projects = ['royalgreen', 'rivertrees-residences', 'waterbay', 'terra-hill', 'one-draycott']
# tower view
bed_num = 2
for p in target_projects:

    filtered_trans = samples[
        (samples['project_name'] == p) & (samples['num_of_bedrooms'] == bed_num)
    ]

    if filtered_trans.empty:
        continue

    compare_tower_views(
        filtered_trans,
        num_of_bedrooms=bed_num
    )

from scatterplot_view import *

for p in target_projects:
    filtered_trans = samples[
        (samples['project_name'] == p)
    ].copy()

    _, pool = model.ranker_layer.construct_pool_data(filtered_trans)
    c_score = model.ranker_layer.core_model.score(pool)

    fig, ax = scatter_plot_view(
        x=filtered_trans[model.ranker_target],
        y=filtered_trans[model.pred_ranker_target],
        label_series=filtered_trans['num_of_bedrooms'],
        title=f'scatter_plot_{p}_{c_score * 100 :.2f}%'
    )

    plt.savefig(figures_dir + f'scatter_plot_{p}.png', dpi=300)

# The following is a replicate of CatBoostRanker.score()
if False:
    # CatBoostRanker.score(pool_data) = CatBoostRanker.score(pool_data, type='Base', denominator='LogPosition')
    filtered_trans = samples[samples['project_name'] == 'midtown-residences']
    filtered_trans = filtered_trans[filtered_trans['num_of_bedrooms'] == 2].copy()

    _, pool = model.ranker_layer.construct_pool_data(filtered_trans)
    c_score = model.ranker_layer.core_model.score(pool)
    print_in_green_bg(f'CatBoost Score: {c_score: g}.')

    # The objects in each group are sorted in descending order of target relevancies
    # The lager the rank number, the less relevant

    y = pool.get_label()
    y_hat = model.ranker_layer.core_model.predict(pool)

    eval_metric(
        y,
        y_hat,
        metric='NDCG',
        group_id=[1, ] * len(y)
    )


    def ndcg(
            actual_values: pd.Series,
            predicted_values: pd.Series
    ):
        base_log_dcg = lambda a, wa: (a / np.log2(wa + 1)).sum()
        res = (
                base_log_dcg(actual_values, predicted_values.rank(method='first', ascending=False)) /
                base_log_dcg(actual_values, actual_values.rank(method='first', ascending=False))
        )

        return res


    r_score = ndcg(pd.Series(y), pd.Series(model.ranker_layer.core_model.predict(pool)))
    print_in_green_bg(f'Replicated Score: {c_score: g}.')
    print()

# The following is an example of usage with a ranking metric:
if False:
    # The dataset consists of five objects. The first two belong to one group
    # reference:
    #   https://catboost.ai/en/docs/concepts/python-reference_utils_eval_metric
    #   https://catboost.ai/en/docs/references/ndcg#calculation

    # and the other three to another.
    group_ids = [1, 1, 2, 2, 2]

    labels = [0.9, 0.1, 0.5, 0.4, 0.8]

    # In ranking tasks it is not necessary to predict the same labels.
    # It is important to predict the right order of objects.
    good_predictions = [0.5, 0.4, 0.2, 0.1, 0.3]
    bad_predictions = [0.4, 0.5, 0.2, 0.3, 0.1]

    good_ndcg = eval_metric(labels, good_predictions, 'NDCG', group_id=group_ids)
    bad_ndcg = eval_metric(labels, bad_predictions, 'NDCG', group_id=group_ids)

    for pred in [good_predictions, bad_predictions]:

        print(f'\nprediction: {pred}')

        df = pd.DataFrame(
            {
                'group': group_ids,
                'rank': labels,
                'prediction': pred
            }
        )

        builtin_score = df.groupby('group').apply(
            lambda a: eval_metric(
                a['rank'].values,
                a['prediction'].values,
                'NDCG',
                group_id=a['group'].values
            )
        )


        def ndcg(
                actual_values: pd.Series,
                predicted_values: pd.Series
        ):
            base_log_dcg = lambda a, wa: (a / np.log2(wa + 1)).sum()
            res = (
                    base_log_dcg(actual_values, predicted_values.rank(ascending=False)) /
                    base_log_dcg(actual_values, actual_values.rank(ascending=False))
            )

            return res


        replicated_score = df.groupby('group').apply(
            lambda a: ndcg(
                a['rank'],
                a['prediction'],
            )
        )

        for name, results in zip(
                ['Catboost Built-in Score', 'Replicated Score'],
                [builtin_score, replicated_score],
        ):
            print_in_green_bg(f'{name} in each group:')
            print(results)
            print_in_green_bg(f'{name} after aggregation:')

            if name == 'Catboost Built-in Score':
                print(
                    eval_metric(
                        df['rank'].values,
                        df['prediction'].values,
                        'NDCG',
                        group_id=df['group'].values
                    )
                )
            else:
                print(results.mean())
