from dataclasses import dataclass
from typing import Literal

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.special import comb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# todo: ResaleRanker
# todo: expand AVMRanker to 2010-2023
# todo: Multi-variable Ranker
@dataclass
class CoreRanker:
    actual_label = 'actual_label'
    actual_ranking = 'actual_ranking'

    pred_label = 'pred_label'
    pred_ranking = 'pred_ranking'

    actual_psf = 'adj_unit_price_psf'
    actual_quantum_price = 'adjust_quantum_price'
    adj_avm_psf = 'adj_avm_psf'
    raw_avm_psf = 'raw_avm_price'

    relative_diff = 'price_relative_difference'
    absolute_diff = 'price_absolute_difference'

    absolute_ranking = 'implicit_ranking_absolute'
    relative_ranking = 'implicit_ranking_relative'

    relative_ranking_label = 'relative_ranking_label'
    absolute_ranking_label = 'absolute_ranking_label'

    dom_ranking = 'dom_ranking'

    groupby_keys = ['project_name', 'num_of_bedrooms']
    row_keys = groupby_keys + ['unit']

    def __init__(
        self,
        target: Literal['label', 'ranking'],
        metrics: Literal['relative', 'absolute'],
        *,
        benchmark_days_on_mkt: int = 14,
        min_year: int = 2015
    ):
        self.metrics = metrics
        self.target = target
        self.min_year = min_year
        self.benchmark_days_on_mkt = benchmark_days_on_mkt

        self.actual_y = {
            'label': self.actual_label,
            'ranking': self.actual_ranking
        }.get(target, lambda a: a)

        self.pred_y = 'pred_' + target

        self.ranking_col = {
            'relative': self.relative_ranking_label,
            'absolute': self.absolute_ranking_label,
        }.get(metrics, lambda a: a)

        self.price_difference_col = {
            'relative': self.relative_diff,
            'absolute': self.absolute_diff
        }.get(metrics, lambda a: a)

    def evaluate(self, *args, **kwargs):
        ...

    @property
    def query_scripts(self):
        ranking_keys = ['dw_project_id', 'num_of_bedrooms']

        scripts = f"""
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
                    b.num_units_launched,
                    case when ura_next_launch_month is null
                        then current_date
                        else ura_next_launch_month 
                        end as next_launch_month
                from (select distinct dw_project_id, project_launch_date from base_transaction) a
                left join (select * from base_launch_round where ura_launch_round = 1) b
                    on a.dw_project_id = b.project_dwid
            ), {self.sql_ranking_table_name} as (
                select distinct
                    d.project_name,
                    b.dw_project_id,
                    b.dw_property_id,
                    b.dw_building_id,
                    b.unit,
                    b.address_stack,
                    b.num_of_bedrooms,
                    datediff(day, project_launch_date::date ,transaction_date::date) days_on_market,
                    datediff(month, project_launch_date::date ,transaction_date::date) months_on_market,
                    concat(left(project_launch_date, 7), '-01')::date as launch_month,
                    case when days_on_market <= {self.benchmark_days_on_mkt} 
                        then True 
                        else False 
                    end as {self.actual_label},
                    sum(case when {self.actual_label} is true then 1 else 0 end) over (
                        partition by {', '.join(['b.' + i for i in ranking_keys])}
                    ) as quantity,
                    adjust_index.hi_avg_improved adjust_i,
                    adj_unit_price_psf * adjust_i as {self.actual_psf},
                    xgb_avg_pred_psf as {self.raw_avm_psf}
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
                where is_river_valley != 1
                order by 1, 2, 3
            )
            """

        return scripts

    @property
    def sql_ranking_table_name(self):
        return 'base_ranking_features'

    def _get_raw_data(self):
        ...


@dataclass
class RandomSelection:

    def __post_init__(self):
        simulate_q = np.arange(1, 11, 1)
        simulate_s = np.arange(1, 11, 1)

        A = np.array([1])
        P = np.array([0])
        for q in simulate_q:
            for s in simulate_s:
                if q > s:
                    continue

                A = np.append(A, self.calculate_random_possibility(s, q))
                P = np.append(P, q / s)

        x = 'percentage'
        y = 'accuracy'

        reference = pd.DataFrame(
            {
                y: A,
                x: P
            }
        ).sort_values('percentage').drop_duplicates()

        spl = UnivariateSpline(
            reference.sort_values(x)[x],
            reference.sort_values(x)[y],
            k=2
        )

        new_x = np.linspace(0, 1+1/50, 50)
        new_y = spl(new_x)
        new_y = np.clip(new_y, 0, 1)

        self.reference = pd.DataFrame(
            {
                y: new_y,
                x: new_x
            }
        ).sort_values('percentage').drop_duplicates()

    @classmethod
    def calculate_random_possibility(cls, N, k):
        all_comb = comb(N, k)
        positive_true = np.arange(0, k + 1)
        positive_false = k - positive_true
        negative_true = N - k - positive_false

        prob = comb(k, positive_true) * comb(N - k, k - positive_true) / all_comb
        score = (positive_true + negative_true) / N

        return sum(prob * score)

    def refer(self, percentage_value):
        temp = self.reference.copy()
        temp['abs_diff'] = abs(temp['percentage'] - percentage_value)
        closest_row = temp.loc[temp['abs_diff'].idxmin()]
        return closest_row['accuracy']
