from dataclasses import dataclass
from typing import Union, Mapping, Optional

import numpy as np
import pandas as pd

from constants.redshift import query_data
from demand_curve_hybrid.scr_coef import query_adjust_coef
from demand_curve_live.le_quest.scr_get_model import linear_models as models, training_data_class
from demand_curve_sep.cls_linear_demand_model import BaseLinearDemandModel
from optimization.cls_base_simulation_results import UnitSalesPath, ProjectSalesPaths, PathGeneratorParams
from optimization.cls_revenue_optimization import BestPathInterface

from optimization.project_config import ProjectConfig

time_index_table = query_data(training_data_class.index_table_query)[
    [
        'transaction_month_index',
        'time_adjust_coef'
    ]
]
transaction_month = pd.to_datetime(
    time_index_table['transaction_month_index'].apply(lambda a: f'{a[:4]}-{a[-2:]}-01')
)
time_index_series = time_index_table.set_index(transaction_month)['time_adjust_coef']


def fill_nominal_data(num_of_bedrooms, size_rebased: bool = True):

    query_params = dict(
        project_name='Le Quest',
        mode='rolling',
        num_of_bedrooms=num_of_bedrooms
    )

    if size_rebased:
        bed_data = training_data_class.get_rebased_project_data(
            **query_params
        )
    else:
        bed_data = training_data_class.get_nominal_project_data(
            **query_params
        )

    t_range = pd.date_range(
        start=bed_data['transaction_month'].min(),
        end=bed_data['transaction_month'].max(),
        freq='3MS'
    )

    filled_t_range = pd.DataFrame(
        data=t_range,
        columns=['transaction_month']
    )

    filled_bed_data = filled_t_range.merge(
        bed_data,
        on='transaction_month',
        how='left'
    )
    filled_bed_data['transaction_month_idx'] = filled_bed_data['transaction_month'].apply(
        lambda t: training_data_class.month_index.loc[t]
    )

    filled_bed_data['time_adjust_coef'] = filled_bed_data.apply(
        lambda row: time_index_series.loc[row['transaction_month']],
        axis=1
    )
    filled_bed_data['price'].interpolate(method='linear', inplace=True)
    filled_bed_data['sales'].fillna(0, inplace=True)
    filled_bed_data['num_of_units_launched'].fillna(0, inplace=True)
    filled_bed_data['num_of_remaining_units'].fillna(method='ffill', inplace=True)

    for c in [
        'dw_project_id', 'project_name', 'num_of_bedrooms',
        'launch_year_month', 'num_of_units', 'proj_num_of_units', 'tenure',
        'floor_area_sqm', 'proj_max_floor', 'zone',
        'neighborhood', 'distance'
    ]:
        filled_bed_data[c].fillna(method='bfill', inplace=True)
        filled_bed_data[c].fillna(method='ffill', inplace=True)

    filled_bed_data = training_data_class.calculate_launching_period(filled_bed_data)

    filled_bed_data['is_first_period'] = filled_bed_data['launching_period'].apply(lambda a: 1 if a <= 3 else 0)
    filled_bed_data['is_minor_first_period'] = filled_bed_data['minor_launching_period'].apply(
        lambda a: 1 if a <= 1 else 0
    )

    return filled_bed_data[filled_bed_data['launching_period'].isin(np.arange(1, 60, 3))].copy()


def calculate_path_revenues(num_of_bedrooms, discount_rate=0.025):

    bed_data = fill_nominal_data(num_of_bedrooms, size_rebased=False)

    valid_quantity_path = np.array([])

    for idx, row in bed_data.iterrows():

        stock = row['num_of_units']
        remaining_units = int(stock - valid_quantity_path.sum())
        bed_data.loc[idx, 'num_of_remaining_units'] = remaining_units

        if remaining_units > 0:

            raw_q = models[num_of_bedrooms].predict(bed_data.iloc[[idx]]).iloc[0]

            if np.isnan(raw_q):
                q = 0
            else:
                q = int(round(raw_q))

        else:
            q = 0

        valid_quantity_path = np.append(valid_quantity_path, q)

    P = bed_data['price'].values / bed_data['time_adjust_coef'].values
    A = bed_data['floor_area_sqm'].values * 10.76
    T = bed_data['launching_period'].values

    revenue = valid_quantity_path * P * A
    discounted_revenue = revenue / (1 + discount_rate) ** T

    results = UnitSalesPath(
        **{
            'bed_num': num_of_bedrooms,
            'quantity_path': valid_quantity_path.astype(int),
            'psf_path': P,
            'price_path': P * A,
            'revenue_path': revenue,
            'discounted_revenue_path': discounted_revenue,
            'total_revenue': np.nansum(revenue),
            'discounted_total_revenue': np.nansum(discounted_revenue)
        }
    )

    return results


def calculate_actual_revenue(num_of_bedrooms, discount_rate=0.025):

    bed_data = fill_nominal_data(num_of_bedrooms, size_rebased=False)

    P = bed_data['price'].values
    Q = bed_data['sales'].astype(int).values
    A = bed_data['floor_area_sqm'].values * 10.76
    T = bed_data['launching_period'].values

    price_path = P * A
    revenue_path = price_path * Q
    discounted_revenue_path = revenue_path / (1 + discount_rate) ** T

    actual_revenue = UnitSalesPath(
        **{
            'bed_num': num_of_bedrooms,
            'quantity_path': Q,
            'psf_path': P,
            'price_path': price_path,
            'revenue_path': revenue_path,
            'discounted_revenue_path': discounted_revenue_path,
            'total_revenue': np.nansum(revenue_path),
            'discounted_total_revenue': np.nansum(discounted_revenue_path)
        }
    )

    return actual_revenue


class LocalBestPath(BestPathInterface):

    def prepare_data(self, cfg: ProjectConfig, *, psf_path) -> pd.DataFrame:
        launch_year_month = pd.to_datetime(f'{cfg.launching_year}-{cfg.launching_month:02d}-01')

        length = len(psf_path)

        t_range = pd.date_range(
            start=launch_year_month,
            freq='3MS',
            periods=length
        )

        launching_period = np.arange(1, length * 3, 3)

        filled_bed_data = pd.DataFrame(
            {
                'project_name': cfg.project_name,
                'num_of_bedrooms': self.num_of_bedrooms,
                'launch_year_month': launch_year_month,
                'transaction_month': t_range,
                'launching_period': launching_period,
                'sales': np.nan,
                'price': psf_path,
                'num_of_units': cfg.get_units_count(self.num_of_bedrooms),
                'num_of_remaining_units': np.nan,
                'proj_num_of_units': sum(cfg.total_unit_count),
                'tenure': 1 if cfg.tenure == 'freehold' else 0,
                'floor_area_sqm': cfg.avg_unit_size_per_bed(self.num_of_bedrooms),
                'proj_max_floor': cfg.max_floor,
                'num_of_units_launched': [300, 0, 0, 115, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0][:length]
            }
        )

        filled_bed_data = training_data_class.calculate_launching_period(filled_bed_data)

        filled_bed_data['transaction_month_idx'] = filled_bed_data['transaction_month'].apply(
            lambda t: training_data_class.month_index.loc[t]
        )

        filled_bed_data['time_adjust_coef'] = filled_bed_data.apply(
            lambda row: time_index_series.loc[row['transaction_month']],
            axis=1
        )

        # no need to adjust size, because all the training data has been adjusted to the project's size before 
        # fitting the model
        filled_bed_data['price'] = filled_bed_data['price'] * filled_bed_data['time_adjust_coef']

        if cfg != self.initial_config:
            coef_to_multiply = self.get_update_coef(new_cfg=cfg)
            filled_bed_data['price'] = filled_bed_data['price'] * coef_to_multiply

        filled_bed_data['num_of_units_launched'].fillna(0, inplace=True)
        filled_bed_data['num_of_remaining_units'].fillna(method='ffill', inplace=True)

        filled_bed_data['is_first_period'] = filled_bed_data['launching_period'].apply(lambda a: 1 if a <= 3 else 0)
        filled_bed_data['is_minor_first_period'] = filled_bed_data['minor_launching_period'].apply(
            lambda a: 1 if a <= 1 else 0
        )

        return filled_bed_data

    def get_update_coef(
        self,
        new_cfg: ProjectConfig
    ) -> float:

        process_config = lambda cfg: pd.DataFrame(
            {
                'floor_area_sqm': [cfg.avg_unit_size_per_bed(self.num_of_bedrooms)],
                'proj_max_floor': cfg.max_floor
            }
        )

        old_coef = query_adjust_coef(
            process_config(self.initial_config)
        )
        new_coef = query_adjust_coef(
            process_config(new_cfg)
        )

        return 1 / old_coef * new_coef

    def calculate_total_revenue(
        self,
        cfg,
        psf_path,
        full_output=False,
        discount_rate=0.025,
        **kwargs
    ) -> Union[float, UnitSalesPath]:

        bed_data = self.prepare_data(cfg=cfg, psf_path=psf_path)

        valid_psf_path = np.array([])
        valid_quantity_path = np.array([])

        for idx, row in bed_data.iterrows():

            stock = row['num_of_units']
            remaining_units = int(stock - valid_quantity_path.sum())
            bed_data.loc[idx, 'num_of_remaining_units'] = remaining_units

            t = row['launching_period']
            p = row['price']

            adj = 1
            if self.num_of_bedrooms == 1:
                if t == 1:
                    adj = 0.972
                if t == 10:
                    adj = 0.94
            if self.num_of_bedrooms == 2:
                if t == 1:
                    adj = 0.92
                if t == 7:
                    adj = 1.3
            if self.num_of_bedrooms == 4:
                if t == 1:
                    adj = 0.925
                if t == 7:
                    adj = 1.025
                if t == 22:
                    adj = 1.065
                if t == 34:
                    adj = 0.95

            bed_data.loc[idx, 'price'] = p * adj

            if remaining_units > 0:

                raw_q = self.demand_model.predict(bed_data.iloc[[idx]]).iloc[0]

                if np.isnan(raw_q):
                    q = 0
                else:
                    q = int(round(raw_q))

            else:
                break

            valid_psf_path = np.append(valid_psf_path, p / row['time_adjust_coef'])
            valid_quantity_path = np.append(valid_quantity_path, q)

        P = valid_psf_path
        A = bed_data['floor_area_sqm'].values[:len(P)] * 10.76
        T = bed_data['launching_period'].values[:len(P)]

        revenue = valid_quantity_path * P * A
        discounted_revenue = revenue / (1 + discount_rate) ** ((T - 1) // 3 + 1)

        results = UnitSalesPath(
            **{
                'bed_num': self.num_of_bedrooms,
                'quantity_path': valid_quantity_path.astype(int),
                'psf_path': P,
                'price_path': P * A,
                'revenue_path': revenue,
                'discounted_revenue_path': discounted_revenue,
                'total_revenue': np.nansum(revenue),
                'discounted_total_revenue': np.nansum(discounted_revenue)
            }
        )

        if full_output:
            return results
        else:
            return results.discounted_total_revenue


@dataclass
class LocalBestPaths:
    demand_models: Mapping[int, BaseLinearDemandModel]
    initial_config: ProjectConfig
    new_config: Optional[ProjectConfig] = None
    transformed_models: Optional[Mapping[int, BestPathInterface]] = None

    def __post_init__(self):

        if self.new_config is None:
            self.__setattr__('new_config', self.initial_config)

        if self.transformed_models is None:
            self.__setattr__(
                'transformed_models',
                {
                    i: LocalBestPath(
                        num_of_bedrooms=i,
                        demand_model=self.demand_models[i],
                        initial_config=self.initial_config
                    )
                    for i in self.demand_models.keys()
                }
            )

    def get_best_selling_paths(
        self,
        path_params: Union[Mapping[int, Mapping[int, PathGeneratorParams]], Mapping[int, PathGeneratorParams]] = None,
        *,
        price_ranges: Union[tuple, dict] = (1600, 1900),
        path_lengths: Union[int, float] = 8,
        max_growth_psf=None,
        max_growth_rate=0.02,
        discount_rate=0.025
    ) -> ProjectSalesPaths:

        cfg = self.new_config
        suggestion_paths = ProjectSalesPaths(
            {
                bed_num: bed_model.get_projects_best_path(
                    cfg,
                    path_params=path_params[bed_num] if isinstance(path_params, dict) else path_params,
                    price_range=price_ranges[bed_num] if isinstance(price_ranges, dict) else price_ranges,
                    path_length=path_lengths[bed_num] if isinstance(path_lengths, dict) else path_lengths,
                    max_growth_psf=max_growth_psf,
                    max_growth_rate=max_growth_rate,
                    discount_rate=discount_rate
                )
                for bed_num, bed_model in self.transformed_models.items()
            }
        )

        return suggestion_paths
