from dataclasses import dataclass
from typing import Mapping, Union, Optional, List

import numpy as np
import pandas as pd

from constants.utils import print_in_green_bg
from demand_curve_sep.cls_linear_demand_model import BaseLinearDemandModel
from demand_curve_sep.scr_common_training import comparable_demand_model
from optimization.cls_base_simulation_results import (
    UnitSalesPath, ProjectSalesPaths, ConfigRevenue
)
from optimization.project_config import ProjectConfig
from src.main.land_config import LandConstraints, ConfigGenerationError
from src.utils.simulated_annealing import EvaluationPredicate, annealing


def customize_optimization(
    initial_state,
    state_generator,
    revenue_calculator,
    detailed_output=False
):

    """
        Customize and perform simulated annealing optimization to find an optimal state.

        Args:
            initial_state (object): The initial state from which optimization begins.
            state_generator (callable): A function that generates neighboring states.
            revenue_calculator (callable): A function that calculates the revenue of a given state.
            detailed_output (bool, optional): If True, return detailed optimization results.

        Returns:
            object: The optimal state that maximizes revenue (or minimizes the negative revenue ratio).

        If detailed_output is True:
        Returns:
            Tuple[object, List[object], List[Tuple[float, float]]]: A tuple containing the following:
            - optimal_state (object): The optimal state that maximizes revenue.
            - evaluated_states (List[object]): A list of evaluated states during optimization.
            - transitions (List[Tuple[float, float]]): A list of transition probabilities and acceptance probabilities.

        The optimization process uses a loss function based on negative revenue ratio, where higher values indicate better states.
        Simulated annealing parameters, such as temperature and iteration limits, are defined internally.
        """

    evaluation_predicate = EvaluationPredicate(
        max_calls=200,
        max_iter=2000,
        window_length=10,
    )

    initial_total_revenue = revenue_calculator(initial_state)

    loss = lambda path: -revenue_calculator(path) / initial_total_revenue

    optimal_state, evaluated_states, transitions = annealing(
        loss,
        initial_state=initial_state,
        state_generator=state_generator,
        temperature=lambda n: 1 / (1 + 10 * n),
        evaluation_predicate=evaluation_predicate
    )

    if detailed_output:
        return optimal_state, evaluated_states, transitions
    return optimal_state


@dataclass
class BaseBestPathModel:

    """
    A wrapper for `RoomTypeDemandModel` to calculate the total revenue of a selling path (given new config if applicable).

    Args:
    num_of_bedrooms (int).
    demand_model (BaseLinearDemandModel): An instance of a demand model for the project.
    initial_config (ProjectConfig): The initial configuration of the project.

    Methods:
    process_config(cfg: ProjectConfig) -> pd.DataFrame:
        Process a project configuration and return relevant data in a DataFrame format.

    get_update_coef(new_cfg: ProjectConfig, time_index_to_multiply: float = 1) -> float:
        Calculate the coefficient for updating the demand model based on a new configuration.
        - time_index_to_multiply (float, optional): A coefficient used to adjust the results from the current date
      back to a previous transaction date (e.g., launch date).

    calculate_total_revenue(
        cfg, psf_path, time_index_to_multiply=1, full_output=False, discount_rate=0.025
    ) -> Union[UnitSalesPath, float]:
        Calculate the total revenue for a given configuration and price path.
        Can be wrapped as a revenue calculator to be put in `customize_optimization` function

    get_projects_best_path(
        cfg, path_length=8, price_range=(1600, 1900), time_index_to_multiply=1,
        max_growth_psf=None, max_growth_rate=0.02, discount_rate=0.025
    ) -> Union[UnitSalesPath, float]:
        Find the best selling path for a project configuration within specified constraints.

    """

    num_of_bedrooms: int
    demand_model: BaseLinearDemandModel
    initial_config: ProjectConfig

    def process_config(self, cfg: ProjectConfig) -> pd.DataFrame:
        launch_year_month = pd.to_datetime(f'{cfg.launching_year}-{cfg.launching_month:02d}-01')
        config_data = pd.DataFrame(
            {
                'project_name': [cfg.project_name],
                'num_of_bedrooms': self.num_of_bedrooms,
                'launch_year_month': launch_year_month,
                'transaction_month': launch_year_month,
                'launching_period': np.nan,
                'sales': np.nan,
                'price': np.nan,
                'num_of_units': cfg.get_units_count(self.num_of_bedrooms),
                'num_of_remaining_units': np.nan,
                'proj_num_of_units': sum(cfg.total_unit_count),
                'tenure': 1 if cfg.tenure == 'freehold' else 0,
                'floor_area_sqm': cfg.avg_unit_size_per_bed(self.num_of_bedrooms),
                'proj_max_floor': cfg.max_floor,
            }
        )

        return config_data

    def get_update_coef(
        self,
        new_cfg:
        ProjectConfig,
        time_index_to_multiply: float = 1
    ) -> float:
        old_coef = comparable_demand_model.query_adjust_coef(
            self.process_config(
                self.initial_config
            )
        )
        new_coef = comparable_demand_model.query_adjust_coef(
            self.process_config(
                new_cfg
            )
        )

        return 1 / old_coef * new_coef * time_index_to_multiply

    def calculate_total_revenue(
        self,
        cfg,
        psf_path,
        time_index_to_multiply=1,
        full_output=False,
        discount_rate=0.025
    ) -> Union[UnitSalesPath, float]:

        data_row = self.process_config(cfg)

        # if cfg == self.initial_config:
        #     coef_to_multiply = time_index_to_multiply
        # else:
        #     coef_to_multiply = self.get_update_coef(
        #         new_cfg=cfg,
        #         time_index_to_multiply=time_index_to_multiply
        #     )

        valid_psf_path = np.array([])
        valid_quantity_path = np.array([])
        floor_area_sqft = data_row['floor_area_sqm'].iloc[0] * 10.76
        stock = data_row['num_of_units'].iloc[0]

        for idx, p in enumerate(psf_path):
            # todo: manual alert
            t = min(1 + idx * 3, 4)
            remaining_units = int(stock - valid_quantity_path.sum())

            coef_to_multiply = self.get_update_coef(
                new_cfg=cfg,
                time_index_to_multiply=time_index_to_multiply if t == 1 else 1
            )

            data = data_row.copy()
            data['price'] = p / coef_to_multiply
            data['launching_period'] = t
            data['num_of_remaining_units'] = remaining_units
            q = int(self.demand_model.predict(data).iloc[0])

            valid_psf_path = np.append(valid_psf_path, p)
            valid_quantity_path = np.append(valid_quantity_path, q)

            if q == remaining_units:
                break

        valid_period_path = np.arange(1, len(valid_quantity_path) + 1)

        revenue = valid_quantity_path * valid_psf_path * floor_area_sqft
        discounted_revenue = revenue / (1 + discount_rate) ** valid_period_path

        results = UnitSalesPath(
            **{
                'bed_num': self.num_of_bedrooms,
                'quantity_path': valid_quantity_path.astype(int),
                'psf_path': valid_psf_path,
                'price_path': valid_psf_path * floor_area_sqft,
                'revenue_path': revenue,
                'discounted_revenue_path': discounted_revenue,
                'total_revenue': np.nansum(revenue),
                'discounted_total_revenue': np.nansum(discounted_revenue)
            }
        )

        if full_output:
            return results

        return results.discounted_total_revenue

    def get_projects_best_path(
        self,
        cfg,
        path_length=8,
        price_range=(1600, 1900),
        time_index_to_multiply=1,
        max_growth_psf=None,
        max_growth_rate=0.02,
        discount_rate=0.025
    ) -> Union[UnitSalesPath, float]:

        if cfg is None:
            cfg = self.initial_config

        def path_generator(current_path, temperature):
            lower_bound = np.random.uniform(*price_range, size=1)

            if max_growth_psf:
                upper_bound = lower_bound + max_growth_psf
            else:
                upper_bound = min(
                    lower_bound * (1 + max_growth_rate) ** path_length,
                    max(*price_range)
                )

            return tuple(
                np.sort(
                    np.random.uniform(lower_bound, upper_bound, size=path_length)
                )
            )

        def revenue_calculator(psf_path, full_output=False):

            return self.calculate_total_revenue(
                cfg=cfg,
                psf_path=psf_path,
                time_index_to_multiply=time_index_to_multiply,
                full_output=full_output,
                discount_rate=discount_rate
            )

        suggestion_path = customize_optimization(
            initial_state=(price_range[0],) * path_length,
            state_generator=path_generator,
            revenue_calculator=revenue_calculator
        )

        res = revenue_calculator(suggestion_path, full_output=True)

        if sum(res.quantity_path) != cfg.get_units_count(self.num_of_bedrooms):
            print_in_green_bg(f'{self.num_of_bedrooms}-bed: fail to sell out.')

        return res

    def get_projects_current_path(
        self,
        price_psf,
        time_index_to_multiply=1,
        discount_rate=0.025
    ) -> Union[UnitSalesPath, float]:

        res = self.calculate_total_revenue(
            cfg=self.initial_config,
            psf_path=(price_psf,) * 20,
            time_index_to_multiply=time_index_to_multiply,
            full_output=True,
            discount_rate=discount_rate
        )

        if sum(res.quantity_path) != self.initial_config.get_units_count(self.num_of_bedrooms):
            print_in_green_bg(f'{self.num_of_bedrooms}-bed: fail to sell out.')

        return res


@dataclass
class BestPathsModels:

    """
    A dataclass representing models for generating best selling paths for different bedroom types.

    Attributes:
    - demand_models (Mapping[int, BaseLinearDemandModel]): A mapping of bedroom types to demand models.
    - initial_config (ProjectConfig): The initial project configuration.
    - new_config (Optional[ProjectConfig]): The new project configuration, if different from the initial config.

    """

    demand_models: Mapping[int, BaseLinearDemandModel]
    initial_config: ProjectConfig
    new_config: Optional[ProjectConfig] = None

    def __post_init__(self):

        """
        Initialize the BestPathsModels instance.

        If a new configuration is not provided, it defaults to the initial configuration.
        Transforms demand models for different bedroom types into BestPathModel instances.
        """

        if self.new_config is None:
            self.__setattr__('new_config', self.initial_config)

        self.transformed_models = {
            i: BaseBestPathModel(
                num_of_bedrooms=i,
                demand_model=self.demand_models[i],
                initial_config=self.initial_config
            )
            for i in self.demand_models.keys()
        }

    def get_best_selling_paths(
        self,
        price_ranges,
        path_lengths: Union[dict, int],
        time_index_to_multiply=1,
        max_growth_psf=None,
        max_growth_rate=0.02,
        discount_rate=0.025
    ) -> ProjectSalesPaths:

        """
            Find the best selling paths for different bedroom types.

            Parameters:
            - price_ranges: A mapping of bedroom types to price ranges.
            - path_lengths: The length of the selling paths for each bedroom type, specified as either an integer or a dictionary.
            - time_index_to_multiply (float): A coefficient used to adjust the results based on time.
            - max_growth_psf: The maximum allowed growth in price per square foot.
            - max_growth_rate: The maximum allowed growth rate for prices.
            - discount_rate: The discount rate used for revenue calculations.

            Returns:
            - ProjectSalesPaths: The best selling paths for different bedroom types as ProjectSalesPaths.

        """

        cfg = self.new_config
        suggestion_paths = {
            bed_num: bed_model.get_projects_best_path(
                cfg,
                price_range=price_ranges[bed_num],
                path_length=path_lengths[bed_num] if isinstance(path_lengths, dict) else path_lengths,
                time_index_to_multiply=time_index_to_multiply,
                max_growth_psf=max_growth_psf,
                max_growth_rate=max_growth_rate,
                discount_rate=discount_rate
            )
            for bed_num, bed_model in self.transformed_models.items()
            if bed_num in cfg.available_bed
        }

        return ProjectSalesPaths(suggestion_paths)

    def get_current_selling_paths(
        self,
        price_dict,
        time_index_to_multiply=1,
        discount_rate=0.025
    ):
        suggestion_paths = {
            bed_num: bed_model.get_projects_current_path(
                price_psf=price_dict[bed_num],
                time_index_to_multiply=time_index_to_multiply,
                discount_rate=discount_rate
            )
            for bed_num, bed_model in self.transformed_models.items()
            if bed_num in self.initial_config.available_bed
        }

        return ProjectSalesPaths(suggestion_paths)


@dataclass
class BestLandModel:
    """
        A dataclass representing a model for optimizing land use decisions for a real estate project.

        Attributes:
        - land_constraints (LandConstraints): Constraints related to land use.
        - demand_models (Mapping[int, BaseLinearDemandModel]): Demand models for different bedroom types.
        - initial_config (ProjectConfig): The initial project configuration.
    """

    land_constraints: LandConstraints
    demand_models: Mapping[int, BaseLinearDemandModel]
    initial_config: ProjectConfig

    def __post_init__(self):
        self.transformed_models = {
            i: BaseBestPathModel(
                num_of_bedrooms=i,
                demand_model=self.demand_models[i],
                initial_config=self.initial_config
            )
            for i in self.demand_models.keys()
        }

    def get_random_project_config(self, project_config_params) -> ProjectConfig:

        n_attempts = 10

        for n in range(n_attempts):

            try:

                bc = self.land_constraints.gen_random_config(
                    project_config_params
                ).aggregated(self.initial_config.static_config)

                pc_dict = {}
                for k, v in bc.__dict__.items():

                    if k == 'postal_code':
                        pc_dict['project_name'] = v
                    else:
                        if isinstance(v, tuple):
                            v = tuple(v)

                        pc_dict[k] = v

                return ProjectConfig(**pc_dict)

            except ConfigGenerationError:

                continue

        print(f'Unable to generate qualified config after {n_attempts} attempts.')

    def get_promising_configs(
        self,
        project_config_params,
        price_ranges,
        max_periods: Union[dict, int],
        time_index_to_multiply,
        output_num: int
    ) -> List[ConfigRevenue]:

        """
        Generates and evaluates promising project configurations to identify top candidates for land use optimization.

        Args:
        - project_config_params: Parameters for generating project configurations.
        - price_ranges: Price ranges for different bedroom types.
        - max_periods: Maximum periods for selling paths (can be a dictionary or integer).
        - time_index_to_multiply: A coefficient for adjusting results from the current date to previous dates.
        - output_num: Number of promising configurations to generate and evaluate.

        This method performs the following steps:
        1. Generates random project configurations based on the given parameters.
        2. Estimates the revenue using the lower bound of the price.
        3. Selects the top 'output_num' project configurations based on estimated revenues.

        Returns a list of ConfigRevenue objects representing the promising configurations and their associated revenues.
        """

        def state_generator(current_state, temperature):

            random_cfg = self.get_random_project_config(
                project_config_params
            )

            waste = 1 - random_cfg.gross_floor_area / self.initial_config.gross_floor_area

            print(f'{waste * 100: .2f}% of gfa is wasted.')

            return random_cfg

        def revenue_calculator(cfg, detailed_output=False):
            selling_paths = ProjectSalesPaths(
                {
                    b: self.transformed_models[b].calculate_total_revenue(
                        cfg=cfg,
                        psf_path=(price_ranges[b][0],) * (
                            max_periods[b]
                            if isinstance(max_periods, dict) else max_periods
                        ),
                        full_output=True,
                        time_index_to_multiply=time_index_to_multiply
                    )
                    for b in np.arange(1, 6) if cfg.get_units_count(b) != 0
                }
            )

            discounted_tr = selling_paths.discounted_revenue

            if detailed_output:
                return ConfigRevenue(cfg=cfg, paths=selling_paths)

            return discounted_tr

        optimal_config, transitions, _ = customize_optimization(
            initial_state=self.initial_config,
            state_generator=state_generator,
            revenue_calculator=revenue_calculator,
            detailed_output=True
        )

        all_transitions = list(transitions.keys())
        sorted_transitions = np.array(
            [transitions[j] for j in all_transitions]
        ).argsort()

        res = []
        for i in sorted_transitions[:output_num]:
            project_config = all_transitions[i]
            res += [revenue_calculator(project_config, detailed_output=True)]

        return res

    def get_best_land_use(
        self,
        project_config_params,
        price_ranges,
        max_periods,
        time_index_to_multiply=1,
        output_num=5
    ):
        """
        Optimizes land use decisions by generating and evaluating promising project configurations.

        Args:
        - project_config_params: Parameters for generating project configurations.
        - price_ranges: Price ranges for different bedroom types.
        - max_periods: Maximum periods for selling paths.
        - time_index_to_multiply: A coefficient for adjusting results from the current date to previous dates.
        - output_num: Number of promising configurations to generate and evaluate.

        This method performs the following steps:
        1. Generates a set of promising project configurations based on given parameters.
        2. Evaluates and selects the best-selling paths for the top 'output_num' promising configurations.
        3. Automatically saves proposed configurations and their associated best-selling paths.
            The results are stored in CSV files for further analysis.
        """

        from demand_curve_live.scr_get_paths import table_dir

        promising_configs = self.get_promising_configs(
            project_config_params=project_config_params,
            price_ranges=price_ranges,
            max_periods=max_periods,
            time_index_to_multiply=time_index_to_multiply,
            output_num=output_num
        )

        output_df = pd.DataFrame()
        for c in promising_configs:

            paths_model = BestPathsModels(
                demand_models=self.demand_models,
                initial_config=self.initial_config,
                new_config=c.cfg
            )

            suggestion_paths = paths_model.get_best_selling_paths(
                price_ranges=price_ranges,
                max_growth_psf=150,
                path_lengths=max_periods,
                time_index_to_multiply=time_index_to_multiply
            )

            paths_record = suggestion_paths.detailed_dataframe()
            paths_record['num_of_units'] = paths_record['bed_num'].apply(
                lambda b: c.cfg.get_units_count(bed=b)
            )
            paths_record['area_sqm'] = paths_record['bed_num'].apply(
                lambda b: c.cfg.avg_unit_size_per_bed(bed=b)
            )

            project_tr_mill = suggestion_paths.revenue / 10 ** 6
            file_name = f"best selling path of best config {c.cfg.project_name} {project_tr_mill: .2f}m.csv"
            paths_record.to_csv(table_dir + file_name.replace(' ', '_'))

            res = ConfigRevenue(cfg=c.cfg, paths=suggestion_paths)
            res.summary()
            temp = res.to_dataframe()
            temp['total_revenue'] = res.revenue

            output_df = pd.concat([output_df, temp], ignore_index=False)

        from demand_curve_live.scr_get_paths import table_dir
        tr_mill = output_df['total_revenue'].max() / 10 ** 6
        output_df.to_csv(
            table_dir +
            f'best_land_use_{self.initial_config.project_name.replace(" ", "_")}_{tr_mill: .2f}m.csv'
        )
