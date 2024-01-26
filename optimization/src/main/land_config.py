import numpy as np

from typing import Tuple, Dict, Union, Optional, Literal, get_args
from dataclasses import dataclass

from itertools import accumulate

import pickle
from base64 import b64encode, b64decode

from optimization.src.main.building_config import TBuildingConfig, FloorConfig, UnstructuredBuildingConfig

from constants.redshift import query_data as load_table
from optimization.src.utils.random_sampling import random, PartialTilingError

from optimization.src.utils.display import grid_view

from optimization.src.utils.structural import flattened
from optimization.src.utils.functional import last, try_expr, raise_expr, If
from optimization.src.utils.wrappers import dict_repr, defaulted, type_checking
from optimization.src.utils.misc import mean, InfiniteTuple
from optimization.src.main.features import ALL_BEDS


# -----------------------------------------------------------------------------------------------------------------------
## location identifier types

class TPostalCode(str):
    pass


class TLandLot(str):
    pass


LocationType = Union[TPostalCode, TLandLot]


# ----------------------------------------------------------------------------------

# a custom exception for indicating random config generation failure
#
class ConfigGenerationError(Exception):
    pass


# ----------------------------------------------------------------------------------

# a generic class for describing plot of land building configuration
#
class LandConfig(Tuple[TBuildingConfig.dynamic, ...]):

    # (context-independent serialization)
    #
    def serialize(self, format: Literal['bytes', 'str'] = 'bytes') -> Union[bytes, str]:

        res = b64encode(
            pickle.dumps(
                tuple(
                    tuple(
                        tuple(
                            tuple(map(float, u))
                            for u in layer.unit_sizes
                        )
                        for layer in cfg.layers
                    )
                    for cfg in self
                )
            )
        )

        if format == 'str':
            res = res.decode('ascii')

        return res

    @classmethod
    def deserialize(cls, data: Union[bytes, str]):

        if isinstance(data, str):
            # treat strings as sequences of ASCII characters
            data = data.encode('ascii')

        raw_object = pickle.loads(b64decode(data))

        return cls(
            TBuildingConfig.dynamic(
                FloorConfig(
                    InfiniteTuple(tuple)(raw_layer)
                )
                for raw_layer in raw_cfg
            )
            for raw_cfg in raw_object
        )

    # returns a sorted configuration
    # (by building height, with each building configuration with sorted unit sizes)
    #
    def sorted(self):
        return type(self)(
            cfg.sorted()
            for cfg in sorted(self, key=lambda c: c.max_floor())
        )

    # aggregate by unit type
    #
    def aggregate_by_type(
        self,
        f=sum,
        config_aggregator=lambda cfg: cfg.units_per_bed(),
        predicate=lambda _: True
    ):

        aggregators = list(map(config_aggregator, self))

        values = (
            f(
                aggregator(bed)
                for aggregator in aggregators
            )
            for bed in ALL_BEDS
        )

        return {
            bed_idx + 1: value
            for bed_idx, value in zip(range(len(ALL_BEDS)), values)
            if predicate(value)
        }

    def units_per_bed(self):

        return self.aggregate_by_type(
            sum,
            lambda cfg: cfg.units_per_bed(),
            predicate=lambda value: value > 0
        )

    # (rounds the values if `n_digits` is provided)
    #
    def avg_unit_sizes(self, n_digits=None):

        return self.aggregate_by_type(
            mean if n_digits is None else lambda x: round(mean(x), n_digits),

            lambda cfg: cfg.avg_unit_size(),
            predicate=lambda value: value > 0
        )

    def height_range(self):
        return tuple(
            f(cfg.max_floor() for cfg in self)
            for f in (min, max)
        )

    def aggregated(self, static_building_cfg: TBuildingConfig.static):

        return UnstructuredBuildingConfig(
            **static_building_cfg,

            total_unit_count=self.units_per_bed(),
            avg_unit_size=self.avg_unit_sizes(),
            max_floor=self.height_range()[-1],
            num_of_stacks=mean(cfg.avg_num_of_stacks() for cfg in self)
        )

    def gross_floor_area(self):
        return sum(cfg.gross_floor_area() for cfg in self)

    def units_total(self):
        return sum(cfg.units_total() for cfg in self)

    def summary(self):

        h = self.height_range()

        h_expr = f'{h[0]}-{h[1]}' if h[0] != h[1] else f'{h[0]}'

        return (
            f'building count:        '
            f'{len(self)}\n'
        ) + f'height range:          {h_expr} storeys\n' + (
            f'total '
            f'number of '
            f'units: '
            f'{self.units_total()}\n') + dict_repr(
            self.units_per_bed(), indent_level=1
        ) + 'average unit sizes:\n' + dict_repr(
            self.avg_unit_sizes(n_digits=1), indent_level=1
        ) + f'gross floor area:      {round(self.gross_floor_area(), 1)} sqm'

    def grid_view(
        self,
        borders=True,
        summary=True,
        page_width=120
    ):

        res = grid_view(
            (cfg.cell_repr(tab='   ') for cfg in self),
            borders=borders,
            page_width=page_width
        )

        if summary:
            res += '\n\n' + self.summary()

        return res


# base class for defining building configuration constraints
# that can be build on a given plot of land
#
# general constraint formulation:
#
# for a given plot of land with constraints `C`:
#
#     let each building on that land be defined by:
#         b[k]: TBuildingConfig.dynamic
#
#     then:
#         b[k].max_floor            | for any `k`   in C.height_range
#         sum(b[k].gross_floor_area | for all `k`)  <= C.gross_floor_area
#         sum(b[k].units_total      | for all `k`)  <= C.max_dwelling_units
#
@type_checking
@defaulted
@dict_repr
@dataclass
class LandConstraints:
    # for `gen_random_config`
    #
    @dict_repr
    @dataclass
    class TConfigParams:

        n_buildings: Union[int, Tuple[int, int]]
        unit_size_ranges: Dict[int, Tuple[float, float]]
        unit_stock_ranges: Optional[Dict[int, Tuple[int, int]]] = None
        max_stacks: int = 10
        min_dwelling_units: Optional[int] = None

    # ----------------------------------------------------------------

    max_height: int  # maximum building height (in storeys)

    max_dwelling_units: int  # maximum number of dwelling units across all the buildings
    # on the plot of land

    gross_floor_area: float  # gross floor area ( = GPR * lot_size)

    # ----------------------------------------------------------------

    @classmethod
    def from_location(cls, location_id: LocationType):

        if not isinstance(location_id, get_args(LocationType)):
            raise Exception(f'Unsupported location id type: {type(location_id)}')

        # ----------------------------------------------------------------------------------

        location_id_desc = {
            TLandLot: "lot key",
            TPostalCode: "postal code",
        }[type(location_id)]

        where_expr = {
            TLandLot: f"land_parcel_name = '{location_id}'",
            TPostalCode: f"postal_code = '{location_id}'",
        }[type(location_id)]

        postal_code_bridge = """
        left join
            (
                select
                    lot_key as land_parcel_name,
                    postal_code
                from
                    feature_sg.de__address__land_info
            )
            using(land_parcel_name)
        """.strip()

        data = load_table(
            f"""
            select distinct
                *,
                lot_size_sqm * gpr_num as gross_floor_area
            from
                developer_tool.ds_land_info
            {postal_code_bridge if isinstance(location_id, TPostalCode) else "--"}
            where
                {where_expr}
            """
        )

        if len(data) == 0:
            raise Exception(f"No data found for the {location_id_desc} '{location_id}'")
        elif len(data) > 1:
            raise Exception(f"Miltiple distinct entries found for the {location_id_desc} '{location_id}'")

        data = data.iloc[0]

        return cls(
            max_height=data['estimated_max_storeys'],
            max_dwelling_units=data['estimated_max_units'],
            gross_floor_area=data['gross_floor_area']
        )

    def validate(self):

        if self.max_height < 1:
            raise Exception('Max building height should be positive')

        if self.max_dwelling_units < 1:
            raise Exception('Max dwelling unit number should be positive')

        if self.gross_floor_area <= 0:
            raise Exception('Gross floor area should be positive')

    # generate a random land building configuration given the constraints
    #
    def gen_random_config(
        self,
        params: TConfigParams,
        *,
        max_trials: int = 10
    ):

        n_buildings = params.n_buildings
        unit_size_ranges = params.unit_size_ranges
        unit_stock_ranges = params.unit_stock_ranges
        max_stacks = params.max_stacks
        min_dwelling_units = params.min_dwelling_units

        # ----------------------------------------------------------------

        dict_merge = lambda q: last(
            res := {},

            [
                res.setdefault(key, []).append(value)
                for key, value in q
            ],

            res
        )

        partition = lambda a, chunk_sizes: last(
            s := list(accumulate(chunk_sizes)),

            [
                a[k1:k2]
                for k1, k2 in zip([0] + s, s)
            ]
        )

        # distribute `n_items` into `n_bins` uniformly (as much as possible)
        #
        distribute = lambda n_items, n_bins: last(
            q := divmod(n_items, n_bins),

            [
                q[0] + (k < q[1])
                for k in range(n_bins)
            ]
        )

        # 'distribute' a range of integers `r` into `n_bins` uniformly
        # i.e. split the range into a list of ranges `q` such that:
        #
        #   len(q) == n_bins
        #   sum(x[k]) is in r | for any x[k] in q[k] | for all k
        #
        # example:
        #   distribute_range((4, 20), 3) == [(2, 7), (1, 7), (1, 6)]
        #
        distribute_range = lambda r, n_bins: list(
            zip(
                *(distribute(a, n_bins) for a in r)
            )
        )

        # ----------------------------------------------------------------------------------

        min_unit_size = None
        min_unit_type = None

        if min_dwelling_units is None:
            max_trials = 1
        else:
            # checking whether a valid configuration is possible in principle

            min_unit_type, min_unit_size = min(
                (
                    (key, value[0])
                    for key, value in unit_size_ranges.items()
                ),
                key=lambda p: p[-1]
            )

            estimated_max_dwelling_units = self.gross_floor_area // min_unit_size

            if estimated_max_dwelling_units < min_dwelling_units:
                raise Exception(
                    f"Unable to generate a land configuration given the `min_dwelling_units` of {min_dwelling_units}"
                    f" (the maximum possible number of units is {estimated_max_dwelling_units})"
                )

        units_total = None
        fixed_n_buildings = 0
        sampled_unit_sizes_per_building = []

        # rejection sampling of unit sizes given the `min_dwelling_units`
        #
        for trial_idx in range(max_trials):

            # fixing the number of buildings
            #
            if isinstance(n_buildings, tuple):
                fixed_n_buildings = random.random_generator.integers(n_buildings, endpoint=True)
            else:
                fixed_n_buildings = n_buildings

            # [assuming mostly uniform buildings]

            unit_types = list(unit_size_ranges.keys())

            if unit_stock_ranges:
                unit_stock_ranges_per_bed = [
                    distribute_range(r, fixed_n_buildings)
                    if r else
                    None
                    for r in (
                        unit_stock_ranges.get(bed, None)
                        for bed in unit_types
                    )
                ]

                unit_stock_ranges_per_building = [
                    [
                        r[k]
                        if r else
                        None
                        for r in unit_stock_ranges_per_bed
                    ]
                    for k in range(fixed_n_buildings)
                ]

            else:
                unit_stock_ranges_per_building = [None] * fixed_n_buildings

            gfas_per_building = distribute(self.gross_floor_area, fixed_n_buildings)

            sampled_unit_sizes_per_building = [
                last(
                    unit_size_range_values := list(unit_size_ranges.values())
                    if trial_idx <= max_trials // 2 else
                    [
                        (r[0], r[0])
                        for r in unit_size_ranges.values()
                    ],

                    q := try_expr(
                        lambda: random.partial_tiling_ex(
                            a_total=gfas_per_building[building_idx],
                            a_ranges=unit_size_range_values,
                            n_ranges=unit_stock_ranges_per_building[building_idx]
                        ),
                        on_fail=lambda e: raise_expr(
                            ConfigGenerationError(
                                "Unable to generate a land configuration given the constraints"
                                " (try adjusting unit stock ranges)\n"
                                f"Exception: {e}"
                            )
                            if isinstance(e, PartialTilingError) else
                            e
                        )
                    ),

                    # sorting, so the smaller units with smaller number of bedrooms are last
                    # (so that the first candidates to drop are last)
                    #
                    sorted(
                        flattened(
                            (
                                (
                                    (unit_types[idx], v)
                                    for v in a
                                )
                                for idx, a in enumerate(q)
                            ),
                            max_depth=1
                        ),

                        reverse=True
                    )
                )
                for building_idx in range(fixed_n_buildings)
            ]

            units_total = sum(map(len, sampled_unit_sizes_per_building))

            if min_dwelling_units is not None and units_total >= min_dwelling_units:
                break

        if min_dwelling_units is not None and units_total < min_dwelling_units:

            # last-ditch attempt

            if isinstance(n_buildings, tuple):
                fixed_n_buildings = max(n_buildings)
            else:
                fixed_n_buildings = n_buildings

            gfas_per_building = distribute(self.gross_floor_area, fixed_n_buildings)

            sampled_unit_sizes_per_building = [
                [
                    (min_unit_type, min_unit_size)
                    for _ in range(round(gfa_per_building / min_unit_size))
                ]
                for gfa_per_building in gfas_per_building
            ]

            units_total = sum(map(len, sampled_unit_sizes_per_building))

            if units_total < min_dwelling_units:
                raise Exception(
                    f"Unable to generate a land configuration given the `min_dwelling_units` of {min_dwelling_units}"
                )

        # ----------------------------------------------------------------
        # dropping the 'excess' units

        n_to_drop = max(0, units_total - self.max_dwelling_units)

        if n_to_drop > 0:
            for idx, n in enumerate(distribute(n_to_drop, fixed_n_buildings)):

                # [`max_dwelling_units` constraint]

                sampled_unit_sizes = sampled_unit_sizes_per_building[idx][:-n]

                # [`max_height` constraint]

                n_units = len(sampled_unit_sizes)

                min_height = max(1, np.ceil(n_units / max_stacks))
                max_height = min(n_units, self.max_height)

                if min_height > max_height:
                    q = max(0, n_units - max_height * max_stacks)
                    sampled_unit_sizes = sampled_unit_sizes[:-q]

                sampled_unit_sizes_per_building[idx] = sampled_unit_sizes

        # ----------------------------------------------------------------

        return LandConfig(
            last(
                n_units := len(sampled_unit_sizes),

                min_height := max(1, int(np.ceil(n_units / max_stacks))),
                max_height := min(n_units, self.max_height),

                If(
                    max_height < min_height,
                    lambda: raise_expr(
                        ConfigGenerationError(
                            "Unable to generate a land configuration given the constraints"
                            " (they may be too restrictive)"
                        )
                    )
                ),

                height := random.random_generator.integers(min_height, max_height, endpoint=True),

                units_per_floor := distribute(n_units, height),

                shuffled_unit_sizes := [
                    sampled_unit_sizes[idx]
                    for idx in random.random_generator.permutation(len(sampled_unit_sizes))
                ],

                raw_floor_configs := sorted(
                    map(
                        sorted,
                        partition(
                            shuffled_unit_sizes,
                            units_per_floor
                        )
                    )
                ),

                TBuildingConfig.dynamic(
                    FloorConfig(dict_merge(cfg))
                    for cfg in raw_floor_configs
                ).sorted('floors')
            )
            for sampled_unit_sizes in sampled_unit_sizes_per_building
            if sampled_unit_sizes
        )

    # generate an initial state for the randomized search
    #
    def gen_initial_config(
        self,
        params: TConfigParams,
        *,
        max_trials: int = 10
    ):

        # ! refactor !
        #
        return self.gen_random_config(
            params,
            max_trials=max_trials
        )
