from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Union

from itertools import groupby, accumulate
from src.main.features import ALL_BEDS

from src.utils.functional import last, try_chain
from src.utils.structural import flattened

from src.utils.wrappers import (
    dict_repr,
    dict_slicer,
    dict_constructor,
    with_downcast,
    with_join
)

from src.utils.misc import InfiniteTuple, is_generator


# -----------------------------------------------------------------------------------------------------------------------

# `number of bedrooms to index` conversion tabler
# (e.g. "one" -> 0, "two" -> 1, 1 -> 0 etc.)
#
def bed_to_idx():

    # string to index
    #
    res = dict(
        zip(ALL_BEDS, range(len(ALL_BEDS)))
    )

    # number to index
    #
    res.update(
        dict(
            (idx + 1, idx)
            for idx in range(len(ALL_BEDS))
        )
    )

    return res


# ----------------------------------------------------------------------------------

# floor configuration definition
#
@dataclass(frozen=True)
class FloorConfig:

    # 'infinite' tuple of tuples, such that:
    #
    #   `unit_sizes[k]` is a tuple of `k+1`-bedroom unit sizes
    #
    unit_sizes: InfiniteTuple(tuple)

    # ----------------------------------------------------------------

    def __post_init__(self):

        # `unit_sizes` can be initialized with a dictionary (as it is more convenient);
        # therefore, need to check and convert to InfiniteTuple when necessary
        #
        if isinstance(self.unit_sizes, dict):

            bed2idx = bed_to_idx()

            as_tuple = try_chain(tuple, lambda v: (v,))

            # [key and value normalization]
            #
            unit_sizes = {
                bed2idx[bed]: as_tuple(values)
                for bed, values in self.unit_sizes.items()
            }

            object.__setattr__(
                self,
                'unit_sizes',
                InfiniteTuple(tuple)(
                    unit_sizes.get(n, ())
                    for n in range(1 + max(unit_sizes.keys()))
                )
            )

    # return a configuration with `unit_sizes` sorted
    #
    def sorted(self):
        return type(self)(
            unit_sizes=InfiniteTuple(tuple)(
                map(sorted, self.unit_sizes)
            )
        )

    def area(self):
        return sum(map(sum, self.unit_sizes))

    # number of units by unit type
    #
    def units_per_bed(self, agg=InfiniteTuple(0)):
        return agg(
            map(len, self.unit_sizes)
        )

    def units_total(self):
        return self.units_per_bed(sum)

    def cell_repr(self, tab='\t'):

        return [f'unit_sizes:'] + [
            tab + f'{idx + 1}: {tab}' + '[' + ', '.join(f'{round(val, 1):g}' for val in s) + ']'
            for idx, s in enumerate(self.unit_sizes)
        ]

    def disp(self, indent_level=0):

        tab = '\t' * indent_level

        return '\n'.join(
            tab + line
            for line in self.cell_repr()
        )

    def __str__(self):
        return self.disp()


# ----------------------------------------------------------------------------------

# a generic building configuration definition
# with 'static' (given as a constant input)
# and 'dynamic' (adjustable during optimization)
# features separated
#
class TBuildingConfig:

    @dict_slicer
    @dict_repr
    @dataclass(frozen=True)
    class static:

        postal_code: str

        launching_year: Optional[int] = None
        launching_month: Optional[int] = None
        completion_year: Optional[int] = None

        is_top10_developer: Optional[Union[int, bool]] = 1
        tenure: Optional[Union[int, str]] = 0

        def __post_init__(self):

            if isinstance(self.is_top10_developer, bool):

                object.__setattr__(
                    self,
                    'is_top10_developer',
                    int(self.is_top10_developer)
                )

            if isinstance(self.tenure, str):

                object.__setattr__(
                    self,
                    'tenure',
                    {
                        'leasehold': 0,
                        'freehold': 1
                    }.get(self.tenure.lower(), 0)
                )

    # ----------------------------------------------------------------

    @dataclass(frozen=True)
    class dynamic:

        layers: Optional[Tuple[FloorConfig, ...]] = None

        # ----------------------------------------------------------------

        def __post_init__(self):

            if self.layers and not isinstance(self.layers, tuple):
                object.__setattr__(
                    self,
                    'layers',
                    tuple(self.layers)
                )

        # return a sorted configuration, depending on `kind`:
        #
        #   'units'  -> units are sorted by size within each floor
        #   'floors' -> floors are sorted by area
        #   'full'   -> [all of the above]
        #
        def sorted(self, kind: Literal['units', 'floors', 'full'] = 'units'):

            res = self

            if kind == 'units' or kind == 'both':
                res = type(self)(
                    layers=tuple(
                        layer.sorted()
                        for layer in res.layers
                    )
                )

            if kind == 'floors' or kind == 'both':
                res = type(self)(
                    layers=sorted(
                        res.layers,
                        key=lambda l: l.area(),
                        reverse=True
                    )
                )

            return res

        def max_floor(self):
            return len(self.layers)

        def avg_num_of_stacks(self):
            return round(
                sum(
                    layer.units_total()
                    for layer in self.layers
                ) / len(self.layers)
            )

        def building_area(self):
            return max(
                layer.area()
                for layer in self.layers
            )

        def gross_floor_area(self):
            return sum(
                layer.area()
                for layer in self.layers
            )

        def units_total(self):
            return sum(
                layer.units_total()
                for layer in self.layers
            )

        # table of unit sizes
        #
        def avg_unit_size(self, default=-1):

            bed2idx = bed_to_idx()

            return lambda bed: last(
                idx := bed2idx[bed],

                s := list(
                    flattened(
                        layer.unit_sizes[idx]
                        for layer in self.layers
                    )
                ),

                default
                if len(s) == 0 else
                sum(s) / len(s)
            )

        # table of unit numbers
        #
        def units_per_bed(self):

            bed2idx = bed_to_idx()

            units_per_floor = tuple(
                layer.units_per_bed()
                for layer in self.layers
            )

            return lambda bed: last(
                idx := bed2idx[bed],

                sum(
                    units_per_bed[idx]
                    for units_per_bed in units_per_floor
                )
            )

        def cell_repr(self, tab='\t'):

            generator_len = lambda g: sum(1 for _ in g)

            floor_range_desc = lambda low, high: f'floors {low}..{high}' if low != high else f'floor {low}'

            layers_counted = [
                (
                    layer,
                    generator_len(q)
                )
                for layer, q in groupby(self.layers)
            ]

            layer_acc_counts = list(
                accumulate(
                    n
                    for (_, n) in layers_counted
                )
            )

            return list(
                flattened(
                    [
                        floor_range_desc(
                            1 + n_cumulative - n_layers,
                            n_cumulative
                        )
                        + ':'
                    ]
                    +
                    [
                        tab + line
                        for line in layer.cell_repr(tab=tab)
                    ]
                    +
                    ['']

                    for (layer, n_layers), n_cumulative in zip(
                        reversed(layers_counted),
                        reversed(layer_acc_counts)
                    )
                )
            )

        def disp(
            self,
            indent_level=0,
            display_name=lambda qualname: qualname
        ):

            n = indent_level
            tab = lambda n: '\t' * n

            if callable(display_name):
                display_name_fn = display_name
            else:
                display_name_fn = lambda _: display_name

            return tab(n) + display_name_fn(type(self).__qualname__) + ':\n\n' + '\n'.join(
                tab(n + 1) + line
                for line in self.cell_repr()
            )

        def __str__(self):
            return self.disp()


# ----------------------------------------------------------------------------------

# Full building configuration
#
# note: base classes are 'added' to the final class from right to left
#       hence the 'reversed' order
#
@with_downcast
@with_join
@dataclass(frozen=True)
class BuildingConfig(TBuildingConfig.dynamic, TBuildingConfig.static):

    @property
    def static(self):
        return self.cast(TBuildingConfig.static)

    @property
    def dynamic(self):
        return self.cast(TBuildingConfig.dynamic)

    def sorted(self):
        return type(self).join(
            self.static,
            self.dynamic.sorted()
        )

    def __str__(self):
        return f'{type(self).__name__}:\n' + '\n' + self.static.disp(display_name='') + '\n' + self.dynamic.disp(
            indent_level=1, display_name='.layers'
        )


# an 'unstructured' representation of a building
# (aggregate level)
#
@with_downcast
@dict_constructor
@dataclass(frozen=True)
class UnstructuredBuildingConfig(TBuildingConfig.static):

    total_unit_count: Optional[InfiniteTuple(0)] = None
    avg_unit_size: Optional[InfiniteTuple(None)] = None  # [sqm]
    max_floor: Optional[int] = None
    num_of_stacks: Optional[int] = None

    # ----------------------------------------------------------------

    @classmethod
    def from_structured(cls, cfg: BuildingConfig):

        return cls(
            **cfg.static,

            total_unit_count=map(cfg.units_per_bed(), ALL_BEDS),
            avg_unit_size=map(cfg.avg_unit_size(None), ALL_BEDS),
            max_floor=cfg.max_floor(),
            num_of_stacks=cfg.avg_num_of_stacks()
        )

    def __post_init__(self):

        try:
            super().__post_init__()
        except AttributeError:
            pass

        bed_converter = bed_to_idx()

        key_normalized = lambda d: dict(
            p
            for p in (
                (
                    bed_converter.get(key, None),
                    value
                )
                for key, value in d.items()
            )
            if p[0] is not None
        )

        if is_generator(self.total_unit_count):
            object.__setattr__(
                self,
                'total_unit_count',
                InfiniteTuple(0)(self.total_unit_count)
            )
        elif isinstance(self.total_unit_count, dict):

            total_unit_count = key_normalized(self.total_unit_count)

            object.__setattr__(
                self,
                'total_unit_count',
                InfiniteTuple(0)(
                    total_unit_count.get(bed_idx, 0)
                    for bed_idx, _ in enumerate(ALL_BEDS)
                )
            )

        if is_generator(self.avg_unit_size):
            object.__setattr__(
                self,
                'avg_unit_size',
                InfiniteTuple(None)(self.avg_unit_size)
            )
        elif isinstance(self.avg_unit_size, dict):

            avg_unit_size = key_normalized(self.avg_unit_size)

            object.__setattr__(
                self,
                'avg_unit_size',
                InfiniteTuple(None)(
                    avg_unit_size.get(bed_idx)
                    for bed_idx, _ in enumerate(ALL_BEDS)
                )
            )

    @property
    def static(self):
        return self.cast(TBuildingConfig.static)

    def units_per_bed(self):

        bed2idx = bed_to_idx()

        return lambda bed: self.total_unit_count[bed2idx[bed]]

    def avg_unit_size_per_bed(self):

        bed2idx = bed_to_idx()

        return lambda bed: self.avg_unit_size[bed2idx[bed]]
