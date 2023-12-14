from dataclasses import dataclass
from typing import Optional

from src.main.building_config import TBuildingConfig


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    launching_year: int
    launching_month: int
    tenure: str

    total_unit_count: tuple
    avg_unit_size: tuple
    max_floor: int
    num_of_stacks: int

    completion_year: Optional[int] = None
    is_top10_developer: Optional[int] = None

    def get_units_count(self, bed):
        return self.total_unit_count[bed - 1]

    def avg_unit_size_per_bed(self, bed):
        return self.avg_unit_size[bed - 1]

    @property
    def static_config(self):
        return TBuildingConfig.static(
            **{
                f: self.__getattribute__(f) if f != 'postal_code' else self.project_name
                for f in TBuildingConfig.static.__dataclass_fields__.keys()
            }
        )

    @property
    def available_bed(self):
        return [idx+1 for idx, num in enumerate(self.total_unit_count) if num != 0]

    @property
    def gross_floor_area(self):
        return sum(
            self.get_units_count(b)*self.avg_unit_size_per_bed(b)
            for b in self.available_bed
        )

    def display(self):

        for k, v in self.__dict__.items():
            print(f'{k}: {v}')