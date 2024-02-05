from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class PathsCollections:
    project_name: Optional[str]
    num_of_bedrooms: Optional[Union[int, str]]
    paths: Optional[str]
