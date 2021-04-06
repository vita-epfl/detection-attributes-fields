from dataclasses import dataclass
from typing import Dict, List, Union

import openpifpaf

from .attribute import ObjectType


@dataclass
class AttributeMeta(openpifpaf.headmeta.Base):
    object_type: ObjectType
    attribute: str
    group: str
    only_on_instance: bool
    is_classification: bool
    is_scalar: bool
    is_spatial: bool
    n_channels: int
    mean: Union[float, List[float]] = None
    std: Union[float, List[float]] = None
    default: Union[int, float, List[float]] = None
    labels: Dict[int, str] = None


    @property
    def n_fields(self):
        return 1
