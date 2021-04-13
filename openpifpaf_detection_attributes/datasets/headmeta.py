from dataclasses import dataclass
from typing import Dict, List, Union

import openpifpaf

from .attribute import ObjectType


@dataclass
class AttributeMeta(openpifpaf.headmeta.Base):
    """Meta information about an attribute.

    Args:
        object_type (ObjectType): Type of object annotated.
        attribute (str): Name of attribute.
        group (str): Group of attribute.
        only_on_instance (bool): Compute targets only on instances.
        is_classification (bool): Classification or regression attribute.
        is_scalar (bool): Scalar or vectorial attribute.
        is_spatial (bool): Attribute affected by stride.
        n_channels (int): Number of channels for annotations.
        mean (Union[float, List[float]]): Mean of attribute for normalization.
        std (Union[float, List[float]]): Standard deviation of attribute for
            normalization.
        default (Union[int, float, List[float]]): Default prediction for
            classificatione evaluation.
        labels (Dict[int, str]): Names of classes.
    """

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
