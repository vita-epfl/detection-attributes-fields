from enum import Enum
from typing import Dict


class ObjectType(Enum):
    """Enum type for categories of objects."""

    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


"""List of object types for every dataset."""
OBJECT_TYPES: Dict[str, ObjectType] = {}
"""List of attribute meta information for every dataset and object type."""
ATTRIBUTE_METAS: Dict[str, Dict[ObjectType, list]] = {}


def get_attribute_metas(dataset: str,
                        attributes: Dict[ObjectType, list]):
    assert dataset in OBJECT_TYPES
    assert dataset in ATTRIBUTE_METAS
    att_metas = []
    for object_type in OBJECT_TYPES[dataset]:
        if (
            (object_type in attributes)
            and (object_type in ATTRIBUTE_METAS[dataset])
        ):
            att_metas += [{'object_type': object_type, **am}
                for am in ATTRIBUTE_METAS[dataset][object_type] if (
                    (am['attribute'] in attributes[object_type])
                    or (am['group'] in attributes[object_type])
                    or ('all' in attributes[object_type])
            )]
    return att_metas
