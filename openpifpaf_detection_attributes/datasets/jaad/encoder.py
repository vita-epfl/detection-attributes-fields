from .attribute import JaadType
from .. import generators


JAAD_ATTRIBUTE_GENERATORS = {
    JaadType.PEDESTRIAN: generators.BoxAttributeGenerator,
}
