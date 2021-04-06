from . import datasets
from . import models


def register():
    datasets.register()
    models.register()
