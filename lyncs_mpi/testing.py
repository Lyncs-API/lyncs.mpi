"""
Just some simple classes used by the tests.
(They are needed here to avoid pickling issues)
"""

import numpy as np
from .abc import Result, Global, Array
from .distributed import DistributedClass
from .cartesian import CartesianClass


class Test(metaclass=DistributedClass):
    "A distributed test class"

    def __init__(self, value=None):
        self.value = value

    @property
    def ten(self) -> Global:
        "Returns 10"
        return 10

    def values(self) -> Result:
        "Returns the values"
        return self.value


class CartTest(Test, metaclass=CartesianClass):
    "A cartesian distributed test class"

    def __init__(self, value=None, comm=None):
        super().__init__(value)

    def ones(self, lshape, **kwargs) -> Array:
        "Returns an array of ones"
        return np.ones(lshape, **kwargs)

    def mul_by_value(self, arr) -> Array:
        "Multiplies the array by value"
        return arr * self.value
