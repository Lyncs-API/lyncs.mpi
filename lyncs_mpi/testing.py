import numpy as np
from .abc import Result, Global, Array
from .distributed import DistributedClass
from .cartesian import CartesianClass


class Test(metaclass=DistributedClass):
    def __init__(self, foo=None):
        self.foo = foo

    def ten(self) -> Global:
        return 10

    def values(self) -> Result:
        return self.foo


class CartTest(Test, metaclass=CartesianClass):
    def __init__(self, foo=None, comm=None):
        super().__init__(foo)

    def ones(self, lshape, **kwargs) -> Array:
        return np.ones(lshape, **kwargs)
