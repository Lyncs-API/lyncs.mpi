"""
Just some simple classes used by the tests.
(They are needed here to avoid pickling issues)
"""

import numpy as np
from .abc import Result, Global, Constant, Array
from .distributed import DistributedClass
from .cartesian import CartesianClass


class DistributedTest(metaclass=DistributedClass):
    "A distributed test class"

    def __init__(self, value=None):
        self.value = value

    @property
    def ten(self) -> Constant:
        "Returns 10"
        return 10

    def range(self, n) -> Global:
        "Returns range(n)"
        return range(n)

    def values(self) -> Result:
        "Returns the values"
        return self.value


class CartesianTest(DistributedTest, metaclass=CartesianClass):
    "A cartesian distributed test class"

    def __init__(self, lshape, value=None, dtype="int", comm=None):
        super().__init__(value)
        self._lshape = lshape
        if comm:
            procs = comm.dims
            if len(procs) < len(lshape):
                procs += (1,) * (len(lshape) - len(procs))
            elif len(procs) > len(lshape):
                assert all((i == 1 for i in procs[len(lshape) :]))
            self._shape = tuple(i * j for i, j in zip(lshape, procs))
        else:
            self._shape = lshape
        self._dtype = dtype

    @property
    def lshape(self) -> Constant:
        return self._lshape

    @property
    def shape(self) -> Constant:
        return self._shape

    @property
    def dtype(self) -> Constant:
        return self._dtype

    def ones(
        self,
    ) -> Array(
        shape=lambda self: self.shape,
        chunks=lambda self: self.lshape,
        dtype=lambda self: self.dtype,
    ):
        "Returns an array of ones"
        return np.ones(self.lshape, self.dtype)

    def mul_by_value(self, arr) -> Array:
        "Multiplies the array by value"
        return arr * self.value
