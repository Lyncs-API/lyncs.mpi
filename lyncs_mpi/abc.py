"""
Some ABC to be used for typing parallel classes
"""
# pylint: disable=too-few-public-methods

__all__ = [
    "Result",
    "Global",
    "Array",
]

from abc import ABC
from numpy import ndarray
from dask.array import Array as daskArray
from .distributed import results
from .cartesian import Cartesian


class Result(ABC):
    "Returns a tuple with the result of the remote calls"

    def __new__(cls, ftrs):
        return results(*ftrs)


class Global(Result):
    "Returns a single value supposed to be the same result for all the remote calls"

    def __new__(cls, ftrs):
        res = super().__new__(cls, ftrs)
        ret = res[0]
        if not all((_ == ret for _ in res)):
            raise RuntimeError(f"Expected global value but got different resuts: {res}")
        return ret


class Array(Result):
    "Returns a dask array supposing the futures output to be numpy-like arrays"

    def __new__(cls, ftrs):
        assert isinstance(ftrs, Cartesian)
        return ftrs.comm.array(ftrs)


Array.register(ndarray)
Array.register(daskArray)
