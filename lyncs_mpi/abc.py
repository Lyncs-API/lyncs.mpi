"""
Some ABC to be used for typing parallel classes
"""

from abc import ABC
from numpy import ndarray
from dask.array import Array as daskArray
from distributed import as_completed


class Result(ABC):
    @classmethod
    def finalize(cls, ftrs):
        return tuple(ftr.result() for ftr in ftrs)


class Global(Result):
    @classmethod
    def finalize(cls, ftrs):
        return next(as_completed(ftrs, with_results=True))[1]


class Array(Result):
    pass


Array.register(ndarray)
Array.register(daskArray)
