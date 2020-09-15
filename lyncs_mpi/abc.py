"""
Some ABC to be used for typing parallel classes
"""

from abc import ABC
from numpy import ndarray
from dask.array import Array as daskArray
from distributed import as_completed


class Result(ABC):
    def __new__(cls, ftrs):
        return tuple(ftr.result() for ftr in ftrs)


class Global(Result):
    def __new__(cls, ftrs):
        res = super().__new__(cls, ftrs)
        ret = res[0]
        if not all((_ == ret for _ in res)):
            raise RuntimeError(f"Expected global value but got different resuts: {res}")
        return ret


class Array(Result):
    def __new__(cls, ftrs):
        return ftrs.comm.array(ftrs)


Array.register(ndarray)
Array.register(daskArray)
