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
from functools import partial
from numpy import ndarray
from dask.array import Array as daskArray
from .distributed import results, Distributed
from .cartesian import Cartesian


class Result(ABC):
    "Returns a tuple with the result of the remote calls"

    def __new__(cls, ftrs, **kwargs):
        return results(*ftrs)


class Global(Result):
    "Returns a single value supposed to be the same result for all the remote calls"

    def __new__(cls, ftrs, **kwargs):
        res = super().__new__(cls, ftrs)
        ret = res[0]
        if not all((_ == ret for _ in res)):
            raise RuntimeError(f"Expected global value but got different resuts: {res}")
        return ret


class Constant(Global):
    "Like Global, but the value is stored for future use and avoiding remote call"

    def __new__(cls, ftrs, caller=None, key=None, **kwargs):
        assert key, "Expected a key"
        assert caller and isinstance(
            caller, Distributed
        ), "Expected a Distributed caller. Got {caller}"
        ret = super().__new__(cls, ftrs, caller=caller, key=key, **kwargs)
        caller._constants[key] = ret
        return ret


class Array(Result):
    "Returns a dask array supposing the futures output to be numpy-like arrays"

    def __new__(cls, *args, **kwargs):
        if not args:
            return partial(cls, **kwargs)

        assert len(args) == 1
        ftrs = args[0]
        assert isinstance(ftrs, Cartesian)

        array_kwargs = {}
        for key in ("shape", "chunks", "dtype", "dims_axes"):
            if key not in kwargs:
                continue
            val = kwargs[key]
            if callable(val):
                assert "caller" in kwargs
                val = val(kwargs["caller"])
            array_kwargs[key] = val

        return ftrs.comm.array(ftrs, **array_kwargs)


Array.register(ndarray)
Array.register(daskArray)
