__all__ = [
    "Serial",
    "Parallel",
    "ParallelClass",
    "CartSerial",
    "CartParallel",
    "CartesianClass",
    "isparallel",
]

from itertools import chain
from distributed.client import Future
from distributed import as_completed
from collections import deque
from functools import wraps
from lyncs_utils import isiterable, interactive
import numpy as np
from .abc import Result, Global, Array
from .comm import Cartcomm


def isparallel(val):
    return isiterable(val, Future)


def anyparallel(*args, **kwargs):
    return any((isparallel(val) for val in chain(args, kwargs.values())))


def insert_args(idxs, vals, *args):
    if not idxs:
        return args
    args = list(args)
    for i, arg in zip(idxs, vals):
        args.insert(i, arg)
    return tuple(args)


class Serial:
    @property
    def dask(self):
        return None

    @property
    def client(self):
        return None

    @property
    def workers(self):
        return ("localhost",)

    @property
    def type(self):
        return type(self)

    def index(self, key):
        if key in self.workers:
            return 0
        raise KeyError(f"Key {key} not found")

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, key):
        assert self.index(key) == 0
        return self


class Parallel:
    __slots__ = [
        "_dask",
    ]

    def __init__(self, dask):
        self._dask = tuple(dask)
        for ftr in as_completed(self.dask):
            if ftr.status == "error":
                ftr.result()

    @property
    def dask(self):
        return self._dask

    @property
    def client(self):
        return next(iter(self)).client

    @property
    def workers(self):
        return tuple(map(next, map(iter, map(self.client.who_has, self))))

    @property
    def type(self):
        return next(iter(self)).type

    def index(self, key):
        if isinstance(key, str) in self.workers:
            return self.workers
        raise KeyError(f"Key {key} not found")

    def __iter__(self):
        return iter(self.dask)

    def __len__(self):
        return len(self.dask)

    def __getitem__(self, key):
        return self.dask[self.index(key)]

    @classmethod
    def _get_finalize(cls, fnc):
        if hasattr(fnc, "__annotations__"):
            ret = fnc.__annotations__.get("return", None)
            if callable(ret):
                return ret
        if isinstance(fnc, property):
            return cls._get_finalize(fnc.fget)
        return lambda _: _

    @classmethod
    def _remote_call(cls, fnc, *args, **kwargs):
        if isinstance(fnc, str):
            call = lambda self, *args, **kwargs: getattr(self, fnc)(*args, **kwargs)
        elif isparallel(fnc):
            call = lambda fnc, *args, **kwargs: fnc(*args, **kwargs)
            args = (fnc,) + args
        else:
            call = fnc

        keys = []
        vals = []
        for i, arg in enumerate(args):
            if isparallel(arg):
                keys.append(i)
                vals.append(arg)

        args = tuple(arg for i, arg in enumerate(args) if i not in keys)
        n_args = len(keys)

        for key, arg in list(kwargs.items()):
            if isparallel(arg):
                keys.append(key)
                vals.append(arg)
                kwargs.pop(key)

        if not vals:
            raise ValueError("No parallel argument found when calling fnc")

        client = next(iter(vals[0])).client

        return cls(
            client.map(
                lambda *vals: call(
                    *insert_args(keys[:n_args], vals[:n_args], *args),
                    **kwargs,
                    **dict(zip(keys[n_args:], vals[n_args:])),
                ),
                *vals,
            )
        )

    def __getattr__(self, key):
        try:
            cls = self.type
            attr = getattr(cls, key)
            finalize = self._get_finalize(attr)
            if callable(attr):
                fnc = lambda *args, **kwargs: finalize(
                    self._remote_call(key, self, *args, **kwargs)
                )
                if interactive():
                    return wraps(attr)(fnc)
                return fnc
            return finalize(self._remote_call(getattr, self, key))
        except AttributeError:
            pass
        return self._remote_call(getattr, self, key)

    def __setattr__(self, key, val):
        if key in dir(type(self)):
            return super().__setattr__(key, val)
        self._remote_call(setattr, self, key, val)

    def __call__(self, *args, **kwargs):
        return self._remote_call(self, *args, **kwargs)


class ParallelClass(type):
    "Metaclass for Dask parallel classes"

    @staticmethod
    def serial_class():
        return Serial

    @staticmethod
    def parallel_class():
        return Parallel

    def __new__(cls, name, bases, class_attrs, **kwargs):
        "Checks that cls does not have methods of Serial and adds Serial to bases"
        serial = cls.serial_class()
        methods = set(dir(serial)).difference(dir(object) + ["__module__"])
        same = methods.intersection(class_attrs)
        if same:
            raise KeyError(f"Class {name} cannot define {same}")
        if Serial not in bases:
            bases = (serial,) + bases
        return super().__new__(cls, name, bases, class_attrs, **kwargs)

    def __call__(cls, *args, **kwargs):
        if anyparallel(*args, **kwargs):
            return type(cls).parallel_class()._remote_call(cls, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            isparallel(instance)
            and all((issubclass(val.type, cls) for val in instance))
        )


class CartSerial(Serial):
    @property
    def comm(self):
        return None

    @property
    def coords(self):
        return (0,)

    @property
    def procs(self):
        return (1,)


class CartParallel(Parallel):
    __slots__ = [
        "_comm",
    ]

    def __init__(self, comm, ftrs):
        assert isinstance(comm, Cartcomm)
        super().__init__(ftrs)
        self._comm = comm

    @property
    def comm(self):
        return self._comm

    @property
    def coords(self):
        "Coordinates of the cartesian communicator"
        return self.comm.coords

    @property
    def procs(self):
        "Number of processes per dimension"
        return self.comm.dims

    @classmethod
    def _remote_call(cls, *args, **kwargs):
        # Looking for cartesian comm
        comms = None
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, (Cartcomm, CartParallel)):
                if comms is not None:
                    raise ValueError(
                        "Multiple cartesian communicator passed to class init"
                    )
                comms = arg
        if comms is None:
            raise ValueError(
                "Cartesian communicator not found in class init in parallel mode"
            )
        if isinstance(comms, CartParallel):
            comms = comms.comm
        return cls(comms, Parallel._remote_call(*args, **kwargs))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return super().__getitem__(self.coords.index(key))
        return super().__getitem__(key)


class CartesianClass(ParallelClass):
    "Metaclass for cartesian parallel classes"

    @staticmethod
    def serial_class():
        return CartSerial

    @staticmethod
    def parallel_class():
        return CartParallel


class Test(metaclass=ParallelClass):
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
