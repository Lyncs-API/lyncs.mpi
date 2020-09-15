"Tools for dask distributed classes"

__all__ = [
    "isdistributed",
    "anydistributed",
    "Distributed",
    "Local",
    "DistributedClass",
]

from itertools import chain
from functools import wraps
from inspect import isclass
from distributed.client import Future
from distributed import as_completed
from lyncs_utils import isiterable, interactive, compute_property


def isdistributed(val):
    "Returns if the argument is a Dask distributed object"
    return isiterable(val, Future)


def anydistributed(*args, **kwargs):
    "Returns if any of the arguments is a Dask distributed object"
    return any((isdistributed(val) for val in chain(args, kwargs.values())))


class Distributed:
    "Class for handling distributed objects"

    __slots__ = [
        "_dask",
        "_workers",
        "_type",
    ]

    def __init__(self, dask, wait=False, cls=None):
        self._dask = tuple(dask)
        if wait:
            self.wait()
        if cls:
            self._type = cls

    def wait(self):
        "Waits for all the distributed futures to be completed and raises errors if any"
        for ftr in as_completed(self.dask):
            if ftr.status == "error":
                ftr.result()

    @property
    def dask(self):
        "Returns the low-level dask objects"
        return self._dask

    @property
    def client(self):
        "Returns the Client managing the distributed objects"
        return next(iter(self)).client

    @compute_property
    def workers(self):
        "Returns the of workers holding the distributed objects"
        self.wait()
        return tuple(map(next, map(iter, map(self.client.who_has, self))))

    @compute_property
    def type(self):
        "Returns the class of the distributed objects"
        self.wait()
        return next(iter(self)).type

    def index(self, key):
        "Returns the index of the dask futures (self.dask) for a given key"
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
        elif isdistributed(fnc):
            call = lambda fnc, *args, **kwargs: fnc(*args, **kwargs)
            args = (fnc,) + args
        else:
            call = fnc

        keys = []
        vals = []
        for i, arg in enumerate(args):
            if isdistributed(arg):
                keys.append(i)
                vals.append(arg)

        args = tuple(arg for i, arg in enumerate(args) if i not in keys)
        n_args = len(keys)

        for key, arg in list(kwargs.items()):
            if isdistributed(arg):
                keys.append(key)
                vals.append(arg)
                kwargs.pop(key)

        if not vals:
            raise ValueError("No distributed argument found when calling fnc")

        client = next(iter(vals[0])).client

        return Distributed(
            client.map(
                lambda *vals: call(
                    *insert_args(keys[:n_args], vals[:n_args], *args),
                    **kwargs,
                    **dict(zip(keys[n_args:], vals[n_args:])),
                ),
                *vals,
            ),
            cls=call if isclass(call) else None,
        )

    def __getattr__(self, key):
        if key in dir(type(self)):
            return getattr(type(self), key).__get__(self)
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
            super().__setattr__(key, val)
        else:
            self._remote_call(setattr, self, key, val)

    def __call__(self, *args, **kwargs):
        return self._remote_call(self, *args, **kwargs)


def insert_args(idxs, vals, *args):
    "Inserts vals into args at idxs position"
    if not idxs:
        return args
    args = list(args)
    for i, arg in zip(idxs, vals):
        args.insert(i, arg)
    return tuple(args)


class Local:
    "Mock class for enabling functions of Distributed into a local class"

    @property
    def dask(self):
        "Mock of Distributed.dask"
        return None

    @property
    def client(self):
        "Mock of Distributed.client"
        return None

    @property
    def workers(self):
        "Mock of Distributed.workers"
        return ("localhost",)

    @property
    def type(self):
        "Mock of Distributed.type"
        return type(self)

    def wait(self):
        "Mock of Distributed.wait"
        return

    def index(self, key):
        "Mock of Distributed.index"
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


class DistributedClass(type):
    "Metaclass for Dask distributed classes"

    @staticmethod
    def local_class():
        "The local class to use"
        return Local

    @staticmethod
    def distributed_class():
        "The distributed class to use"
        return Distributed

    def __new__(cls, name, bases, class_attrs, **kwargs):
        "Checks that cls does not have methods of Local and adds Local to bases"
        local = cls.local_class()
        methods = set(dir(local)).difference(dir(object) + ["__module__"])
        same = methods.intersection(class_attrs)
        if same:
            raise KeyError(f"Class {name} cannot define {same}")
        if local not in bases:
            bases = (local,) + bases
        return super().__new__(cls, name, bases, class_attrs, **kwargs)

    def __call__(cls, *args, **kwargs):
        if anydistributed(*args, **kwargs):
            return type(cls).distributed_class()._remote_call(cls, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            isdistributed(instance) and issubclass(Distributed(instance).type, cls)
        )