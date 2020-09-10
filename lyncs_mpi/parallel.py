__all__ = [
    "Serial",
    "Parallel",
    "ParallelClass",
    "isparallel",
]

from itertools import chain
from distributed.client import Future
from distributed import as_completed
from collections import deque
from functools import wraps
from lyncs_utils import isiterable, interactive
from .abc import Result, Global


def isparallel(val):
    return isiterable(val, Future)


class Serial:
    @property
    def client(self):
        return None

    @property
    def workers(self):
        return ("localhost",)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, key):
        if key != 0:
            raise IndexError("index out of range")
        return self


class Parallel(tuple):
    def __new__(cls, ftrs):
        for ftr in as_completed(ftrs):
            if ftr.status == "error":
                ftr.result()
        return super().__new__(cls, ftrs)

    @property
    def client(self):
        return next(iter(self)).client

    @property
    def workers(self):
        return tuple(map(next, map(iter, map(self.client.who_has, self))))

    def __getattr__(self, key):
        cls = self[0].type
        finalize = True
        try:
            attr = getattr(cls, key)
            finalize = get_finalize(attr)
            if callable(attr):
                fnc = lambda *args, **kwargs: parallel_call(
                    key, self, *args, finalize=finalize, **kwargs
                )
                if interactive():
                    return wraps(attr)(fnc)
                return fnc
        except AttributeError:
            pass
        return parallel_call(getattr, self, key, finalize=finalize)

    def __setattr__(self, key, val):
        parallel_call(setattr, self, key, val)

    def __call__(self, *args, **kwargs):
        return parallel_call(self, *args, **kwargs)


def insert_args(idxs, vals, *args):
    if not idxs:
        return args
    args = list(args)
    for i, arg in zip(idxs, vals):
        args.insert(i, arg)
    return tuple(args)


def get_return(fnc):
    if hasattr(fnc, "__annotations__"):
        return fnc.__annotations__.get("return", None)
    if isinstance(fnc, property):
        return get_return(fnc.fget)
    return None


def get_finalize(val):
    if not val:
        return lambda _: _
    ret = get_return(val)
    if ret and (
        isinstance(ret, Result) or (isinstance(ret, type) and issubclass(ret, Result))
    ):
        return ret.finalize
    return Parallel


def parallel_call(fnc, *args, finalize=True, **kwargs):

    if isinstance(fnc, str):
        call = lambda self, *args, **kwargs: getattr(self, fnc)(*args, **kwargs)
    elif isparallel(fnc):
        call = lambda fnc, *args, **kwargs: fnc(*args, **kwargs)
        args = (fnc,) + args
    else:
        call = fnc

    if not callable(finalize):
        finalize = get_finalize(fnc)

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

    assert vals
    client = vals[0][0].client

    return finalize(
        client.map(
            lambda *vals: call(
                *insert_args(keys[:n_args], vals[:n_args], *args),
                **kwargs,
                **dict(zip(keys[n_args:], vals[n_args:])),
            ),
            *vals,
        )
    )


class ParallelClass(type):
    def __new__(cls, name, bases, class_attrs, **kwargs):
        "Checks that cls does not have methods of Serial and adds Serial to bases"
        methods = set(dir(Serial)).difference(dir(object) + ["__module__"])
        same = methods.intersection(class_attrs)
        if same:
            raise KeyError(f"Class {name} cannot define {same}")
        if Serial not in bases:
            bases = (Serial,) + bases
        return super().__new__(cls, name, bases, class_attrs, **kwargs)

    def __call__(cls, *args, **kwargs):
        if any((isparallel(val) for val in chain(args, kwargs.values()))):
            return parallel_call(cls, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            isparallel(instance)
            and all((issubclass(val.type, cls) for val in instance))
        )


class Test(metaclass=ParallelClass):
    def __init__(self, foo=None):
        self.foo = foo

    def ten(self) -> Global:
        return 10

    def values(self) -> Result:
        return self.foo
