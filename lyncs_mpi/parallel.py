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
from lyncs_utils import isiterable


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
        deque(as_completed(ftrs, raise_errors=True))
        return super().__new__(cls, ftrs)

    @property
    def client(self):
        return next(iter(self)).client

    @property
    def workers(self):
        return tuple(map(next, map(iter, map(self.client.who_has, self))))

    def __getattr__(self, key):
        cls = self[0].type
        try:
            if callable(getattr(cls, key)):
                return lambda *args, **kwargs: self.__callattr__(key, *args, **kwargs)
        except AttributeError:
            pass
        return Parallel(self.client.map(lambda self: getattr(self, key), self))

    def __callattr__(self, key, *args, **kwargs):
        parallel = []
        for i, arg in enumerate(args):
            if isparallel(arg):
                parallel.append((i, arg))

        for key, arg in kwargs.items():
            if isparallel(arg):
                parallel.append((key, arg))

        keys, vals = tuple(zip(*parallel))
        return Parallel(
            self.client.map(
                lambda self, *vals: getattr(self, key)(
                    *replace_args(keys, vals, *args),
                    **replace_kwargs(keys, vals, **kwargs),
                ),
                self,
                *vals,
            )
        )

    def __setattr__(self, key, val):
        Parallel(self.client.map(lambda fut: setattr(fut, key, val), self))

        

def replace_args(keys, vals, *args):
    if not keys:
        return args
    for i, arg in enumerate(args):
        try:
            yield vals[keys.index(i)]
        except IndexError:
            yield arg


def replace_kwargs(keys, vals, **kwargs):
    if not keys:
        return kwargs
    for key, val in zip(keys, vals):
        if isinstance(key, str):
            kwargs[key] = val
    return kwargs


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
        parallel = []
        for i, arg in enumerate(args):
            if isparallel(arg):
                parallel.append((i, arg))

        for key, arg in kwargs.items():
            if isparallel(arg):
                parallel.append((key, arg))

        if not parallel:
            return super().__call__(*args, **kwargs)

        client = parallel[0][1][0].client
        keys, vals = tuple(zip(*parallel))
        return Parallel(
            client.map(
                lambda *vals: cls(
                    *replace_args(keys, vals, *args),
                    **replace_kwargs(keys, vals, **kwargs),
                ),
                *vals,
            )
        )

    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            isparallel(instance)
            and all((issubclass(val.type, cls) for val in instance))
        )


class Test(metaclass=ParallelClass):
    def __init__(self, foo=None):
        self.foo = foo
