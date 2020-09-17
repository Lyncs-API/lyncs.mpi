"Tools for distributed classes on a Cartesian communicator"

__all__ = [
    "CommLocal",
    "Cartesian",
    "CartesianClass",
]

from functools import wraps
from itertools import chain
from dask.array import Array
from .comm import CartComm
from .distributed import Distributed, DistributedClass, Local
from .cart_array import get_cart_arrays


class Cartesian(Distributed):
    "Class for distributed objects on a Cartesian communicator"

    __slots__ = [
        "_comm",
    ]

    def __init__(self, ftrs, comm):
        assert isinstance(comm, CartComm)
        super().__init__(ftrs)
        self._comm = comm

    @property
    def comm(self):
        "Returns the communicator"
        return self._comm

    @property
    def coords(self):
        "Coordinates of the cartesian communicator"
        return self.comm.coords

    @property
    def procs(self):
        "Number of processes per dimension"
        return self.comm.dims

    @property
    def ranks(self):
        "Ranks of the communicator"
        return self.comm.ranks

    @classmethod
    def _remote_call(cls, *args, **kwargs):
        # Looking for cartesian comm
        comms = None
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, Cartesian):
                arg = arg.comm
            if isinstance(arg, CartComm):
                if comms is not None and comms != arg:
                    raise ValueError(
                        "Different cartesian communicator passed to class init"
                    )
                comms = arg
        if comms is None:
            raise ValueError(
                "Cartesian communicator not found in class init in parallel mode"
            )

        # Looking for dask arrays
        get_arrays = (
            lambda arr: get_cart_arrays(comms, arr, wait=False)
            if isinstance(arr, Array)
            else arr
        )
        args = (get_arrays(arg) for arg in args)
        kwargs = {key: get_arrays(arg) for key, arg in kwargs.items()}

        return Cartesian(super()._remote_call(*args, **kwargs), comms)

    @wraps(CartComm.index)
    def index(self, key):
        return self.comm.index(key)


class CommLocal(Local):
    "Mock class of Cartesian"

    @property
    def comm(self):
        "Mock of Cartesian.comm"
        return None

    @property
    def coords(self):
        "Mock of Cartesian.coords"
        return ((0,),)

    @property
    def procs(self):
        "Mock of Cartesian.procs"
        return (1,)

    @property
    def ranks(self):
        "Mock of Cartesian.ranks"
        return (0,)

    def index(self, key):
        "Mock of Cartesian.index"
        if key == 0:
            return 0
        if isinstance(key, tuple) and all((_ == 0 for _ in key)):
            return 0
        return super().index(key)


class CartesianClass(DistributedClass):
    "Metaclass for cartesian distributed classes"

    @staticmethod
    def local_class():
        return CommLocal

    @staticmethod
    def distributed_class():
        return Cartesian
