"Distributed MPI communicators"

__all__ = [
    "Comm",
    "CartComm",
]

from lyncs_utils import compute_property
from .distributed import Distributed, results


class Comm(Distributed):
    "MPI communicator"

    __slots__ = [
        "_ranks",
    ]

    def __init__(self, comms):
        super().__init__(comms, cls=self.type)

    @property
    def type(self):
        "Returns the MPI type of the class"
        # pylint: disable=import-outside-toplevel
        from mpi4py import MPI

        return MPI.Comm

    @property
    def size(self):
        "Size of the communicator"
        return len(self)

    @compute_property
    def ranks(self):
        "Ranks of the communicator with respective worker"
        return results(*self.rank)

    @property
    def ranks_workers(self):
        "Mapping between ranks and workers"
        return dict(zip(self.ranks, self.workers))

    def create_cart(self, dims, periods=True, reorder=False):
        """
        Makes a new communicator to which topology information has been attached

        Parameters
        ----------
        dims: list
            integer array specifying the number of processes in each dimension
        periods: boolean [list]
            logical [array] specifying whether the grid is periodic (True) or not (False)
        reorder: boolean
            ranking may be reordered (True) or not (False)
        """
        return CartComm(self.Create_cart(dims, periods=periods, reorder=reorder))

    def index(self, key):
        "Returns the index of key that can be either rank(int) or worker(str)"
        if isinstance(key, int) and key in self.ranks:
            return self.ranks.index(key)
        if isinstance(key, str) and key in self.workers:
            return self.workers.index(key)
        raise KeyError(f"{key} is neither a rank or a worker of {self}")


class CartComm(Comm):
    "Cartesian communicator"

    __slots__ = [
        "_dims",
        "_periods",
        "_coords",
    ]

    def __init__(self, comms):
        super().__init__(comms)
        topos = results(*self.Get_topo())
        self._dims = tuple(topos[0][0])
        self._periods = tuple(bool(_) for _ in topos[0][1])
        self._coords = tuple(tuple(topo[2]) for topo in topos)

    @property
    def type(self):
        "Returns the MPI type of the class"
        # pylint: disable=import-outside-toplevel
        from mpi4py import MPI

        return MPI.Cartcomm

    @property
    def dims(self):
        "Dimensions of the cartesian communicator"
        return self._dims

    @property
    def periods(self):
        "Periodicity of the cartesian communicator"
        return self._periods

    @property
    def coords(self):
        "Coordinates of the cartesian communicator"
        return self._coords

    @property
    def ranks_coords(self):
        "Coordinates of the ranks of the cartesian communicator"
        return dict(zip(self.ranks, self.coords))

    def index(self, key):
        "Returns the index of key that can be either rank(int) or worker(str) or coord(tuple)"
        if isinstance(key, tuple):
            lkey = len(key)
            ldims = len(self.dims)
            if len(key) <= len(self.dims):
                key = key + (0,) * (ldims - lkey)
            elif key[ldims:] == (0,) * (lkey - ldims):
                key = key[:ldims]
            else:
                raise KeyError(f"{key} out of range {self.dims}")
            return self.coords.index(key)
        return super().index(key)
