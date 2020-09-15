"Distributed MPI communicators"

__all__ = [
    "Comm",
    "Cartcomm",
]


class Comm:
    "MPI communicator"

    __slots__ = [
        "_comms",
        "_ranks",
        "_workers",
    ]

    def __init__(self, comms):
        self._comms = tuple(comms)
        self._ranks = self.client.map(lambda comm: comm.rank, self)
        self._ranks = tuple(_.result() for _ in self._ranks)
        self._workers = tuple(map(next, map(iter, map(self.client.who_has, self))))
        assert len(set(self.workers)) == len(self), "Not unique set of workers"

    @property
    def client(self):
        "Client of the communicator"
        return self._comms[0].client

    @property
    def size(self):
        "Size of the communicator"
        return len(self)

    @property
    def ranks(self):
        "Ranks of the communicator with respective worker"
        return self._ranks

    @property
    def workers(self):
        "Workers of the communicator with respective rank"
        return self._workers

    @property
    def ranks_worker(self):
        "Mapping between ranks and workers"
        return dict(zip(self._ranks, self._workers))

    def create_cart(self, dims, periods=None, reorder=False):
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
        return Cartcomm(
            self.client.map(
                lambda comm: comm.Create_cart(dims, periods=periods, reorder=reorder),
                self,
            )
        )

    def index(self, key):
        "Returns the index of key"
        if isinstance(key, int) and key in self.ranks:
            return self.ranks.index(key)
        if isinstance(key, str) and key in self.workers:
            return self.workers.index(key)
        raise KeyError(f"{key} is neither a rank or a worker of {self}")

    def __getitem__(self, key):
        return self._comms[self.index(key)]

    def __len__(self):
        return len(self._comms)

    def __iter__(self):
        return iter(self._comms)


class Cartcomm(Comm):
    "Cartesian communicator"

    __slots__ = [
        "_dims",
        "_periods",
        "_coords",
    ]

    def __init__(self, comms):
        super().__init__(comms)
        topos = self.client.map(lambda comm: comm.Get_topo(), self)
        topos = tuple(_.result() for _ in topos)
        self._dims = tuple(topos[0][0])
        self._periods = tuple(topos[0][1])
        self._periods = tuple(bool(_) for _ in self._periods)
        self._coords = tuple(tuple(topo[2]) for topo in topos)

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
        "Returns the index of key"
        if isinstance(key, tuple):
            lkey = len(key)
            ldims = len(self.dims)
            if len(key) <= len(self.dims):
                key = key + (0,) * (ldims - lkey)
            elif key[ldims:] == (0,) * (lkey - ldims):
                key = key[:ldims]
            else:
                raise IndexError(f"{key} out of range {self.dims}")
            return self.coords.index(key)
        return super().index(key)
