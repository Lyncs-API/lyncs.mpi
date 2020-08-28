"""
Utils for interfacing dask and MPI.
Most of the content of this file is explained in the [notebook](notebooks/Dask-mpi.ipynb).
"""

__all__ = [
    "default_client",
    "Client",
]

import os
import time
import shutil
import signal
import atexit
import tempfile
import multiprocessing
from functools import wraps

import sys
from packaging import version
import sh

# The following can be deleted when sh==1.13.2 has been released
if version.parse(sh.__version__) < version.parse("1.13.2"):

    sys.meta_path.pop(0)

    def find_spec(self, fullname, path=None, target=None):
        """ find_module() is deprecated since Python 3.4 in favor of find_spec() """

        from importlib.machinery import ModuleSpec

        found = self.find_module(fullname, path)
        return ModuleSpec(fullname, found) if found is not None else None

    sh._SelfWrapper__self_module.ModuleImporterFromVariables.find_spec = find_spec
    sh._SelfWrapper__self_module.register_importer()


from dask_mpi import initialize
from dask.distributed import Client as _Client
from dask.distributed import default_client as _default_client
from .lib import default_comm


@wraps(_default_client)
def default_client():
    "Returns the default client"
    client = _default_client()
    assert isinstance(client, Client), "No MPI client found"
    return client


class Client(_Client):
    "Wrapper to dask.distributed.Client"

    def __init__(self, num_workers=None, threads_per_worker=1, launch=None):
        """
        Returns a Client connected to a cluster of `num_workers` workers.
        """
        self._server = None

        if launch is None:
            launch = default_comm().size == 1

        # pylint: disable=import-outside-toplevel,
        if not launch:
            # Then the script has been submitted in parallel with mpirun
            num_workers = num_workers or default_comm().size - 1
            assert (
                default_comm().size == num_workers + 1
            ), """
            Error: (num_workers + 1) processes required.
            The script has not been submitted on enough processes.
            Got %d processes instead of %d.
            """ % (
                default_comm().size,
                num_workers + 1,
            )

            initialize(nthreads=threads_per_worker, nanny=False)

            _Client.__init__(self)

        else:
            num_workers = num_workers or (multiprocessing.cpu_count() + 1)

            # Since dask-mpi produces several file we create a temporary directory
            self._dir = tempfile.mkdtemp()
            self._out = self._dir + "/log.out"
            self._err = self._dir + "/log.err"

            # The command runs in the background (_bg=True)
            # and the stdout(err) is stored in self._out(err)
            pwd = os.getcwd()
            sh.cd(self._dir)
            self._server = sh.mpirun(
                "-n",
                num_workers + 1,
                "dask-mpi",
                "--no-nanny",
                "--nthreads",
                threads_per_worker,
                "--scheduler-file",
                "scheduler.json",
                _bg=True,
                _out=self._out,
                _err=self._err,
            )
            sh.cd(pwd)

            atexit.register(self.close_server)

            _Client.__init__(self, scheduler_file=self._dir + "/scheduler.json")

        # Waiting for all the workers to connect
        def handler(signum, frame):
            if self._server is not None:
                self.close_server()
            raise RuntimeError(
                "Couldn't connect to %d processes. Got %d workers."
                % (num_workers, len(self.workers))
            )

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)

        while len(self.workers) != num_workers:
            time.sleep(0.001)

        signal.alarm(0)

        self.ranks = {key: val["name"] for key, val in self.workers.items()}
        self._comm = self.create_comm()

    @property
    def comm(self):
        "Returns the global communicator of the clients"
        return self._comm

    @property
    def workers(self):
        "Returns the list of workers."
        return self.scheduler_info()["workers"]

    def close_server(self):
        """
        Closes the running server
        """
        assert self._server is not None
        self.shutdown()
        self.close()
        self._server.wait()
        shutil.rmtree(self._dir)
        self._server = None
        atexit.unregister(self.close_server)

    def __del__(self):
        """
        In case of server started, closes the server
        """
        if self._server is not None:
            self.close_server()

        if hasattr(_Client, "__del__"):
            _Client.__del__(self)

    def who_has(self, *args, overload=True, **kwargs):
        """
        Overloading of distributed.Client who_has.
        Checks that only one worker owns the futures and returns the list of workers.

        Parameters
        ----------
        overload: bool, default true
            If false the original who_has is used
        """
        if overload:
            _workers = list(_Client.who_has(self, *args, **kwargs).values())
            workers = [w[0] for w in _workers if len(w) == 1]
            assert len(workers) == len(
                _workers
            ), "More than one process has the same reference"
            return workers

        return _Client.who_has(self, *args, **kwargs)

    def select_workers(
        self, num_workers=None, workers=None, exclude=None, resources=None
    ):
        """
        Selects `num_workers` from the one available.

        Parameters
        ----------
        workers: list, default all
          List of workers to choose from.
        exclude: list, default none
            List of workers to exclude from the total.
        resources: dict, default none
            Defines the resources the workers should have.
        """

        if not workers:
            workers = list(self.ranks.keys())

        workers = set(workers)
        workers = workers.intersection(self.ranks.keys())

        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            workers = workers.difference(exclude)

        if resources:
            # TODO select accordingly requested resources
            raise NotImplementedError("Resources not implemented.")

        if not num_workers:
            num_workers = len(workers)

        if num_workers > len(workers):
            raise RuntimeError("Available workers are less than required")

        # TODO implement some rules to choose wisely n workers
        # e.g. workers less busy, close to each other, etc
        selected = list(workers)[:num_workers]

        return selected

    def create_comm(self, actor=False, **kwargs):
        """
        Return a MPI communicator involving workers available by the client.

        Parameters
        ----------
        actor: bool, default True
            Wether the returned communicator should be a dask actor.
        **kwargs: params
            Following list of parameters for the function select_workers.
        """

        workers = self.select_workers(**kwargs)
        ranks = [[self.ranks[w] for w in workers]] * len(workers)
        ranks = self.scatter(ranks, workers=workers, hash=False, broadcast=False)

        # Checking the distribution of the group
        _workers = self.who_has(ranks)
        assert set(workers) == set(
            _workers
        ), """
        Error: Something wrong with scatter. Not all the workers got a piece.
        Expected workers = %s
        Got workers = %s
        """ % (
            workers,
            _workers,
        )

        def _create_comm(ranks):
            comm = default_comm()
            return comm.Create_group(comm.group.Incl(ranks))

        return Comm(self.map(_create_comm, ranks, actor=actor))


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

    def __getitem__(self, key):
        if isinstance(key, int) and key in self.ranks:
            return self._comms[self.ranks.index(key)]
        if isinstance(key, str) and key in self.workers:
            return self._comms[self.workers.index(key)]
        raise KeyError(f"{key} is neither a rank or a worker of {self}")

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

    def __getitem__(self, key):
        if isinstance(key, tuple) and key in self.coords:
            return self._comms[self.coords.index(key)]
        return super().__getitem__(key)
