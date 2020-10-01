"""
Specialized dask Client for MPI communicators
"""

__all__ = [
    "default_client",
    "Client",
]

import os
import sys
import time
import shutil
import signal
import atexit
import tempfile
import multiprocessing
from functools import wraps
import sh
from dask_mpi import initialize
from dask.distributed import Client as _Client
from dask.distributed import default_client as _default_client
from .lib import default_comm
from .comm import Comm


@wraps(_default_client)
def default_client():
    "Returns the default client"
    client = _default_client()
    assert isinstance(client, Client), "No MPI client found"
    return client


class Client(_Client):
    """
    Subclass of dask.distributed.Client specialized for MPI communicators
    The initialization follows the guidelines of http://mpi.dask.org/
    automatizing the process of creating MPI-distributed dask workers.
    """

    def __init__(
        self, num_workers=None, threads_per_worker=1, launch=None, out=None, err=None
    ):
        """
        Returns a Client connected to a cluster of `num_workers` workers.
        """
        self._server = None

        if launch is None:
            launch = default_comm().size == 1

        # pylint: disable=import-outside-toplevel,
        if not launch:
            # Then the script has been submitted in parallel with mpirun
            num_workers = num_workers or default_comm().size - 2
            if num_workers < 0 or default_comm().size != num_workers + 2:
                raise RuntimeError(
                    f"""
                Error: (num_workers + 2) processes required.
                The script has not been submitted on enough processes.
                Got {default_comm().size} processes instead of {num_workers + 2}.
                """
                )

            initialize(nthreads=threads_per_worker, nanny=False)

            super().__init__()

        else:
            num_workers = num_workers or (multiprocessing.cpu_count() + 1)

            # Since dask-mpi produces several file we create a temporary directory
            self._dir = tempfile.mkdtemp()

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
                _out=out or sys.stdout,
                _err=err or sys.stderr,
            )
            sh.cd(pwd)

            atexit.register(self.close_server)

            super().__init__(scheduler_file=self._dir + "/scheduler.json")

        # Waiting for all the workers to connect
        def handler(signum, frame):
            if self.server is not None:
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

    @property
    def server(self):
        "Returns the running server if available"
        return self._server

    def close_server(self):
        """
        Closes the running server
        """
        if self.server is None:
            raise RuntimeError("No MPI-server started by the client")
        self.shutdown()
        self.close()
        self.server.wait()
        shutil.rmtree(self._dir)
        self._server = None
        atexit.unregister(self.close_server)

    def __del__(self):
        """
        In case of server started, closes the server
        """
        if self.server is not None:
            self.close_server()

        if hasattr(self, "_timeout"):
            super().__del__()

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
            _workers = list(super().who_has(*args, **kwargs).values())
            workers = [w[0] for w in _workers if len(w) == 1]
            assert len(workers) == len(
                _workers
            ), "More than one process has the same reference"
            return workers

        return super().who_has(*args, **kwargs)

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

    @wraps(select_workers)
    def create_comm(self, *args, **kwargs):
        """
        Return a MPI communicator involving workers available by the client.

        Parameters
        ----------
        *args, **kwargs: params
            Following list of parameters for the function select_workers.
        """

        workers = self.select_workers(*args, **kwargs)
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

        return Comm(self.map(_create_comm, ranks))
