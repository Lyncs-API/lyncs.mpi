"Loading and using MPI via cppyy"

__all__ = [
    "lib",
    "default_comm",
    "initialized",
    "finalized",
    "get_comm",
    "MPI",
]

from array import array
from ctypes import c_int
import numpy as np
from lyncs_cppyy import Lib
from lyncs_utils import static_property, lazy_import
from lyncs_cppyy.ll import cast
from . import __path__
from .config import MPI_INCLUDE_DIRS, MPI_LIBRARIES

PATHS = list(__path__)
MPI = lazy_import("mpi4py.MPI")


def get_comm(comm):
    assert isinstance(comm, MPI.Comm)
    return lib.MPI_Comm(MPI._handleof(comm))


class MPILib(Lib):
    def load(self):
        super().load()

        if MPI.get_vendor() != self.get_vendor():
            print("mpi4py vendor:", MPI.get_vendor())
            print("lyncs_mpi vendor:", self.get_vendor())
            raise RuntimeError(
                """
            mpi4py and lyncs_mpi are not using the same MPI
            Try to recompily mpi4py using:
            pip install --force-reinstall --no-cache-dir mpi4py
            """
            )

    def get_vendor(self):
        major = array("i", [0])
        minor = array("i", [0])
        micro = array("i", [0])
        name = self.PyMPI_Get_vendor(major, minor, micro)
        return str(name), (major[0], minor[0], micro[0])


lib = MPILib(
    path=PATHS,
    include=MPI_INCLUDE_DIRS.split(";"),
    header=["mpi.h", "pympivendor.h"],
    library=MPI_LIBRARIES.split(";"),
    c_include=False,
    check="MPI_Init",
)


COMM = None


def default_comm():
    "Returns the default communicator to be used (MPI_COMM_WORLD by default)"
    global COMM
    if not COMM:
        return MPI.COMM_WORLD
    return COMM


def initialized():
    "Whether MPI has been initialized"
    val = c_int(0)
    lib.MPI_Initialized(val)
    return bool(val)


def finalized():
    "Whether MPI has been finalized"
    val = c_int(0)
    lib.MPI_Finalized(val)
    return bool(val)
