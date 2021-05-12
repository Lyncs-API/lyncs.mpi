"Loading and using MPI via cppyy"

__all__ = [
    "lib",
    "default_comm",
    "initialized",
    "finalized",
]

from ctypes import c_int
from lyncs_cppyy import Lib
from array import array
import numpy as np
from . import __path__
from .config import MPI_INCLUDE_DIRS, MPI_LIBRARIES

PATHS = list(__path__)


class MPILib(Lib):
    def load(self):
        super().load()

        from mpi4py import MPI

        if MPI.get_vendor() != self.get_vendor():
            print("mpi4py vendor:", MPI.get_vendor())
            print("lyncs_mpi vendor:", self.get_vendor())
            raise RuntimeError(
                "mpi4py and lyncs_mpi have not been compiled with the same MPI"
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
    # pylint: disable=import-outside-toplevel,no-name-in-module,redefined-outer-name,global-statement
    global COMM
    if not COMM:
        from mpi4py.MPI import COMM_WORLD as COMM

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
