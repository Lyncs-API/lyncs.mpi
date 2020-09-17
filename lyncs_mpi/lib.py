"Loading and using MPI via cppyy"

__all__ = [
    "lib",
    "default_comm",
    "initialized",
    "finalized",
]

from ctypes import c_int
from lyncs_cppyy import Lib
from .config import MPI_INCLUDE_DIRS, MPI_LIBRARIES

lib = Lib(
    include=MPI_INCLUDE_DIRS.split(";"),
    header="mpi.h",
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
