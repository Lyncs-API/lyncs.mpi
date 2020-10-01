# Utils for interfacing to MPI libraries using mpi4py and dask

[![python](https://img.shields.io/pypi/pyversions/lyncs_mpi.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_mpi/)
[![pypi](https://img.shields.io/pypi/v/lyncs_mpi.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_mpi/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.mpi?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.mpi/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.mpi/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.mpi/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.mpi?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.mpi)
[![pylint](https://img.shields.io/badge/pylint%20score-9.6%2F10-green?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)

This package provides tools for interfacing to MPI libraries based on `mpi4py` and `dask`:

- [mpi4py] is a complete Python API of the MPI standard.

- [dask] is a flexible library for parallel computing in Python.
  Particurally, we use the following sub-modules of the latter:
  
  - [dask.distributed] for distributed computing,
  
  - [dask.array] for distributed numpy arrays and
  
  - [dask_mpi] for interfacing to MPI.

[mpi4py]: https://mpi4py.readthedocs.org/

[dask]: https://docs.dask.org/

[dask.distributed]: https://distributed.dask.org/

[dask.array]: https://docs.dask.org/en/latest/array.html

[dask_mpi]: http://mpi.dask.org/

## Installation

**NOTE**: lyncs_mpi requires a working MPI installation.
This can be installed via `apt-get`:

```
sudo apt-get install libopenmpi-dev openmpi-bin
```

OR using `conda`:

```
conda install -c anaconda mpi4py
```

The package can be installed via `pip`:

```
pip install [--user] lyncs_mpi
```

## Documentation


In this package we implement several low-level tools for supporting classes distributed over MPI.
These are described in this [guide]() for developers. In the following we describe the high-level tools
provided in this package.

### Client

The Client is a wrapper of `dask.distributed.Client` made MPI compatible following the instructions
of [dask-mpi](http://mpi.dask.org) documentation.

```python
from lyncs_mpi import Client

client = Client(num_workers=4)
```

If the above script is run in a interactive shell, the Client will start an MPI server in the background
running over `num_workers+1` processes. The workers are the effective processes involved in the calculation.
The extra process (+1) is the scheduler that will manage the task scheduling.

The client, the interactive shell in this example, will proceed processing the script: submitting tasks to
the scheduler that will run them on the workers.

The same script can be run directly via `mpirun`. In this case one needs to execute

```bash
mpirun -n $((num_workers + 2)) python script.py
```

that will run on `num_workers+2` processes (as above +1 for the scheduler and +1 for the client that processes the script).

### Communicators

Another feature that make `lyncs_mpi.Client` MPI compatible is the support of MPI communicators.

```python
comm = client.comm
comm1 = client.create_comm(num_workers=2)
comm2 = client.create_comm(exclude=comm1.workers)
```

In the example, `comm = client.comm` is the main MPI communicator involving all the workers.
The second `comm1` and third `comm2` communicators, instead, are communicators over 2 workers each.
The first two workers have been optimally chosen by the client, the other two instead are the remaining
one excluding the workers of `comm1`.

Another kind of communicators are Cartesian MPI communicators. They can be initialized doing

```python
cart = comm.create_cart([2,2])
```

where `[2,2]` are the dimensions of the multi-dimensional grid where the processes are distributed.

Cartesian communicators directly support [Dask arrays](https://docs.dask.org/en/latest/array.html)
and e.g. `cart.zeros([4,4,3,2,1])` instantiates a distributed Dask array assigned to the workers
of the communicator with local shape (chunks) `(2,2,3,2,1)`.