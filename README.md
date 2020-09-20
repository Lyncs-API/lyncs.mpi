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

