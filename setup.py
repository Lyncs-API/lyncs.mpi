import glob
from lyncs_setuptools import setup, CMakeExtension

requirements = [
    "lyncs-setuptools",
    "mpi4py",
    "lyncs-cppyy",
    "lyncs-utils>=0.2.2",
    "dask",
    "distributed",
    "dask[array]",
    "dask[distributed]",
    "dask-mpi",
    "sh>=1.14.0",
]


setup(
    "lyncs_mpi",
    ext_modules=[CMakeExtension("lyncs_mpi.lib", ".")],
    exclude=["*.config"],
    data_files=[
        (".", ["config.py.in"]),
        ("lyncs_mpi", glob.glob("lyncs_mpi/include/*.h")),
    ],
    install_requires=requirements,
    extras_require={"test": ["pytest", "pytest-cov", "pytest-benchmark"]},
    package_data={"lyncs_mpi": ["include/*.h"]},
    include_package_data=True,
)
