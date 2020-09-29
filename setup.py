from lyncs_setuptools import setup, CMakeExtension

requirements = [
    "mpi4py",
    "lyncs-cppyy",
    "lyncs-utils",
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
    data_files=[(".", ["config.py.in"])],
    install_requires=requirements,
    extras_require={"test": ["pytest", "pytest-cov", "pytest-benchmark"]},
)
