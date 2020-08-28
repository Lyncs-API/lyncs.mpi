from lyncs_setuptools import setup, CMakeExtension

requirements = [
    "mpi4py",
    "lyncs-cppyy",
    "dask",
    "distributed",
    "dask[array]",
    "dask[distributed]",
    "dask-mpi",
    "sh",
]


setup(
    "lyncs_mpi",
    exclude=["*.config"],
    ext_modules=[CMakeExtension("lyncs_mpi.lib", ".")],
    data_files=[(".", ["config.py.in"])],
    install_requires=requirements,
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
        ]
    },
    keywords=[
        "Lyncs",
        "MPI",
        "mpi4py",
        "dask-mpi",
    ],
)
