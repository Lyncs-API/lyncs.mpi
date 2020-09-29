from dask.base import wait
from lyncs_mpi import Client
from lyncs_mpi.testing import CartesianTest

client = Client(2)
comm = client.create_comm().create_cart((2,))


def test_bench_init(benchmark):
    benchmark(CartesianTest, (4, 4), comm=comm)


def test_bench_result(benchmark):
    test = CartesianTest((4, 4), comm=comm)
    benchmark(test.range, 10)


def test_bench_array(benchmark):
    test = CartesianTest((4, 4), comm=comm)
    benchmark(test.ones)


def test_bench_array_wait(benchmark):
    test = CartesianTest((4, 4), comm=comm)
    benchmark(lambda: wait(test.ones()))
