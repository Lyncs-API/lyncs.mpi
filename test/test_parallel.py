from lyncs_mpi import Client
from lyncs_mpi.dask_mpi import Cartcomm
from lyncs_mpi.parallel import *
from lyncs_mpi.parallel import Test, CartTest
from dask.array import Array
from numpy import ndarray


def test_serial():
    foo = Test(10)
    assert foo.foo == 10
    assert isinstance(foo, Serial)
    assert isinstance(foo, Test)
    assert len(foo) == 1
    assert foo.client == None
    assert foo.workers == ("localhost",)
    assert foo in foo


def test_parallel():
    client = Client(2)
    init = client.scatter((1, 2))
    assert len(init) == 2
    assert len(set(client.who_has(init))) == 2

    foo = Test(init)
    assert isinstance(foo, Parallel)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert len(foo) == 2
    assert set(foo.workers) == set(client.who_has(init))
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 2)

    foo = Test(foo=init)
    assert isinstance(foo, Parallel)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert set(foo.workers) == set(client.who_has(init))
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 2)


def test_cartserial():
    foo = CartTest(10)
    assert foo.foo == 10
    assert isinstance(foo, CartSerial)
    assert isinstance(foo, Serial)
    assert isinstance(foo, CartTest)
    assert isinstance(foo, Test)
    assert len(foo) == 1
    assert foo.client == None
    assert foo.workers == ("localhost",)
    assert foo in foo

    arr = foo.ones((4, 2))
    assert arr.shape == (4, 2)
    assert isinstance(arr, ndarray)
    assert arr.sum() == 8


def test_cartparallel():
    client = Client(2)
    cart = client.create_comm().create_cart((2,))
    assert len(cart) == 2

    foo = CartTest(1, comm=cart)
    assert isinstance(foo, CartParallel)
    assert isinstance(foo, Parallel)
    assert isinstance(foo, CartTest)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert len(foo) == 2
    assert set(foo.workers) == set(cart.workers)
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 1)

    assert isinstance(foo.comm, Cartcomm)
    arr = foo.ones((2, 2))
    assert arr.shape == (4, 2)
    assert isinstance(arr, Array)
    assert arr.sum() == 8
