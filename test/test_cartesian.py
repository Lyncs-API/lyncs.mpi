from lyncs_mpi import Client, CartComm
from lyncs_mpi.distributed import *
from lyncs_mpi.cartesian import *
from lyncs_mpi.testing import Test, CartTest
from dask.array import Array
from numpy import ndarray


def test_commlocal():
    foo = CartTest(10)
    assert foo.foo == 10
    assert isinstance(foo, CommLocal)
    assert isinstance(foo, Local)
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


def test_cartesian():
    client = Client(2)
    cart = client.create_comm().create_cart((2,))
    assert len(cart) == 2

    foo = CartTest(1, comm=cart)
    assert isinstance(foo, Cartesian)
    assert isinstance(foo, Distributed)
    assert isinstance(foo, CartTest)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert len(foo) == 2
    assert set(foo.workers) == set(cart.workers)
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 1)

    assert isinstance(foo.comm, CartComm)
    arr = foo.ones((2, 2))
    assert arr.shape == (4, 2)
    assert isinstance(arr, Array)
    assert arr.sum() == 8
