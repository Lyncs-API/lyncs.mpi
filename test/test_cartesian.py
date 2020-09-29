from pytest import raises
from lyncs_mpi import Client, CartComm
from lyncs_mpi.distributed import *
from lyncs_mpi.cartesian import *
from lyncs_mpi.testing import DistributedTest, CartesianTest
from dask.array import Array
from numpy import ndarray


def test_commlocal():
    test = CartesianTest((4, 2), value=10)
    assert test.value == 10
    assert isinstance(test, CommLocal)
    assert isinstance(test, Local)
    assert isinstance(test, CartesianTest)
    assert isinstance(test, DistributedTest)
    assert len(test) == 1
    assert test.client == None
    assert test.workers == ("localhost",)
    assert test in test

    assert test.comm is None
    assert test.coords == ((0,),)
    assert test.procs == (1,)
    assert test.ranks == (0,)

    assert test[0] == test["localhost"]
    assert test[0] == test[0, 0, 0, 0, 0]

    arr = test.ones()
    assert arr.shape == (4, 2)
    assert isinstance(arr, ndarray)
    assert arr.sum() == 8


def test_cartesian():
    client = Client(2)
    cart = client.create_comm().create_cart((2,))
    assert len(cart) == 2

    test = CartesianTest((2, 2), value=1, comm=cart)
    assert isinstance(test, Cartesian)
    assert isinstance(test, Distributed)
    assert isinstance(test, CartesianTest)
    assert isinstance(test, DistributedTest)
    assert test.client is client
    assert len(test) == 2
    assert set(test.workers) == set(cart.workers)

    assert test.ten == 10
    assert test.values() == (1, 1)

    assert test.comm is cart
    assert test.coords == ((0,), (1,))
    assert test.procs == (2,)
    assert test.ranks == (0, 1)

    assert test[0] == test[0, 0]
    assert test[1] == test[1, 0]

    assert isinstance(test.comm, CartComm)
    arr = test.ones()
    assert arr.shape == (4, 2)
    assert isinstance(arr, Array)
    assert arr.sum() == 8

    test.value = 10
    assert (test.mul_by_value(arr) == arr * 10).all()

    with raises(ValueError):
        cart2 = client.create_comm(1).create_cart((1,))
        test = CartesianTest(cart2, comm=cart)

    with raises(ValueError):
        init = client.scatter((1, 2))
        test = CartesianTest(init)
