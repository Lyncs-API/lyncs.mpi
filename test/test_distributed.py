from pytest import raises
from lyncs_mpi import Client
from lyncs_mpi.distributed import *
from lyncs_mpi.testing import DistributedTest


def test_local():
    test = DistributedTest(10)
    assert test.value == 10
    assert isinstance(test, Local)
    assert isinstance(test, DistributedTest)
    assert len(test) == 1
    assert test.client == None
    assert test.workers == ("localhost",)
    assert test in test
    assert test["localhost"] is test
    assert test.dask is None
    assert test.type is DistributedTest
    assert test.wait() is test

    assert test.ten == 10
    assert test.range(5) == range(5)

    with raises(KeyError):
        test["bar"]

    with raises(ValueError):
        Distributed._remote_call(int, 1)

    with raises(TypeError):
        Distributed._remote_call(1)

    assert Distributed._set_and_return(test, "value", 1) is test
    assert test.value == 1

    assert Distributed._insert_args((), (), 1, 2, 3) == (1, 2, 3)
    assert Distributed._insert_args((0, 2), (1, 3), 2) == (1, 2, 3)
    assert test.value == 1


def test_distributed():
    client = Client(2)
    init = client.scatter((1, 2))
    assert len(init) == 2
    assert len(set(client.who_has(init))) == 2

    test = DistributedTest(init)
    assert isinstance(test, Distributed)
    assert isinstance(test, DistributedTest)
    assert test.client is client
    assert len(test) == 2
    assert set(test.workers) == set(client.who_has(init))
    # assert set(test.workers) == set(client.who_has(test))
    assert test.ten == 10
    assert test.values() == (1, 2)

    test = DistributedTest(value=init)
    assert isinstance(test, Distributed)
    assert isinstance(test, DistributedTest)
    assert test.client is client
    assert set(test.workers) == set(client.who_has(init))
    assert test.ten == 10
    assert "ten" in test._constants
    assert test.values() == (1, 2)

    assert test.range(5) == range(5)

    assert test.value.type == int

    test.value = "bar"
    assert test.value.type == str

    with raises(TypeError):
        test(1).wait()

    assert test[test.workers[0]] is next(iter(test))

    with raises(KeyError):
        test["bar"]
