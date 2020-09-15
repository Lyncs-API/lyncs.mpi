from lyncs_mpi import Client
from lyncs_mpi.distributed import *
from lyncs_mpi.testing import Test


def test_local():
    foo = Test(10)
    assert foo.foo == 10
    assert isinstance(foo, Local)
    assert isinstance(foo, Test)
    assert len(foo) == 1
    assert foo.client == None
    assert foo.workers == ("localhost",)
    assert foo in foo


def test_distributed():
    client = Client(2)
    init = client.scatter((1, 2))
    assert len(init) == 2
    assert len(set(client.who_has(init))) == 2

    foo = Test(init)
    assert isinstance(foo, Distributed)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert len(foo) == 2
    assert set(foo.workers) == set(client.who_has(init))
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 2)

    foo = Test(foo=init)
    assert isinstance(foo, Distributed)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert set(foo.workers) == set(client.who_has(init))
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten() == 10
    assert foo.values() == (1, 2)