from lyncs_mpi import Client
from lyncs_mpi.parallel import *
from lyncs_mpi.parallel import Test


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
    assert set(foo.workers) == set(client.who_has(foo))

    foo = Test(foo=init)
    assert isinstance(foo, Parallel)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert set(foo.workers) == set(client.who_has(init))
    assert set(foo.workers) == set(client.who_has(foo))
