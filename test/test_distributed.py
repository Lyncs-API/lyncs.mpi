from pytest import raises
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
    assert foo["localhost"] is foo
    assert foo.dask is None
    assert foo.type is Test
    assert foo.wait() is foo

    with raises(KeyError):
        foo["bar"]

    with raises(ValueError):
        Distributed._remote_call(int, 1)

    with raises(TypeError):
        Distributed._remote_call(1)

    assert Distributed._set_and_return(foo, "foo", 1) is foo
    assert foo.foo == 1

    assert Distributed._insert_args((), (), 1, 2, 3) == (1, 2, 3)
    assert Distributed._insert_args((0, 2), (1, 3), 2) == (1, 2, 3)
    assert foo.foo == 1


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
    assert foo.ten == 10
    assert foo.values() == (1, 2)

    foo = Test(foo=init)
    assert isinstance(foo, Distributed)
    assert isinstance(foo, Test)
    assert foo.client is client
    assert set(foo.workers) == set(client.who_has(init))
    # assert set(foo.workers) == set(client.who_has(foo))
    assert foo.ten == 10
    assert foo.values() == (1, 2)

    assert foo.foo.type == int

    foo.foo = "bar"
    assert foo.foo.type == str

    with raises(TypeError):
        foo(1).wait()

    assert foo[foo.workers[0]] is next(iter(foo))

    with raises(KeyError):
        foo["bar"]
