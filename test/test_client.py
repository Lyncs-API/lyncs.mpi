from pytest import raises
from lyncs_mpi import (
    Client,
    default_client,
)


def test_client():
    client = Client(num_workers=1)
    assert client._server is not None
    assert default_client() is client
    assert len(client.workers) == 1

    with raises(RuntimeError):
        client.select_workers(2)

    with raises(RuntimeError):
        client.select_workers(1, exclude=client.workers)

    with raises(NotImplementedError):
        client.select_workers(1, resources="GPU")

    client.close_server()
    assert client._server is None

    with raises(RuntimeError):
        client.close_server()

    client = Client(num_workers=1)
    client.__del__()
