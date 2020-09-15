from lyncs_mpi import (
    Client,
    default_client,
)


def test_client():
    client = Client(num_workers=1, launch=True)
    assert client._server is not None
    assert default_client() is client
    assert len(client.workers) == 1

    client.close_server()
    assert client._server is None
