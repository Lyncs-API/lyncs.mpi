from pytest import raises
from lyncs_mpi import Client, default_client
from dask.distributed import wait


def test_client():
    client = Client(num_workers=1, launch=True)
    assert client._server is not None
    assert default_client() is client
    assert len(client.workers) == 1

    client.close_server()
    assert client._server is None

def test_comm():
    client = Client(num_workers=4, launch=True)
    comm = client.create_comm()
    assert len(comm) == len(client.workers)
    assert len(comm) == comm.size
    assert set(comm.ranks) == set(range(4))
    assert set(comm.workers) == set(client.workers)
    assert comm[0] == comm[comm.ranks[0]]
    
    with raises(KeyError):
        comm[5]

    cart = comm.create_cart([2,2], periods=True)
    assert cart.dims == (2,2)
    assert all(cart.periods)
    assert cart[0] == cart[cart.ranks_coords[0]]
    assert cart[cart.ranks[0]] == cart[cart.workers_coords[cart.ranks[0]]]
