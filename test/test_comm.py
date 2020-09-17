from pytest import raises
from lyncs_mpi import (
    Client,
    default_client,
)


def test_comm():
    client = Client(num_workers=4, launch=True)
    comm = client.comm
    assert len(comm) == len(client.workers)
    assert len(comm) == comm.size
    assert set(comm.ranks) == set(range(4))
    assert set(comm.workers) == set(client.workers)
    assert comm[0] == comm[comm.ranks_workers[0]]

    with raises(KeyError):
        comm[5]

    cart = comm.create_cart([2, 2], periods=True)
    assert cart.dims == (2, 2)
    assert all(cart.periods)
    assert cart[0] == cart[cart.ranks_coords[0]]

    comm1 = client.create_comm(2)
    comm2 = client.create_comm(2, exclude=comm1.workers)
    assert not set(comm1.workers).intersection(comm2.workers)
    assert set(comm1.workers + comm2.workers) == set(client.workers)
