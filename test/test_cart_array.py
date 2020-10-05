from pytest import raises
from lyncs_mpi import Client


def test_array():
    client = Client(num_workers=4, launch=True)
    cart = client.comm.create_cart([2, 2, 1])

    for shape in (4, 4), (4, 2, 2), (2, 2, 2, 2):
        arr = cart.ones(shape)
        assert arr.sum() == 16

        futures = cart.get_futures(arr)
        assert len(futures) == len(cart)
        assert cart.workers == tuple(client.who_has(ftr)[0] for ftr in futures)

        arr = cart.array(futures)
        assert arr.sum() == 16
        assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    # Test errors
    with raises(ValueError):
        cart.ones((1, 2))

    with raises(TypeError):
        cart.get_futures(cart)

    with raises(ValueError):
        cart.array(futures[:1])

    cart2 = client.comm.create_cart([2, 1, 2])

    arr = cart2.ones((4, 2, 2))
    with raises(ValueError):
        cart.get_futures(arr)

    arr = cart.ones((4, 4))
    with raises(ValueError):
        cart2.get_futures(arr)

    with raises(ValueError):
        cart2.ones((4, 4))
