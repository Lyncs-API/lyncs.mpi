from pytest import raises
from lyncs_mpi import (
    Client,
    default_client,
    cart_zeros,
    cart_ones,
    get_cart_arrays,
    cart_array,
)
from dask.distributed import wait


def test_array():
    client = Client(num_workers=4, launch=True)
    cart = client.comm.create_cart([2, 2], periods=True)

    arr = cart_zeros(cart, (4, 4), chunks=(2, 2))
    assert arr.sum() == 0

    arrs = get_cart_arrays(cart, arr)
    assert len(arrs) == len(cart)

    arr = cart_array(cart, arrs, (4, 4))
    assert arr.sum() == 0
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    arr = cart_zeros(tuple(cart), (4, 4), chunks=(2, 2))
    assert arr.sum() == 0

    arrs = get_cart_arrays(tuple(cart), arr)
    assert len(arrs) == len(cart)

    arr = cart_array(tuple(cart), arrs, (4, 4))
    assert arr.sum() == 0
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    arr = cart_ones(cart, (2, 4, 4), chunks=(2, 2, 2), dims_axes=(1, 2))
    assert arr.sum() == 32

    arrs = get_cart_arrays(cart, arr, dims_axes=(1, 2))
    assert len(arrs) == len(cart)

    arr = cart_array(cart, arrs, (2, 4, 4), dims_axes=(1, 2))
    assert arr.sum() == 32
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    arr = cart_ones(cart, (4, 4, 2))
    assert arr.sum() == 32

    arrs = get_cart_arrays(cart, arr)
    assert len(arrs) == len(cart)

    arr = cart_array(cart, arrs)
    assert arr.sum() == 32
    assert arr.shape == (4, 4, 2)
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    arr = cart_ones(cart, (2, 4, 4), dims_axes=(1, 2))
    assert arr.sum() == 32

    arrs = get_cart_arrays(cart, arr, dims_axes=(1, 2))
    assert len(arrs) == len(cart)

    arr = cart_ones(cart, (4, 2, 4), dims_axes=(2, 0))
    assert arr.sum() == 32

    arrs = get_cart_arrays(cart, arr, dims_axes=(2, 0))
    assert len(arrs) == len(cart)

    arr = cart_array(cart, arrs, (4, 2, 4), dims_axes=(2, 0))
    assert arr.sum() == 32
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    # Testing errors
    with raises(ValueError):
        arr = cart_zeros(cart, (6, 4), chunks=(2, 2))

    with raises(ValueError):
        arr = cart_zeros(cart, (4, 4, 4), chunks=(2, 2, 2))

    with raises(ValueError):
        arr = cart_zeros(cart, (4, 4), chunks=(2, 2), dims_axes=(0, 1, 2))

    with raises(ValueError):
        arr = cart_zeros(cart, (2, 4, 4), chunks=(2, 2, 2), dims_axes=(0, 1))

    with raises(ValueError):
        arr = cart_zeros(cart, (1, 1))

    with raises(ValueError):
        arr = cart_zeros(cart, (4, 4), dims_axes=(0, 0))

    with raises(ValueError):
        arr = cart_zeros(cart, (4, 4), dims_axes=(3, 4))

    with raises(ValueError):
        arr = cart_array(cart, arrs[:-1], (4, 2, 4))

    # Testing a weird cart
    cart = client.comm.create_cart([2, 1, 2], periods=True)
    arr = cart_ones(cart, (4, 2, 4), chunks=(2, 2, 2))
    assert arr.sum() == 32

    cart = client.comm.create_cart([1, 2, 1, 1, 2, 1], periods=True)
    arr = cart_ones(
        cart, (1, 4, 2, 4, 1), chunks=(1, 2, 2, 2, 1), dims_axes=(-1, 3, -1, -1, 1, -1)
    )
    assert arr.sum() == 32

    # Testing a cart with different dims
    client = Client(num_workers=6, launch=True)
    cart = client.comm.create_cart([3, 2], periods=True)
    arr = cart_ones(cart, (6, 4), chunks=(2, 2))
    assert arr.sum() == 24

    arr = cart_ones(cart, (4, 6), chunks=(2, 2), dims_axes=(1, 0))
    assert arr.sum() == 24

    arrs = get_cart_arrays(cart, arr, dims_axes=(1, 0))
    assert len(arrs) == len(cart)

    arr = cart_array(cart, arrs, (4, 6), dims_axes=(1, 0))
    assert arr.sum() == 24
    assert tuple(arr.dask.keys()) == tuple(val.key for val in arr.dask.values())

    with raises(ValueError):
        arr = cart_ones(cart, (4, 6), chunks=(2, 2))
