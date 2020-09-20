"Cartesian arrays, arrays defined over a cartesian communicator"

__all__ = [
    "cart_zeros",
    "cart_ones",
    "cart_empty",
    "cart_full",
    "cart_array",
    "get_cart_arrays",
]

from math import ceil
from dask.array import Array, zeros, ones, empty, full
from dask.array.core import normalize_chunks
from dask.distributed import wait as wait_for
from .comm import CartComm


def get_cart_arrays(cart, array, dims_axes=None, wait=True):
    """
    Return the list of futures of an array associated to a cartesian communicator.

    Parameters
    ----------
    cart: CartComm
        A cartesian communicator with dimensions equal to the number of chunks
    array: Dask Array
        A dask array distributed on the cartesian communicator
    dims_axes: tuple
        The axes associated to the dimensions of the cart
    wait: bool
        Whether to wait for the computation of the array to finish
        for checking the actual list of workers.
        Disable only if you are sure of the location of the array.
    """
    if not isinstance(cart, CartComm):
        cart = CartComm(cart)

    # As first checking if it is compatible and getting the axes
    cart_axes, arr_axes = get_axes(cart.dims, dims_axes, array.shape, array.chunks)

    array = array.persist()
    assert len(cart) == len(array.dask.values())
    if wait:
        wait_for(array.dask.values())
        workers = cart.client.who_has(array.dask.values())
        if not set(workers) == set(cart.workers):
            raise RuntimeError("Workers of cart and array do not match")

    # Creating dict of coords -> worker
    workers = {}
    for coords, worker in zip(cart.coords, cart.workers):
        workers[tuple(coords[_i] for _i in cart_axes)] = worker

    # Creating arrays list
    arrays = [None] * len(cart)
    for key, val in array.dask.items():
        assert len(key) == len(array.shape) + 1
        coords = tuple(key[_i + 1] for _i in arr_axes)
        worker = workers[coords]
        if wait:
            real_worker = cart.client.who_has(val)
            if worker not in real_worker:
                raise RuntimeError("Expected worker {worker} but got {real_worker}")
        arrays[cart.workers.index(worker)] = val

    assert None not in arrays
    return arrays


def cart_array(cart, arrays, shape=None, dims_axes=None, chunks=None, dtype=None):
    """
    Turns a set of future arrays (result of a distributed operation),
    associated to a cartesian communicator, into a Dask Array.

    Parameters
    ----------
    cart: CartComm
        A cartesian communicator with dimensions equal to the number of chunks
    arrays: tuple(futures)
        A set of future arrays associated to the cart
    dims_axes: tuple
        The axes associated to the dimensions of the cart
    shape: tuple(int)
        The shape of the array
    chunks: tuple(int)
        The chunks of the array
    dtype: tuple(int)
        The dtype of the array
    """
    if not isinstance(cart, CartComm):
        cart = CartComm(cart)

    if not len(arrays) == len(cart):
        raise ValueError("arrays and cart must have the same length")

    if chunks is None or dtype is None:
        infos = cart.client.map(lambda arr: (arr.dtype, arr.shape), arrays)
        infos = tuple(_.result() for _ in infos)

        if dtype is None:
            dtype = infos[0][0]
        if not all((dtype == dtp for (dtp, _) in infos)):
            raise TypeError(
                f"Arrays have different dtypes {[info[0] for info in infos]}"
            )

        if chunks is None:
            chunks = infos[0][1]
            if not all((chunks == chn for (_, chn) in infos)):
                # TODO: normalize chunks using shape
                raise NotImplementedError(
                    "Arrays with non-uniform chunks not supported yet"
                )

    if shape is None:
        shape = list(chunks)
        for (_, _l), _i in zip(*normalize_cart_dims(cart.dims, dims_axes, len(chunks))):
            shape[_i] *= _l

    chunks = normalize_chunks(chunks, shape, dtype=dtype)

    cart_axes, arr_axes = get_axes(cart.dims, dims_axes, shape, chunks)

    dask = {}
    for coords, array in zip(cart.coords, arrays):
        key = [0] * len(shape)
        coords = tuple(coords[_i] for _i in cart_axes)

        for _i, _j in zip(coords, arr_axes):
            key[_j] = _i

        name = next(iter(arrays)).key
        if isinstance(name, tuple):
            name = name[0]
        assert isinstance(name, str)
        key = (name,) + tuple(key)
        dask[key] = array

    return Array(dask, next(iter(dask.keys()))[0], chunks, dtype=dtype, shape=shape)


def normalize_cart_dims(dims, dims_axes=None, shape_size=None):
    "Auxiliary function that removes non-distributed dimensions from dims and dims_axes"

    cart_dims = tuple((_i, _l) for _i, _l in enumerate(dims) if _l > 1)

    if dims_axes is None:
        dims_axes = tuple(_i for _i, _l in cart_dims)
    else:
        if not len(dims_axes) == len(dims):
            raise ValueError(
                f"""
                Lengths of dims_axes {dims_axes} and dims {dims} do not match
                """
            )

        dims_axes = tuple(dims_axes[_i] for _i, _l in cart_dims)

    if not len(set(dims_axes)) == len(dims_axes):
        raise ValueError(f"Repeated values in dims_axes {dims_axes}")
    if not set(dims_axes) <= set(range(shape_size)):
        raise ValueError(
            f"Values in dims_axes {dims_axes} out of shape range {shape_size}"
        )

    return cart_dims, dims_axes


def get_axes(dims, dims_axes, shape, chunks):
    "Auxiliary function that matches the dims_axes with the chunks"

    cart_dims, dims_axes = normalize_cart_dims(dims, dims_axes, len(shape))
    cdims = tuple(_l for _i, _l in cart_dims)

    # Extracting the chunked axes and cart dims
    arr_dims = tuple((_i, _l) for _i, _l in enumerate(map(len, chunks)) if _l > 1)
    adims = tuple(_l for _i, _l in arr_dims)

    if sorted(adims) != sorted(cdims):
        raise ValueError(
            f"""
            Number of chunks {adims} different from cart dims {cdims}.
            """
        )

    # Reordering cart.dims with dims_axes
    if dims_axes is not None:
        aaxes = tuple(_i for _i, _l in arr_dims)
        if not sorted(dims_axes) == sorted(aaxes):
            raise ValueError(
                f"""
                Cart axes {sorted(aaxes)} different from chunked axes {sorted(dims_axes)}.
                """
            )
        cart_dims = tuple(cart_dims[dims_axes.index(_i)] for _i in aaxes)
        cdims = tuple(_l for _i, _l in cart_dims)

    if adims != cdims:
        raise ValueError(
            f"""
            Number of chunks {adims} and cart dims {cdims} do not match.
            You can use dims_axes to reorder them.
            """
        )
    return tuple(_i for _i, _l in cart_dims), tuple(_i for _i, _l in arr_dims)


def array_wrapper(mth):
    """
    Cartesian variant of %s (documentation follows).
    Assigns to the workers of a cartesian communicator chunks of the array.

    The first argument `cart` should be a cartesian communicator
    with dimensions equal to the number of chunks of the array.

    Additional option is `dims_axes` that can be used to reorder
    the dimensions of the communicator to match the chunked axes.

    --------- END ----------
    """

    def wrapper(cart, shape, dims_axes=None, chunks=None, **kwargs):
        if not isinstance(cart, CartComm):
            cart = CartComm(cart)

        if chunks is None:
            chunks = list(shape)
            loop_over = enumerate(cart.dims)
            if dims_axes:
                cart_dims, cart_axes = normalize_cart_dims(
                    cart.dims, dims_axes, len(shape)
                )
                cdims = tuple(_l for _i, _l in cart_dims)
                loop_over = zip(cart_axes, cdims)
            for _i, _l in loop_over:
                if _l > shape[_i]:
                    raise ValueError(f"Cannot chunk shape on axis {_i} in {_l} pieces")
                chunks[_i] = ceil(chunks[_i] / _l)

        arr = mth(shape, chunks=chunks, **kwargs)

        cart_axes, arr_axes = get_axes(cart.dims, dims_axes, arr.shape, arr.chunks)

        # Creating dict of coords -> worker
        workers = {}
        for coords, worker in zip(cart.coords, cart.workers):
            workers[tuple(coords[_i] for _i in cart_axes)] = worker

        # Creating dask dict
        dask = {}
        for key, val in arr.dask.items():
            coords = tuple(key[_i + 1] for _i in arr_axes)
            dask[key] = cart.client.submit(*val, workers=[workers[coords]], pure=False)

        return Array(
            dask,
            next(iter(dask.keys()))[0],
            arr.chunks,
            dtype=arr.dtype,
            meta=arr._meta,
            shape=arr.shape,
        )

    wrapper.__name__ = "cart_" + mth.__name__
    wrapper.__doc__ = array_wrapper.__doc__ % mth.__name__ + mth.__doc__
    return wrapper


cart_zeros = array_wrapper(zeros)
cart_ones = array_wrapper(ones)
cart_empty = array_wrapper(empty)
cart_full = array_wrapper(full)
CartComm.zeros = cart_zeros
CartComm.ones = cart_ones
CartComm.empty = cart_empty
CartComm.full = cart_full
CartComm.array = cart_array
