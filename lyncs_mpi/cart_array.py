"Cartesian arrays, arrays defined over a cartesian communicator"

__all__ = [
    "CartComm",
]

from math import ceil
from dask.core import flatten
from dask.array import Array, zeros, ones, empty, full
from dask.array.core import normalize_chunks
from .comm import CartComm

add_to = lambda cls: lambda mth: setattr(cls, mth.__name__, mth)


class KeyPatch(tuple):
    "This is a patch for accepting a tuple as a dask key"

    def __dask_graph__(self):
        return {self: None}

    def __dask_keys__(self):
        return [self]


@add_to(CartComm)
def check_dims(self, nchunks):
    "Checks the number of chunks of an array versus the communicator dimensions"
    dims = self.dims
    if len(dims) < len(nchunks):
        dims += (1,) * (len(nchunks) - len(dims))
    elif len(dims) > len(nchunks):
        if not all((dim == 1 for dim in dims[len(nchunks) :])):
            raise ValueError("Array shape smaller than distributed axes")
        dims = dims[: len(nchunks)]
    if nchunks != dims:
        raise ValueError(
            f"""Number of chunks not compatible with comm dims.
        Got nchunks = {nchunks}, while expected {dims}."""
        )


@add_to(CartComm)
def get_futures(self, array):
    """
    Return the list of futures of an array associated to a cartesian communicator.

    Parameters
    ----------
    array: Dask Array
        A dask array distributed as the cartesian communicator. Note: the array can
        have more axes than the communicator as long as they are not distributed.
    """
    if not isinstance(array, Array):
        raise TypeError(f"Expected a Dask Array; got {type(array)}.")

    self.check_dims(tuple(len(chunks) for chunks in array.chunks))

    idxs, _ = zip(*self.normalize_dims())
    coords = self.normalize_coords()
    keys = tuple(flatten(array.__dask_keys__()))
    key_idx = {}
    for key in keys:
        coord = tuple(key[_i + 1] for _i in idxs)
        key_idx[key] = coords.index(coord)

    keys = sorted(keys, key=key_idx.__getitem__)
    restrictions = {KeyPatch(key): worker for key, worker in zip(keys, self.workers)}

    array = array.persist(workers=restrictions)
    assert len(self) == len(array.dask.values())

    return list(array.dask[key] for key in keys)


@add_to(CartComm)
def array(self, futures, shape=None, chunks=None, dtype=None):
    """
    Turns a set of future arrays (result of a distributed operation),
    associated to a cartesian communicator, into a Dask Array.

    Parameters
    ----------
    cart: CartComm
        A cartesian communicator with dimensions equal to the number of chunks
    futures: tuple(futures)
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
    if not len(futures) == len(self):
        raise ValueError("futures and cart must have the same length")

    if chunks is None or dtype is None:
        infos = self.client.map(lambda arr: (arr.dtype, arr.shape), futures)
        infos = tuple(_.result() for _ in infos)

        if dtype is None:
            dtype = infos[0][0]
        if not all((dtype == dtp for (dtp, _) in infos)):
            raise TypeError(
                f"Futures have different dtypes {[info[0] for info in infos]}"
            )

        if chunks is None:
            chunks = infos[0][1]
            if not all((chunks == chn for (_, chn) in infos)):
                # TODO: normalize chunks using shape
                raise NotImplementedError(
                    "Futures with non-uniform chunks not supported yet"
                )

    if shape is None:
        shape = list(chunks)
        for _i, _l in self.normalize_dims():
            shape[_i] *= _l

    chunks = normalize_chunks(chunks, shape, dtype=dtype)

    self.check_dims(tuple(len(chunk) for chunk in chunks))

    dask = {}
    idxs, _ = zip(*self.normalize_dims())
    for coords, future in zip(self.normalize_coords(), futures):
        key = [0] * len(shape)
        for _i, _c in zip(idxs, coords):
            key[_i] = _c

        name = next(iter(futures)).key
        if isinstance(name, tuple):
            name = name[0]
        assert isinstance(name, str)
        key = (name,) + tuple(key)
        dask[key] = future

    return Array(dask, next(iter(dask.keys()))[0], chunks, dtype=dtype, shape=shape)


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

    def wrapper(self, shape, **kwargs):
        chunks = list(shape)
        for _i, _l in self.normalize_dims():
            if _i >= len(shape):
                raise ValueError(f"Array shape smaller than distributed axes")
            if _l > shape[_i]:
                raise ValueError(f"Cannot chunk shape on axis {_i} in {_l} pieces")
            chunks[_i] = ceil(chunks[_i] / _l)

        array = mth(shape, chunks=chunks, **kwargs)

        idxs, _ = zip(*self.normalize_dims())
        coords = self.normalize_coords()
        workers = self.workers
        keys = flatten(array.__dask_keys__())
        restrictions = {}
        for key in keys:
            coord = tuple(key[_i + 1] for _i in idxs)
            restrictions[KeyPatch(key)] = workers[coords.index(coord)]

        return array.persist(workers=restrictions)

    wrapper.__name__ = "cart_" + mth.__name__
    wrapper.__doc__ = array_wrapper.__doc__ % mth.__name__ + mth.__doc__
    return wrapper


CartComm.zeros = array_wrapper(zeros)
CartComm.ones = array_wrapper(ones)
CartComm.empty = array_wrapper(empty)
CartComm.full = array_wrapper(full)
