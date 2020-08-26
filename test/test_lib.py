from lyncs_mpi import initialized, initialize, finalize, finalized


def test_init():
    if not initialized():
        initialize()
    from mpi4py import MPI

    assert initialized() == True
    assert MPI.Is_initialized() == True
    finalize()
    assert finalized() == True
    assert MPI.Is_finalized() == True
