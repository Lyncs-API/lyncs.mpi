from lyncs_mpi import initialized, finalized


def test_init():
    if not initialized():
        initialize()
    from mpi4py import MPI

    assert initialized() == True
    assert MPI.Is_initialized() == True
    assert finalized() == False
    assert MPI.Is_finalized() == False
