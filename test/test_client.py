import os
import sys
import sh
import tempfile
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


def test_not_launch():
    with raises(RuntimeError):
        Client(launch=False)

    pwd = os.getcwd()
    sh.cd(tempfile.mkdtemp())
    test = sh.mpirun(
        "-n",
        3,
        sys.executable,
        "-c",
        "from lyncs_mpi import Client; Client(1, launch=False)",
    )
    sh.cd(pwd)
    assert test.exit_code == 0
