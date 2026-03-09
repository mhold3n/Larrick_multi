from __future__ import annotations

from types import SimpleNamespace

from larrak2.adapters.docker_openfoam import DockerOpenFoam


def test_check_availability_accepts_success_without_ready_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="Welcome to OpenFOAM\n", stderr=""),
    )
    assert DockerOpenFoam().check_availability() is True
