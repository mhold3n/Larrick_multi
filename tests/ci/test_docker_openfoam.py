from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from larrak2.adapters.docker_openfoam import DockerOpenFoam, DockerOpenFoamConfig


def test_check_availability_accepts_success_without_ready_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Welcome to OpenFOAM\n", stderr=""
        ),
    )
    assert DockerOpenFoam().check_availability() is True


def test_default_config_targets_latest_supported_microfluidica_image() -> None:
    runner = DockerOpenFoam()
    assert runner.cfg.image == "microfluidica/openfoam:2512"
    assert "/usr/lib/openfoam/openfoam2512/etc/bashrc" in runner._bashrc_candidates()


def test_openfoam11_image_still_resolves_legacy_bashrc_path() -> None:
    runner = DockerOpenFoam(DockerOpenFoamConfig(image="openfoam/openfoam11-paraview510"))
    assert "/opt/openfoam11/etc/bashrc" in runner._bashrc_candidates()


def test_bashrc_setup_command_emits_valid_if_elif_shell_structure() -> None:
    runner = DockerOpenFoam()
    cmd = runner._foam_env_setup_cmd()
    assert "then source" in cmd
    assert "; elif " in cmd
    assert cmd.endswith("fi")


def test_run_solver_prepends_custom_solver_path(monkeypatch, tmp_path: Path) -> None:
    runner = DockerOpenFoam()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    custom_dir = tmp_path / "custom-bin"
    custom_dir.mkdir()
    observed: dict[str, object] = {}

    def _fake_run_docker_script(**kwargs):
        observed.update(kwargs)
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_docker_script", _fake_run_docker_script)

    code, _, _ = runner.run_solver(
        solver="larrakEngineFoam",
        case_dir=case_dir,
        custom_solver_dirs=[custom_dir],
    )

    assert code == 0
    assert "/custom-bin-0" in str(observed["script"])
    assert any(
        mount[1] == "/custom-bin-0"
        for mount in observed["mounts"]  # type: ignore[index]
    )


def test_ensure_custom_solver_builds_and_reuses_cache(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "larrakEngineFoam"
    (source_dir / "Make").mkdir(parents=True)
    (source_dir / "Make" / "files").write_text(
        "larrakEngineFoam.C\n\nEXE = $(FOAM_USER_APPBIN)/larrakEngineFoam\n",
        encoding="utf-8",
    )
    (source_dir / "Make" / "options").write_text("", encoding="utf-8")
    (source_dir / "larrakEngineFoam.C").write_text("int main() { return 0; }\n", encoding="utf-8")

    cache_root = tmp_path / "cache"
    runner = DockerOpenFoam(DockerOpenFoamConfig(custom_solver_cache_root=str(cache_root)))
    calls: list[dict[str, object]] = []

    def _fake_run_docker_script(**kwargs):
        calls.append(kwargs)
        for host_path, container_path, _mode in kwargs["mounts"]:
            if container_path == "/solverCache":
                (Path(host_path) / "bin").mkdir(parents=True, exist_ok=True)
                ((Path(host_path) / "bin") / "larrakEngineFoam").write_text(
                    "binary\n", encoding="utf-8"
                )
        return 0, "built", ""

    monkeypatch.setattr(runner, "_run_docker_script", _fake_run_docker_script)

    first = runner.ensure_custom_solver(source_dir=source_dir, solver_name="larrakEngineFoam")
    second = runner.ensure_custom_solver(source_dir=source_dir, solver_name="larrakEngineFoam")

    assert Path(first["binary_path"]).exists()
    assert second["source_hash"] == first["source_hash"]
    assert len(calls) == 1
