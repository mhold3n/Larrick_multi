from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from larrak2.adapters.docker_openfoam import DockerOpenFoam, DockerOpenFoamConfig


def test_check_availability_accepts_success_without_ready_token(monkeypatch) -> None:
    monkeypatch.setattr(DockerOpenFoam, "resolve_docker_bin", lambda self: "/usr/local/bin/docker")
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


def test_resolve_docker_bin_prefers_explicit_override(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "docker-explicit"
    explicit.write_text("", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    runner = DockerOpenFoam(DockerOpenFoamConfig(docker_bin=str(explicit)))

    assert runner.resolve_docker_bin() == str(explicit.resolve())


def test_resolve_docker_bin_uses_environment_override(tmp_path: Path, monkeypatch) -> None:
    env_bin = tmp_path / "docker-env"
    env_bin.write_text("", encoding="utf-8")
    monkeypatch.delenv("LARRAK_DOCKER_BIN", raising=False)
    monkeypatch.setenv("LARRAK_DOCKER_BIN", str(env_bin))
    monkeypatch.setattr("shutil.which", lambda _name: None)

    runner = DockerOpenFoam()

    assert runner.resolve_docker_bin() == str(env_bin.resolve())


def test_resolve_docker_bin_falls_back_to_usr_local(monkeypatch) -> None:
    original_exists = Path.exists
    monkeypatch.delenv("LARRAK_DOCKER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: str(self) == "/usr/local/bin/docker" or original_exists(self),
    )

    runner = DockerOpenFoam()

    assert runner.resolve_docker_bin() in {
        "/usr/local/bin/docker",
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    }


def test_resolve_docker_bin_falls_back_to_docker_desktop_app_path(monkeypatch) -> None:
    original_exists = Path.exists
    monkeypatch.delenv("LARRAK_DOCKER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: (
            str(self) == "/Applications/Docker.app/Contents/Resources/bin/docker"
            or original_exists(self)
        ),
    )

    runner = DockerOpenFoam()

    assert runner.resolve_docker_bin() == "/Applications/Docker.app/Contents/Resources/bin/docker"


def test_docker_preflight_reports_missing_cli_candidates(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LARRAK_DOCKER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    log_file = tmp_path / "docker_preflight.log"
    details = DockerOpenFoam().docker_preflight(log_file=log_file)

    assert details["ok"] is False
    assert details["failure_class"] == "docker_cli_missing"
    assert "/usr/local/bin/docker" in details["candidate_paths"]
    assert "Docker CLI not found" in details["message"]
    assert "docker_cli_missing" in log_file.read_text(encoding="utf-8")


def test_docker_preflight_autostarts_docker_desktop_when_daemon_is_down(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = DockerOpenFoam()
    monkeypatch.setattr(runner, "resolve_docker_bin", lambda: "/usr/local/bin/docker")
    monkeypatch.setattr(runner, "_can_autostart_docker_desktop", lambda: True)

    info_calls = iter(
        [
            (1, "", "Cannot connect to the Docker daemon at unix:///tmp/docker.sock. Is the docker daemon running?"),
            (0, "Server Version: 29.2.1", ""),
        ]
    )

    monkeypatch.setattr(runner, "_run_docker_info_command", lambda **kwargs: next(info_calls))
    monkeypatch.setattr(runner, "_open_docker_desktop", lambda **kwargs: (0, "", ""))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    log_file = tmp_path / "docker_preflight.log"
    details = runner.docker_preflight(log_file=log_file)

    assert details["ok"] is True
    assert details["docker_autostart_attempted"] is True
    assert details["docker_autostart_succeeded"] is True
    assert "docker_autostart_attempted" in log_file.read_text(encoding="utf-8")


def test_docker_preflight_reports_launch_failed_when_autostart_does_not_recover(
    monkeypatch,
) -> None:
    runner = DockerOpenFoam()
    monkeypatch.setattr(runner, "resolve_docker_bin", lambda: "/usr/local/bin/docker")
    monkeypatch.setattr(runner, "_can_autostart_docker_desktop", lambda: True)
    monkeypatch.setattr(
        runner,
        "_run_docker_info_command",
        lambda **kwargs: (
            1,
            "",
            "Cannot connect to the Docker daemon at unix:///tmp/docker.sock. Is the docker daemon running?",
        ),
    )
    monkeypatch.setattr(runner, "_open_docker_desktop", lambda **kwargs: (0, "", ""))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    details = runner.docker_preflight(timeout_s=1)

    assert details["ok"] is False
    assert details["failure_class"] == "docker_launch_failed"
    assert details["docker_autostart_attempted"] is True
    assert details["docker_autostart_succeeded"] is False
