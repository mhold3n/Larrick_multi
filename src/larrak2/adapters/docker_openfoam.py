"""Docker-based OpenFOAM runner.

This is a minimal, self-contained port of the Docker OpenFOAM workflow used in campro.
It avoids macOS OpenFOAM/dyld issues and makes DOE runs reproducible.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockerOpenFoamConfig:
    image: str = "openfoam/openfoam11-paraview510"
    platform: str = "linux/amd64"
    bashrc_path: str = "/opt/openfoam11/etc/bashrc"


class DockerOpenFoam:
    def __init__(self, cfg: DockerOpenFoamConfig | None = None):
        self.cfg = cfg or DockerOpenFoamConfig()

    def run_solver(
        self,
        *,
        solver: str,
        case_dir: str | Path,
        args: list[str] | None = None,
        timeout_s: int = 3600,
        log_file: str | Path | None = None,
    ) -> tuple[int, str, str]:
        case_path = Path(case_dir).resolve()
        foam_cmd = f"source {self.cfg.bashrc_path} && cd /case && {solver}"
        if args:
            foam_cmd += " " + " ".join(args)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            self.cfg.platform,
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{case_path}:/case:rw",
            "-w",
            "/case",
            self.cfg.image,
            "-c",
            foam_cmd,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            if log_file is not None:
                Path(log_file).write_text(
                    "=== Command (Timeout) ===\n"
                    + " ".join(cmd)
                    + "\n\n=== stdout ===\n"
                    + (str(e.stdout) if e.stdout is not None else "")
                    + "\n\n=== stderr ===\n"
                    + (str(e.stderr) if e.stderr is not None else "")
                )
            return -1, "", "Timeout expired"
        except FileNotFoundError:
            return -2, "", "Docker not found"

        if log_file is not None:
            Path(log_file).write_text(
                "=== Command ===\n"
                + " ".join(cmd)
                + "\n\n=== stdout ===\n"
                + (str(result.stdout) if result.stdout is not None else "")
                + "\n\n=== stderr ===\n"
                + (str(result.stderr) if result.stderr is not None else "")
            )

        return result.returncode, result.stdout, result.stderr

    def run_utility(
        self,
        *,
        utility: str,
        case_dir: str | Path,
        args: list[str] | None = None,
        timeout_s: int = 300,
        log_file: str | Path | None = None,
    ) -> tuple[int, str, str]:
        return self.run_solver(
            solver=utility,
            case_dir=case_dir,
            args=args,
            timeout_s=timeout_s,
            log_file=log_file,
        )

    def check_availability(self) -> bool:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            self.cfg.platform,
            self.cfg.image,
            "bash",
            "-c",
            f"source {self.cfg.bashrc_path} && echo ready",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0 and "ready" in (result.stdout or "")
        except Exception:
            return False

