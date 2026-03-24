"""Docker-based OpenFOAM runner.

This is a minimal, self-contained port of the Docker OpenFOAM workflow used in campro.
It avoids macOS OpenFOAM/dyld issues and makes DOE runs reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockerOpenFoamConfig:
    image: str = "microfluidica/openfoam:2512"
    platform: str = "linux/amd64"
    bashrc_path: str | None = None
    custom_solver_cache_root: str = "outputs/validation_runtime/openfoam_custom_solvers"


class DockerOpenFoam:
    def __init__(self, cfg: DockerOpenFoamConfig | None = None):
        self.cfg = cfg or DockerOpenFoamConfig()

    @staticmethod
    def _sha_tree(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _write_log(log_file: str | Path | None, cmd: list[str], stdout: str, stderr: str) -> None:
        if log_file is None:
            return
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "=== Command ===\n"
            + " ".join(cmd)
            + "\n\n=== stdout ===\n"
            + stdout
            + "\n\n=== stderr ===\n"
            + stderr
        )

    def _image_token(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.cfg.image)).strip("_") or "openfoam"

    def _bashrc_candidates(self) -> list[str]:
        candidates: list[str] = []
        if self.cfg.bashrc_path:
            candidates.append(self.cfg.bashrc_path)

        image_name = str(self.cfg.image).lower()
        version_tokens: list[str] = []
        for pattern in (r"openfoam[:/-]v?(\d{4})", r"openfoam(\d{1,4})"):
            for token in re.findall(pattern, image_name):
                if token not in version_tokens:
                    version_tokens.append(token)

        for token in version_tokens:
            candidates.extend(
                [
                    f"/usr/lib/openfoam/openfoam{token}/etc/bashrc",
                    f"/opt/OpenFOAM/OpenFOAM-v{token}/etc/bashrc",
                    f"/opt/openfoam{token}/etc/bashrc",
                ]
            )

        candidates.extend(
            [
                "/usr/lib/openfoam/openfoam2512/etc/bashrc",
                "/opt/OpenFOAM/OpenFOAM-v2512/etc/bashrc",
                "/usr/lib/openfoam/openfoam2312/etc/bashrc",
                "/opt/OpenFOAM/OpenFOAM-v2312/etc/bashrc",
                "/opt/openfoam11/etc/bashrc",
            ]
        )

        ordered: list[str] = []
        for path in candidates:
            if path and path not in ordered:
                ordered.append(path)
        return ordered

    def _foam_env_setup_cmd(self) -> str:
        candidates = self._bashrc_candidates()
        clauses: list[str] = []
        for idx, path in enumerate(candidates):
            keyword = "if" if idx == 0 else "elif"
            quoted_path = shlex.quote(path)
            clauses.append(f"{keyword} [ -f {quoted_path} ]; then source {quoted_path};")
        clauses.append('else echo "OpenFOAM bashrc not found" >&2; exit 127;')
        clauses.append("fi")
        return " ".join(clauses)

    def _foam_cmd(self, executable: str, args: list[str] | None = None) -> str:
        exec_cmd = shlex.quote(executable)
        if args:
            exec_cmd += " " + " ".join(shlex.quote(arg) for arg in args)
        return f"{self._foam_env_setup_cmd()} && cd /case && {exec_cmd}"

    def _run_docker_script(
        self,
        *,
        script: str,
        mounts: list[tuple[Path, str, str]],
        workdir: str,
        timeout_s: int,
        log_file: str | Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            self.cfg.platform,
            "--entrypoint",
            "/bin/bash",
        ]

        for host_path, container_path, mode in mounts:
            resolved = Path(host_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if resolved.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
            cmd.extend(["-v", f"{resolved}:{container_path}:{mode}"])

        for key, value in sorted((extra_env or {}).items()):
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(
            [
                "-w",
                workdir,
                self.cfg.image,
                "-c",
                script,
            ]
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            stdout = str(e.stdout) if e.stdout is not None else ""
            stderr = str(e.stderr) if e.stderr is not None else "Timeout expired"
            self._write_log(log_file, cmd, stdout, stderr)
            return -1, stdout, stderr
        except FileNotFoundError:
            self._write_log(log_file, cmd, "", "Docker not found")
            return -2, "", "Docker not found"

        self._write_log(
            log_file,
            cmd,
            str(result.stdout) if result.stdout is not None else "",
            str(result.stderr) if result.stderr is not None else "",
        )
        return result.returncode, result.stdout, result.stderr

    def ensure_custom_solver(
        self,
        *,
        source_dir: str | Path,
        solver_name: str = "larrakEngineFoam",
        cache_root: str | Path | None = None,
        refresh: bool = False,
        log_file: str | Path | None = None,
    ) -> dict[str, str]:
        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Custom OpenFOAM solver source not found: {source_path}")

        cache_root_path = Path(cache_root or self.cfg.custom_solver_cache_root).resolve()
        source_hash = self._sha_tree(source_path)
        cache_dir = cache_root_path / f"{solver_name}_{self._image_token()}_{source_hash[:12]}"
        manifest_path = cache_dir / "build_manifest.json"
        binary_path = cache_dir / "bin" / solver_name

        if not refresh and manifest_path.exists() and binary_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_hash") == source_hash
                and manifest.get("solver_name") == solver_name
                and manifest.get("image") == self.cfg.image
            ):
                manifest["binary_path"] = str(binary_path)
                manifest["cache_dir"] = str(cache_dir)
                return manifest

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_dir = cache_dir / "build"
        build_src = build_dir / source_path.name
        bin_dir = cache_dir / "bin"
        shutil.copytree(source_path, build_src)
        bin_dir.mkdir(parents=True, exist_ok=True)

        script = (
            f"{self._foam_env_setup_cmd()} && "
            "mkdir -p /solverCache/bin && "
            "export FOAM_USER_APPBIN=/solverCache/bin && "
            f"cd /solverCache/build/{shlex.quote(source_path.name)} && wmake"
        )

        code, stdout, stderr = self._run_docker_script(
            script=script,
            mounts=[
                (cache_dir, "/solverCache", "rw"),
            ],
            workdir="/solverCache",
            timeout_s=1800,
            log_file=log_file,
        )
        if code != 0:
            raise RuntimeError(
                f"Failed to build custom OpenFOAM solver '{solver_name}' from {source_path}: {stderr or stdout}"
            )
        if not binary_path.exists():
            raise RuntimeError(
                f"Custom OpenFOAM solver build completed without emitting binary '{binary_path}'"
            )

        manifest = {
            "solver_name": solver_name,
            "image": self.cfg.image,
            "platform": self.cfg.platform,
            "source_dir": str(source_path),
            "source_hash": source_hash,
            "cache_dir": str(cache_dir),
            "binary_path": str(binary_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def run_solver(
        self,
        *,
        solver: str,
        case_dir: str | Path,
        args: list[str] | None = None,
        timeout_s: int = 3600,
        log_file: str | Path | None = None,
        custom_solver_dirs: list[str | Path] | None = None,
    ) -> tuple[int, str, str]:
        case_path = Path(case_dir).resolve()
        mounts: list[tuple[Path, str, str]] = [(case_path, "/case", "rw")]
        prepended_paths: list[str] = []
        for index, solver_dir in enumerate(custom_solver_dirs or []):
            container_dir = f"/custom-bin-{index}"
            mounts.append((Path(solver_dir).resolve(), container_dir, "ro"))
            prepended_paths.append(container_dir)

        exec_cmd = shlex.quote(solver)
        if args:
            exec_cmd += " " + " ".join(shlex.quote(arg) for arg in args)
        path_prefix = (
            f"export PATH={':'.join(prepended_paths)}:$PATH && " if prepended_paths else ""
        )
        foam_cmd = f"{self._foam_env_setup_cmd()} && {path_prefix}cd /case && {exec_cmd}"
        return self._run_docker_script(
            script=foam_cmd,
            mounts=mounts,
            workdir="/case",
            timeout_s=timeout_s,
            log_file=log_file,
        )

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
        try:
            code, _, _ = self._run_docker_script(
                script=f"{self._foam_env_setup_cmd()} && command -v blockMesh >/dev/null && echo ready",
                mounts=[],
                workdir="/",
                timeout_s=60,
            )
            return code == 0
        except Exception:
            return False
