"""OpenFOAM Adapter.

Manages execution of OpenFOAM cases for Scavenging analysis.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .docker_openfoam import DockerOpenFoam, DockerOpenFoamConfig


class OpenFoamRunner:
    """Manages OpenFOAM case execution."""

    def __init__(
        self,
        template_dir: str | Path,
        solver_cmd: str = "pisoFoam",
        *,
        backend: str = "docker",
        docker_image: str | None = None,
    ):
        self.template_dir = Path(template_dir)
        self.solver_cmd = solver_cmd
        self.backend = backend
        self._docker = (
            DockerOpenFoam(DockerOpenFoamConfig(image=docker_image))
            if docker_image is not None
            else DockerOpenFoam()
        )

    def setup_case(self, run_dir: Path, params: dict[str, float]):
        """Clone template and update dictionaries."""
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Clone
        shutil.copytree(self.template_dir, run_dir)

        # Replace placeholders across the entire case directory.
        # This allows templates to parameterize arbitrary files under 0/, constant/, system/, etc.
        self._replace_placeholders(run_dir, params)

    def _replace_placeholders(self, run_dir: Path, params: dict[str, float]) -> None:
        """Replace `{{key}}` placeholders in all text files under run_dir."""

        # Pre-compute placeholder map as strings (OpenFOAM dicts are text)
        replacements = {f"{{{{{k}}}}}": str(v) for k, v in params.items()}

        # Walk all files and attempt text replacement.
        # We skip large files to avoid expensive operations.
        for p in run_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue

            try:
                text = p.read_text()
            except Exception:
                continue

            updated = text
            for ph, val in replacements.items():
                if ph in updated:
                    updated = updated.replace(ph, val)

            if updated != text:
                p.write_text(updated)

    def run(self, run_dir: Path, log_name: str = "solver.log", *, timeout_s: int = 300) -> bool:
        """Execute solver in run_dir."""
        log_path = run_dir / log_name

        print(f"Running {self.solver_cmd} in {run_dir}...")

        try:
            if self.backend == "docker":
                code, _, _ = self._docker.run_solver(
                    solver=self.solver_cmd,
                    case_dir=run_dir,
                    timeout_s=timeout_s,
                    log_file=log_path,
                )
                if code != 0:
                    raise subprocess.CalledProcessError(code, [self.solver_cmd])
            else:
                with open(log_path, "w") as log_file:
                    subprocess.run(
                        [self.solver_cmd],
                        cwd=run_dir,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout_s,
                    )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"OpenFOAM execution failed: {e}")
            return False

    def parse_results(self, run_dir: Path, log_name: str = "solver.log") -> dict[str, float]:
        """Parse log file for metrics (Efficiency, Trapped Mass)."""
        log_path = run_dir / log_name
        if not log_path.exists():
            return {}

        text = log_path.read_text()

        metrics = {}

        # Accept scientific notation where relevant
        float_re = r"([0-9]+(?:\.[0-9]+)?(?:[eE][\-\+]?[0-9]+)?)"

        eff_matches = re.findall(rf"Scavenging Efficiency\s*=\s*{float_re}", text)
        if eff_matches:
            metrics["scavenging_efficiency"] = float(eff_matches[-1])

        mass_matches = re.findall(rf"Trapped Mass\s*=\s*{float_re}", text)
        if mass_matches:
            metrics["trapped_mass"] = float(mass_matches[-1])

        resid_matches = re.findall(rf"Residual Fraction\s*=\s*{float_re}", text)
        if resid_matches:
            metrics["residual_fraction"] = float(resid_matches[-1])

        o2_matches = re.findall(rf"Trapped O2 Mass\s*=\s*{float_re}", text)
        if o2_matches:
            metrics["trapped_o2_mass"] = float(o2_matches[-1])

        return metrics

    def execute(self, run_dir: Path, params: dict[str, float]) -> dict[str, float]:
        """Full execution pipeline."""
        self.setup_case(run_dir, params)
        success = self.run(run_dir)
        if not success:
            return {"error": 1.0}
        return self.parse_results(run_dir)
