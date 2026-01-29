"""OpenFOAM Adapter.

Manages execution of OpenFOAM cases for Scavenging analysis.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class OpenFoamRunner:
    """Manages OpenFOAM case execution."""

    def __init__(self, template_dir: str | Path, solver_cmd: str = "pisoFoam"):
        self.template_dir = Path(template_dir)
        self.solver_cmd = solver_cmd

    def setup_case(self, run_dir: Path, params: dict[str, float]):
        """Clone template and update dictionaries."""
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Clone
        shutil.copytree(self.template_dir, run_dir)

        # Update dictionaries
        # We assume specific file locations for parameters
        # e.g. constant/pistonMeshDict
        self._update_piston_mesh(run_dir, params)
        # self._update_boundary_conditions(run_dir, params)

    def _update_piston_mesh(self, run_dir: Path, params: dict[str, float]):
        """Update mesh generation parameters."""
        # This is highly specific to the case structure.
        # We implement a simple sed-like replacement for now.

        target = run_dir / "constant" / "pistonMeshDict"
        if not target.exists():
            return

        content = target.read_text()

        # Replace keys like {{compression_ratio}}
        # Params expected: compression_ratio, expansion_ratio, etc.
        for k, v in params.items():
            placeholder = f"{{{{{k}}}}}"  # {{key}}
            if placeholder in content:
                content = content.replace(placeholder, str(v))

        target.write_text(content)

    def run(self, run_dir: Path, log_name: str = "solver.log") -> bool:
        """Execute solver in run_dir."""
        log_path = run_dir / log_name

        print(f"Running {self.solver_cmd} in {run_dir}...")

        try:
            with open(log_path, "w") as log_file:
                subprocess.run(
                    [self.solver_cmd],
                    cwd=run_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,  # 5 min timeout
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

        # Placeholder Regex
        # "Scavenging Efficiency = 0.85"
        metrics = {}

        eff_match = re.search(r"Scavenging Efficiency\s*=\s*([0-9\.]+)", text)
        if eff_match:
            metrics["scavenging_efficiency"] = float(eff_match.group(1))

        mass_match = re.search(r"Trapped Mass\s*=\s*([0-9\.eE\-\+]+)", text)
        if mass_match:
            metrics["trapped_mass"] = float(mass_match.group(1))

        return metrics

    def execute(self, run_dir: Path, params: dict[str, float]) -> dict[str, float]:
        """Full execution pipeline."""
        self.setup_case(run_dir, params)
        success = self.run(run_dir)
        if not success:
            return {"error": 1.0}
        return self.parse_results(run_dir)
