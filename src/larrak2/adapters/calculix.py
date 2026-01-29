"""CalculiX Adapter.

Manages execution of CalculiX (ccx) cases for Gear Stress analysis.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class CalculiXRunner:
    """Manages CalculiX execution."""

    def __init__(self, template_path: str | Path, solver_cmd: str = "ccx"):
        self.template_path = Path(template_path)
        self.solver_cmd = solver_cmd

    def generate_inp(self, run_dir: Path, job_name: str, params: dict[str, float]):
        """Generate input file from template."""
        if not run_dir.exists():
            run_dir.mkdir(parents=True)

        target_inp = run_dir / f"{job_name}.inp"

        if not self.template_path.exists():
            # If template missing, maybe write a stub for testing?
            # Or raise.
            # raise FileNotFoundError(f"Template {self.template_path} missing")
            # For now, let's write a dummy one if missing so we can test the runner logic
            print(f"Warning: Template {self.template_path} missing, using stub.")
            content = "*NODE\n1, 0,0,0\n*ELEMENT\n*STEP\n*END STEP"
        else:
            content = self.template_path.read_text()

        # Replace placeholders
        for k, v in params.items():
            placeholder = f"{{{{{k}}}}}"
            if placeholder in content:
                content = content.replace(placeholder, str(v))

        target_inp.write_text(content)

    def run(self, run_dir: Path, job_name: str) -> bool:
        """Execute ccx."""
        print(f"Running {self.solver_cmd} {job_name} in {run_dir}...")

        try:
            # ccx usually prints to stdout
            log_path = run_dir / f"{job_name}.log"
            with open(log_path, "w") as log_file:
                subprocess.run(
                    [self.solver_cmd, job_name],
                    cwd=run_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"CalculiX execution failed: {e}")
            return False

    def parse_results(self, run_dir: Path, job_name: str) -> dict[str, float]:
        """Parse .dat file for max stress."""
        # CalculiX writes to .dat if *NODE PRINT or *EL PRINT is used
        dat_path = run_dir / f"{job_name}.dat"
        metrics = {}

        if not dat_path.exists():
            # Try parsing log for info?
            return {}

        text = dat_path.read_text()

        # Parse Max Stress (S Mises)
        # Typically looking for a table or summary
        # Placeholder regex for finding "MAXIMUM"

        match = re.search(r"MAXIMUM\s+.*?\s+([0-9\.E\+\-]+)", text)
        if match:
            metrics["max_stress"] = float(match.group(1))

        return metrics

    def execute(self, run_dir: Path, job_name: str, params: dict[str, float]) -> dict[str, float]:
        """Full execution pipeline."""
        self.generate_inp(run_dir, job_name, params)
        success = self.run(run_dir, job_name)
        if not success:
            return {"error": 1.0}
        return self.parse_results(run_dir, job_name)
