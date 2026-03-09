"""OpenFOAM Adapter.

Manages execution of OpenFOAM cases for Scavenging analysis.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .docker_openfoam import DockerOpenFoam, DockerOpenFoamConfig

R_SPECIFIC_AIR = 287.05
O2_MASS_FRACTION_AIR = 0.233


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

    def parse_results(self, run_dir: Path, log_name: str = "solver.log") -> dict[str, Any]:
        """Parse emitted metrics from the solver log and/or sidecar metrics file."""
        log_path = run_dir / log_name
        metrics_path = run_dir / "openfoam_metrics.json"
        metrics: dict[str, Any] = {}

        if metrics_path.exists():
            try:
                loaded = json.loads(metrics_path.read_text())
                if isinstance(loaded, dict):
                    metrics.update(loaded)
            except Exception:
                pass

        if not log_path.exists():
            return metrics

        text = log_path.read_text()

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

    @staticmethod
    def latest_time_dir(run_dir: Path) -> Path | None:
        time_dirs: list[tuple[float, Path]] = []
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                time_value = float(child.name)
            except ValueError:
                continue
            time_dirs.append((time_value, child))
        if not time_dirs:
            return None
        time_dirs.sort(key=lambda item: item[0])
        return time_dirs[-1][1]

    @staticmethod
    def _read_scalar_field(path: Path) -> list[float]:
        text = path.read_text()
        uniform_match = re.search(
            r"internalField\s+uniform\s+([0-9]+(?:\.[0-9]+)?(?:[eE][\-\+]?[0-9]+)?)",
            text,
        )
        if uniform_match:
            return [float(uniform_match.group(1))]

        list_match = re.search(
            r"internalField\s+nonuniform List<scalar>\s*\n(\d+)\s*\n\((.*?)\n\)",
            text,
            re.S,
        )
        if not list_match:
            raise ValueError(f"Unable to parse scalar field: {path}")
        count = int(list_match.group(1))
        values = [float(token) for token in list_match.group(2).split()]
        if len(values) != count:
            raise ValueError(
                f"Field length mismatch in {path}: expected {count}, got {len(values)}"
            )
        return values

    @classmethod
    def compute_field_metrics(
        cls,
        run_dir: Path,
        *,
        p_manifold_Pa: float,
        intake_temp_K: float | None = None,
    ) -> dict[str, Any]:
        latest_dir = cls.latest_time_dir(run_dir)
        if latest_dir is None:
            return {}

        rho_path = latest_dir / "rho"
        temperature_path = latest_dir / "T"
        cell_volume_path = latest_dir / "Vc"
        if not (rho_path.exists() and temperature_path.exists() and cell_volume_path.exists()):
            return {}

        rho = cls._read_scalar_field(rho_path)
        temperature = cls._read_scalar_field(temperature_path)
        cell_volume = cls._read_scalar_field(cell_volume_path)
        if not (len(rho) == len(temperature) == len(cell_volume)):
            return {}

        trapped_mass = float(sum(r * v for r, v in zip(rho, cell_volume)))
        domain_volume = float(sum(cell_volume))
        if trapped_mass <= 0.0 or domain_volume <= 0.0:
            return {}

        mass_weighted_temperature = float(
            sum(r * t * v for r, t, v in zip(rho, temperature, cell_volume)) / trapped_mass
        )

        if intake_temp_K is None:
            initial_temperature_path = run_dir / "0" / "T"
            if initial_temperature_path.exists():
                initial_temperature_values = cls._read_scalar_field(initial_temperature_path)
                intake_temp_K = float(
                    sum(initial_temperature_values) / len(initial_temperature_values)
                )
            else:
                intake_temp_K = 300.0

        fresh_mass_reference = float(
            max(p_manifold_Pa, 1.0)
            * domain_volume
            / (R_SPECIFIC_AIR * max(float(intake_temp_K), 1.0))
        )
        fresh_charge_fraction = float(
            max(0.0, min(1.0, trapped_mass / max(fresh_mass_reference, 1.0e-12)))
        )
        residual_fraction = float(max(0.0, min(1.0, 1.0 - fresh_charge_fraction)))
        scavenging_efficiency = float(max(0.0, min(1.0, fresh_charge_fraction)))
        trapped_o2_mass = float(trapped_mass * O2_MASS_FRACTION_AIR * fresh_charge_fraction)

        return {
            "trapped_mass": trapped_mass,
            "scavenging_efficiency": scavenging_efficiency,
            "residual_fraction": residual_fraction,
            "trapped_o2_mass": trapped_o2_mass,
            "metric_source": "field_postprocess_mass_reference_v1",
            "strict_metric_authority": "case_fields",
            "metric_time_dir": latest_dir.name,
            "domain_volume_m3": domain_volume,
            "mass_weighted_temperature_K": mass_weighted_temperature,
            "fresh_mass_reference_kg": fresh_mass_reference,
            "fresh_charge_fraction": fresh_charge_fraction,
        }

    @staticmethod
    def emit_metrics(
        run_dir: Path, metrics: dict[str, Any], *, log_name: str = "solver.log"
    ) -> None:
        (run_dir / "openfoam_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )
        log_path = run_dir / log_name
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nOpenFOAM Postprocessed Metrics\n")
            if "scavenging_efficiency" in metrics:
                handle.write(f"Scavenging Efficiency = {metrics['scavenging_efficiency']:.9g}\n")
            if "trapped_mass" in metrics:
                handle.write(f"Trapped Mass = {metrics['trapped_mass']:.9g}\n")
            if "residual_fraction" in metrics:
                handle.write(f"Residual Fraction = {metrics['residual_fraction']:.9g}\n")
            if "trapped_o2_mass" in metrics:
                handle.write(f"Trapped O2 Mass = {metrics['trapped_o2_mass']:.9g}\n")
            if "metric_source" in metrics:
                handle.write(f"Metric Source = {metrics['metric_source']}\n")

    def execute(self, run_dir: Path, params: dict[str, float]) -> dict[str, float]:
        """Full execution pipeline."""
        self.setup_case(run_dir, params)
        success = self.run(run_dir)
        if not success:
            return {"error": 1.0}
        return self.parse_results(run_dir)
