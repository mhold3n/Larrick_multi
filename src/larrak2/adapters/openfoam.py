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

from larrak2.simulation_validation.cantera_mechanisms import sanitize_chemkin_file_text

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
        self.setup_case_with_assets(run_dir, params, staged_inputs=None)

    def setup_case_with_assets(
        self,
        run_dir: Path,
        params: dict[str, float],
        *,
        staged_inputs: list[dict[str, str]] | None = None,
    ) -> None:
        """Clone template, update dictionaries, and optionally stage external inputs."""
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Clone
        shutil.copytree(self.template_dir, run_dir)

        # Replace placeholders across the entire case directory.
        # This allows templates to parameterize arbitrary files under 0/, constant/, system/, etc.
        self._replace_placeholders(run_dir, params)
        self._stage_inputs(run_dir, staged_inputs or [])

    @staticmethod
    def _stage_inputs(run_dir: Path, staged_inputs: list[dict[str, str]]) -> None:
        for item in staged_inputs:
            source = Path(str(item.get("source", "")).strip())
            target = Path(str(item.get("target", "")).strip())
            sanitizer = str(item.get("sanitizer", "")).strip().lower()
            sanitizer_profile = str(item.get("sanitizer_profile", "")).strip()
            if not source or not str(source):
                raise ValueError("OpenFOAM staged input is missing 'source'")
            if not target or not str(target):
                raise ValueError("OpenFOAM staged input is missing 'target'")
            if not source.is_absolute():
                source = (Path.cwd() / source).resolve()
            if not source.exists():
                raise FileNotFoundError(f"OpenFOAM staged input not found: {source}")

            destination = run_dir / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if sanitizer:
                    raise ValueError(
                        "OpenFOAM staged input sanitizer cannot be used with directories"
                    )
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            elif sanitizer in {
                "llnl_gasoline_input",
                "llnl_gasoline_thermo",
                "llnl_gasoline_transport",
            }:
                file_kind = sanitizer.removeprefix("llnl_gasoline_")
                destination.write_text(
                    sanitize_chemkin_file_text(
                        source_file=source,
                        file_kind=file_kind,
                        profile=sanitizer_profile or "llnl_detailed_gasoline_surrogate",
                    ),
                    encoding="utf-8",
                )
            elif sanitizer == "strip_chemkin_comments":
                sanitized_lines: list[str] = []
                for raw_line in source.read_text(encoding="utf-8").splitlines():
                    content = raw_line.split("!", 1)[0].rstrip()
                    if content.strip():
                        sanitized_lines.append(content)
                destination.write_text("\n".join(sanitized_lines) + "\n", encoding="utf-8")
            else:
                shutil.copy2(source, destination)

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

    @staticmethod
    def _extract_named_block(text: str, block_name: str) -> str | None:
        pattern = re.compile(
            rf"(^\s*{re.escape(block_name)}\s*\n\s*\{{)(.*?)(^\s*\}})",
            re.M | re.S,
        )
        match = pattern.search(text)
        if not match:
            return None
        return match.group(2)

    @classmethod
    def _default_ami_patch_value(cls, text: str) -> str:
        inlet_block = cls._extract_named_block(text, "inlet")
        if inlet_block is not None:
            inlet_value = re.search(r"value\s+uniform\s+([^;]+);", inlet_block)
            if inlet_value:
                return inlet_value.group(1).strip()

        uniform_value = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if uniform_value:
            return uniform_value.group(1).strip()

        field_class = re.search(r"class\s+(volVectorField|volScalarField);", text)
        if field_class and field_class.group(1) == "volVectorField":
            return "(0 0 0)"

        first_scalar = re.search(
            r"internalField\s+nonuniform List<scalar>\s*\n\d+\s*\n\(\s*([^\s\)]+)",
            text,
            re.S,
        )
        if first_scalar:
            return first_scalar.group(1).strip()

        first_vector = re.search(
            r"internalField\s+nonuniform List<vector>\s*\n\d+\s*\n\(\s*\(([^)]+)\)",
            text,
            re.S,
        )
        if first_vector:
            return f"({first_vector.group(1).strip()})"

        return "0"

    @classmethod
    def repair_ami_boundary_values(cls, run_dir: Path, *, time_dir: str = "0") -> list[str]:
        """Replace createBaffles zero-initialized AMI patch values with valid field values."""
        target_dir = run_dir / time_dir
        if not target_dir.exists():
            return []

        updated_fields: list[str] = []
        patch_pattern = re.compile(
            r"(^\s*AMI_[^\s]+\s*\n\s*\{)(.*?)(^\s*\})",
            re.M | re.S,
        )

        for field_path in sorted(target_dir.iterdir()):
            if not field_path.is_file():
                continue
            try:
                text = field_path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "AMI_" not in text:
                continue

            fallback_value = cls._default_ami_patch_value(text)

            def _rewrite_patch(match: re.Match[str]) -> str:
                head, body, tail = match.groups()
                if "type" not in body or "cyclicAMI" not in body:
                    return match.group(0)
                if re.search(r"value\s+uniform\s+[^;]+;", body):
                    body = re.sub(
                        r"value\s+uniform\s+[^;]+;",
                        f"value           uniform {fallback_value};",
                        body,
                    )
                else:
                    body = body.rstrip() + f"\n        value           uniform {fallback_value};\n"
                return head + body + tail

            updated = patch_pattern.sub(_rewrite_patch, text)
            if updated != text:
                field_path.write_text(updated, encoding="utf-8")
                updated_fields.append(field_path.name)

        return updated_fields

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
    def latest_numeric_subdir(base_dir: Path) -> Path | None:
        time_dirs: list[tuple[float, Path]] = []
        if not base_dir.exists():
            return None
        for child in base_dir.iterdir():
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
    def cell_volume_field_path(time_dir: Path) -> Path | None:
        """Resolve the cell-volume field name across OpenFOAM variants."""
        for field_name in ("Vc", "V"):
            path = time_dir / field_name
            if path.exists():
                return path
        return None

    @classmethod
    def has_cell_volume_field(cls, time_dir: Path) -> bool:
        return cls.cell_volume_field_path(time_dir) is not None

    @staticmethod
    def clear_validation_outputs(
        run_dir: Path,
        *,
        sample_root: str | None = None,
        purge_numeric_time_dirs: bool = False,
    ) -> None:
        """Remove stale generated validation outputs before a forced live run."""
        sample_dir = run_dir / str(sample_root or "postProcessing")
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        metrics_path = run_dir / "openfoam_metrics.json"
        if metrics_path.exists():
            metrics_path.unlink()
        if purge_numeric_time_dirs:
            for child in run_dir.iterdir():
                if not child.is_dir():
                    continue
                try:
                    time_value = float(child.name)
                except ValueError:
                    continue
                if time_value <= 0.0:
                    continue
                shutil.rmtree(child)

    @staticmethod
    def _read_numeric_table(path: Path) -> list[list[float]]:
        rows: list[list[float]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            pieces = stripped.replace(",", " ").split()
            try:
                row = [float(piece) for piece in pieces]
            except ValueError:
                continue
            if row:
                rows.append(row)
        if not rows:
            raise ValueError(f"No numeric rows found in sample table '{path}'")
        return rows

    @classmethod
    def _resolve_sample_generator_cfg(
        cls,
        run_dir: Path,
        *,
        regime_name: str,
        extractor_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = dict(extractor_cfg or {})
        generator_path = str(
            cfg.get("generator_config_path", "system/liveValidationSamples.json")
        ).strip()
        if not generator_path:
            raise ValueError(
                f"{regime_name} live validation sample generation requires generator_config_path"
            )
        payload = json.loads((run_dir / generator_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at '{run_dir / generator_path}'")
        payload.setdefault("generator_config_path", generator_path)
        return payload

    @staticmethod
    def _read_vector_field(path: Path) -> list[tuple[float, float, float]]:
        text = path.read_text()
        uniform_match = re.search(
            r"internalField\s+uniform\s+\(([^)]+)\)",
            text,
        )
        if uniform_match:
            values = [float(token) for token in uniform_match.group(1).split()]
            if len(values) != 3:
                raise ValueError(f"Vector field '{path}' is not 3D")
            return [(values[0], values[1], values[2])]

        list_match = re.search(
            r"internalField\s+nonuniform List<vector>\s*\n(\d+)\s*\n\((.*?)\n\)",
            text,
            re.S,
        )
        if not list_match:
            raise ValueError(f"Unable to parse vector field: {path}")
        count = int(list_match.group(1))
        values = [
            tuple(float(token) for token in match.group(1).split())
            for match in re.finditer(r"\(([^()]+)\)", list_match.group(2))
        ]
        if len(values) != count:
            raise ValueError(
                f"Field length mismatch in {path}: expected {count}, got {len(values)}"
            )
        return [(value[0], value[1], value[2]) for value in values]

    @classmethod
    def _load_field_values(
        cls,
        run_dir: Path,
        source_dir: Path,
        spec: dict[str, Any],
    ) -> tuple[list[Any], str]:
        field_name = str(spec.get("field", "")).strip()
        if not field_name:
            raise ValueError("Live validation sample spec requires 'field'")
        field_kind = str(spec.get("field_kind", "scalar")).strip().lower()
        candidate_dirs = [source_dir]
        fallback_time_dir = str(spec.get("fallback_time_dir", "")).strip()
        if fallback_time_dir:
            candidate_dirs.append(run_dir / fallback_time_dir)
        if run_dir / "0" not in candidate_dirs:
            candidate_dirs.append(run_dir / "0")

        last_error: Exception | None = None
        for directory in candidate_dirs:
            path = directory / field_name
            if not path.exists():
                continue
            try:
                if field_kind in {"scalar", "surface_scalar"}:
                    return cls._read_scalar_field(path), directory.name
                if field_kind == "vector":
                    return cls._read_vector_field(path), directory.name
                raise ValueError(f"Unsupported field_kind '{field_kind}' for '{field_name}'")
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"Field '{field_name}' not found in '{source_dir}' or fallbacks")

    @staticmethod
    def _reduce_field_values(
        values: list[Any],
        spec: dict[str, Any],
    ) -> float:
        if not values:
            raise ValueError("Cannot reduce an empty field value list")
        statistic = str(spec.get("statistic", "mean")).strip().lower()
        scale = float(spec.get("scale", 1.0))
        offset = float(spec.get("offset", 0.0))
        component_name = str(spec.get("component", "x")).strip().lower()

        if isinstance(values[0], tuple):
            component_index = {"x": 0, "y": 1, "z": 2}.get(component_name, 0)
            components = [float(item[component_index]) for item in values]
            magnitudes = [
                (float(item[0]) ** 2 + float(item[1]) ** 2 + float(item[2]) ** 2) ** 0.5
                for item in values
            ]
            if statistic == "mean_component":
                base_value = sum(components) / len(components)
            elif statistic == "max_component":
                base_value = max(components)
            elif statistic == "min_component":
                base_value = min(components)
            elif statistic == "last_component":
                base_value = components[-1]
            elif statistic == "mean_magnitude":
                base_value = sum(magnitudes) / len(magnitudes)
            elif statistic == "max_magnitude":
                base_value = max(magnitudes)
            elif statistic == "last_magnitude":
                base_value = magnitudes[-1]
            else:
                raise ValueError(f"Unsupported vector statistic '{statistic}'")
        else:
            scalars = [float(item) for item in values]
            if statistic == "mean":
                base_value = sum(scalars) / len(scalars)
            elif statistic == "max":
                base_value = max(scalars)
            elif statistic == "min":
                base_value = min(scalars)
            elif statistic == "last":
                base_value = scalars[-1]
            elif statistic == "first":
                base_value = scalars[0]
            else:
                raise ValueError(f"Unsupported scalar statistic '{statistic}'")
        return base_value * scale + offset

    @classmethod
    def generate_live_validation_samples(
        cls,
        run_dir: Path,
        *,
        regime_name: str,
        extractor_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate fresh validation sample tables from the current case fields."""
        generator_cfg = cls._resolve_sample_generator_cfg(
            run_dir,
            regime_name=regime_name,
            extractor_cfg=extractor_cfg,
        )
        latest_dir = cls.latest_time_dir(run_dir)
        if latest_dir is None:
            raise FileNotFoundError(f"No numeric time directories found in '{run_dir}'")

        source_time_dir = str(generator_cfg.get("source_time_dir", "latest_time")).strip()
        source_dir = (
            latest_dir
            if not source_time_dir or source_time_dir == "latest_time"
            else run_dir / source_time_dir
        )
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Live validation source time directory '{source_dir}' does not exist"
            )

        output_time_dir = str(generator_cfg.get("output_time_dir", "latest_time")).strip()
        output_dir_name = (
            source_dir.name
            if not output_time_dir or output_time_dir == "latest_time"
            else output_time_dir
        )
        sample_root = run_dir / str(generator_cfg.get("sample_root", "postProcessing"))
        output_dir = sample_root / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = dict(generator_cfg.get("metrics", {}) or {})
        if not metrics:
            raise ValueError(
                f"{regime_name} live validation sample generator requires at least one metric"
            )

        field_sources: dict[str, str] = {}
        sample_time_value = source_dir.name
        for relative_path, spec in metrics.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Live validation sample spec for '{relative_path}' must be an object"
                )
            values, resolved_source_dir = cls._load_field_values(run_dir, source_dir, spec)
            field_sources[str(relative_path)] = resolved_source_dir
            value = cls._reduce_field_values(values, spec)
            sample_path = output_dir / str(relative_path)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(
                f"{sample_time_value} {value:.12g}\n",
                encoding="utf-8",
            )

        return {
            "sample_root": str(sample_root.relative_to(run_dir)),
            "source_time_dir": source_dir.name,
            "output_time_dir": output_dir_name,
            "generator_config_path": str(generator_cfg.get("generator_config_path", "")),
            "field_sources": field_sources,
        }

    @classmethod
    def _sample_scalar_metric(
        cls,
        sample_dir: Path,
        relative_path: str,
        *,
        value_column: int = -1,
    ) -> float:
        table_path = sample_dir / relative_path
        rows = cls._read_numeric_table(table_path)
        row = rows[-1]
        index = value_column if value_column >= 0 else len(row) + value_column
        if index < 0 or index >= len(row):
            raise ValueError(
                f"Value column {value_column} is out of range for sample table '{table_path}'"
            )
        return float(row[index])

    @classmethod
    def extract_spray_validation_metrics(
        cls,
        run_dir: Path,
        *,
        extractor_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = dict(extractor_cfg or {})
        sample_root = run_dir / str(cfg.get("sample_root", "postProcessing/sprayValidation"))
        sample_dir = (
            sample_root / str(cfg["sampled_time_dir"])
            if str(cfg.get("sampled_time_dir", "")).strip()
            else cls.latest_numeric_subdir(sample_root)
        )
        if sample_dir is None or not sample_dir.exists():
            raise FileNotFoundError(f"Spray validation samples not found under '{sample_root}'")
        metric_files = {
            "liquid_penetration_max_mm_sprayG": {
                "path": "liquidPenetration_mm.dat",
                "value_column": -1,
            },
            "vapor_spreading_angle_deg_sprayG": {
                "path": "vaporSpreadingAngle_deg.dat",
                "value_column": -1,
            },
            "droplet_smd_um_sprayG_z15mm": {
                "path": "dropletSMD_z15mm_um.dat",
                "value_column": -1,
            },
            "gas_axial_velocity_m_s_sprayG_z15mm_t1ms": {
                "path": "gasAxialVelocity_z15mm_t1ms.dat",
                "value_column": -1,
            },
        }
        metric_files.update(dict(cfg.get("metric_files", {}) or {}))
        metrics = {
            metric_id: cls._sample_scalar_metric(
                sample_dir,
                str(spec.get("path", "")),
                value_column=int(spec.get("value_column", -1)),
            )
            for metric_id, spec in metric_files.items()
        }
        metrics.update(
            {
                "metric_source": "live_case_fields",
                "metric_authority": "live_case_fields",
                "extractor_name": str(cfg.get("name", "spray_g_v1")),
                "sampled_time_dir": sample_dir.name,
            }
        )
        return metrics

    @classmethod
    def extract_reacting_validation_metrics(
        cls,
        run_dir: Path,
        *,
        extractor_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = dict(extractor_cfg or {})
        sample_root = run_dir / str(cfg.get("sample_root", "postProcessing/reactingValidation"))
        sample_dir = (
            sample_root / str(cfg["sampled_time_dir"])
            if str(cfg.get("sampled_time_dir", "")).strip()
            else cls.latest_numeric_subdir(sample_root)
        )
        if sample_dir is None or not sample_dir.exists():
            raise FileNotFoundError(
                f"Reacting-flow validation samples not found under '{sample_root}'"
            )
        metric_files = {
            "gas_temperature_K_iso_octane_reacting": {
                "path": "temperature_K.dat",
                "value_column": -1,
            },
            "CO2_molefrac_iso_octane_reacting": {
                "path": "CO2_molefrac.dat",
                "value_column": -1,
            },
            "OH_molefrac_iso_octane_reacting": {
                "path": "OH_molefrac.dat",
                "value_column": -1,
            },
            "bulk_velocity_m_s_iso_octane_reacting": {
                "path": "bulkVelocity_m_s.dat",
                "value_column": -1,
            },
        }
        metric_files.update(dict(cfg.get("metric_files", {}) or {}))
        metrics = {
            metric_id: cls._sample_scalar_metric(
                sample_dir,
                str(spec.get("path", "")),
                value_column=int(spec.get("value_column", -1)),
            )
            for metric_id, spec in metric_files.items()
        }
        metrics.update(
            {
                "metric_source": "live_case_fields",
                "metric_authority": "live_case_fields",
                "extractor_name": str(cfg.get("name", "reacting_iso_octane_v1")),
                "sampled_time_dir": sample_dir.name,
            }
        )
        return metrics

    @classmethod
    def extract_validation_metrics(
        cls,
        run_dir: Path,
        *,
        regime_name: str,
        extractor_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = regime_name.strip().lower()
        if normalized == "spray":
            return cls.extract_spray_validation_metrics(run_dir, extractor_cfg=extractor_cfg)
        if normalized == "reacting_flow":
            return cls.extract_reacting_validation_metrics(run_dir, extractor_cfg=extractor_cfg)
        raise ValueError(f"No live validation extractor registered for regime '{regime_name}'")

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
        cell_volume_path = cls.cell_volume_field_path(latest_dir)
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
