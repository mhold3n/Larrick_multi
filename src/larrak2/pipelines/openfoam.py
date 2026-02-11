"""OpenFOAM Execution Pipeline.

This module consolidates the orchestration logic for running OpenFOAM cases,
including template cloning, parameter substitution, geometry generation,
and the complex meshing/solving sequence (overset, sliding mesh, etc.).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from larrak2.adapters.docker_openfoam import DockerOpenFoam
from larrak2.adapters.openfoam import OpenFoamRunner


class OpenFoamPipeline:
    """Orchestrates the end-to-end execution of an OpenFOAM case."""

    def __init__(
        self,
        template_dir: str | Path,
        solver_cmd: str = "pisoFoam",
        docker_timeout_s: int = 1800,
    ):
        self.template_dir = Path(template_dir)
        self.solver_cmd = solver_cmd
        self.docker_timeout_s = docker_timeout_s
        self.docker = DockerOpenFoam()
        self.runner = OpenFoamRunner(
            template_dir=template_dir,
            solver_cmd=solver_cmd,
            backend="docker",
        )

    def setup_case(self, run_dir: Path, params: dict[str, Any]) -> None:
        """Clone template and substitute parameters."""
        self.runner.setup_case(run_dir, params)

    def generate_geometry(
        self,
        run_dir: Path,
        bore_mm: float,
        stroke_mm: float,
        intake_port_area_m2: float,
        exhaust_port_area_m2: float,
    ) -> None:
        """Generate STL geometry for the case."""
        tri_dir = run_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)

        from larrak2.geometry.generate_stl import generate_stl_workflow

        # Logic adapted from generate_stl.py main() to avoid shelling out
        # We can also just call the workflow function if we restructure it to accept args object or kwargs
        # But let's reuse the logic since we have access to it.
        # Actually, simpler: construct a dummy args object and call generate_stl_workflow?
        # Or better: refactor generate_stl.py to have a clean python API.

        # For now, let's implement the logic here using the imported functions
        # OR better yet, let's update generate_stl.py to have a `generate_geometry_files` function
        # and call that.

        # Let's assume we update generate_stl.py first.
        # But since I can't do that in this atomic step without multi-file edit (which is fine),
        # I'll just use the provided generate_stl_workflow but I need to mock args.

        class GeometryArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        args = GeometryArgs(
            outdir=str(tri_dir),
            bore_mm=bore_mm,
            stroke_mm=stroke_mm,
            intake_port_area_m2=intake_port_area_m2,
            exhaust_port_area_m2=exhaust_port_area_m2,
        )

        generate_stl_workflow(args)

    def _solver_completed_successfully(self, log_file: Path) -> bool:
        """Check if solver completed by looking for 'End' marker."""
        if not log_file.exists():
            return False
        try:
            with log_file.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(2048, size)
                f.seek(max(0, size - read_size))
                tail = f.read().decode("utf-8", errors="ignore")
            # Check for 'End' marker
            return "\nEnd\n" in tail or tail.strip().endswith("End")
        except Exception:
            return False

    def run_meshing(self, run_dir: Path) -> tuple[bool, str]:
        """Execute the meshing sequence (BlockMesh -> Snappy -> ...)."""
        is_overset = (run_dir / "system" / "snappyHexMeshDict.background").exists()
        is_sliding = (run_dir / "system" / "snappyHexMeshDict").exists() and not is_overset

        if is_overset:
            # 1. Background Mesh
            c, _, _ = self.docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                log_file=run_dir / "blockMesh.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "blockMesh"

            c, _, _ = self.docker.run_utility(
                utility="snappyHexMesh",
                args=["-dict", "system/snappyHexMeshDict.background", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "snappyHexMesh.background.log",
                timeout_s=1200,
            )
            if c != 0:
                return False, "snappyHexMesh.background"

            # Save background
            if (run_dir / "constant/polyMesh.background").exists():
                shutil.rmtree(run_dir / "constant/polyMesh.background")
            shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.background")

            # 2. Valve Mesh
            c, _, _ = self.docker.run_utility(
                utility="blockMesh", case_dir=run_dir, timeout_s=600
            )  # Re-run blockMesh for valve domain

            c, _, _ = self.docker.run_utility(
                utility="snappyHexMesh",
                args=["-dict", "system/snappyHexMeshDict.valve", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "snappyHexMesh.valve.log",
                timeout_s=1200,
            )
            if c != 0:
                return False, "snappyHexMesh.valve"

            # Save valve
            if (run_dir / "constant/polyMesh.valves").exists():
                shutil.rmtree(run_dir / "constant/polyMesh.valves")
            shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.valves")

            # Restore background as master
            shutil.move(run_dir / "constant/polyMesh.background", run_dir / "constant/polyMesh")

            # 3. Merge
            valve_case_dir = run_dir / "valve_mesh_case"
            if valve_case_dir.exists():
                shutil.rmtree(valve_case_dir)
            valve_case_dir.mkdir()
            (valve_case_dir / "constant").mkdir()
            shutil.copytree(run_dir / "system", valve_case_dir / "system")
            shutil.copytree(
                run_dir / "constant/polyMesh.valves", valve_case_dir / "constant/polyMesh"
            )

            c, _, _ = self.docker.run_utility(
                utility="mergeMeshes",
                args=[".", "valve_mesh_case", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "mergeMeshes.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "mergeMeshes"

            if valve_case_dir.exists():
                shutil.rmtree(valve_case_dir)

            # 4. Zones (TopoSet)
            c, _, _ = self.docker.run_utility(
                utility="topoSet",
                case_dir=run_dir,
                log_file=run_dir / "topoSet.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "topoSet"

            # 5. Fields (SetFields)
            c, _, _ = self.docker.run_utility(
                utility="setFields",
                case_dir=run_dir,
                log_file=run_dir / "setFields.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "setFields"

        else:
            # Standard / Sliding Mesh Pipeline
            c, _, _ = self.docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                timeout_s=300,
                log_file=run_dir / "blockMesh.log",
            )
            if c != 0:
                return False, "blockMesh"

            if is_sliding:
                c, _, _ = self.docker.run_utility(
                    utility="snappyHexMesh",
                    args=["-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "snappyHexMesh.log",
                    timeout_s=1200,
                )
                if c != 0:
                    return False, "snappyHexMesh"

                c, _, _ = self.docker.run_utility(
                    utility="topoSet",
                    case_dir=run_dir,
                    log_file=run_dir / "topoSet.log",
                )
                if c != 0:
                    return False, "topoSet"

                if (run_dir / "system/createBafflesDict").exists():
                    c, _, _ = self.docker.run_utility(
                        utility="createBaffles",
                        args=["-overwrite"],
                        case_dir=run_dir,
                        log_file=run_dir / "createBaffles.log",
                    )
                    if c != 0:
                        return False, "createBaffles"

                if (run_dir / "system/createPatchDict").exists():
                    c, _, _ = self.docker.run_utility(
                        utility="createPatch",
                        args=["-overwrite"],
                        case_dir=run_dir,
                        log_file=run_dir / "createPatch.log",
                    )
                    if c != 0:
                        return False, "createPatch"

                c, _, _ = self.docker.run_utility(
                    utility="setFields",
                    case_dir=run_dir,
                    log_file=run_dir / "setFields.log",
                )
                if c != 0:
                    return False, "setFields"

        return True, ""

    def run_solver(self, run_dir: Path) -> tuple[bool, str]:
        """Execute the solver."""
        solver_log_file = run_dir / f"{self.solver_cmd}.log"

        code, _, _ = self.docker.run_solver(
            solver=self.solver_cmd,
            case_dir=run_dir,
            timeout_s=self.docker_timeout_s,
            log_file=solver_log_file,
        )

        if code != 0:
            # Check if solver actually completed check
            if self._solver_completed_successfully(solver_log_file):
                return True, ""
            return False, "solver"

        return True, ""

    def parse_results(self, run_dir: Path) -> dict[str, float]:
        """Parse results using the runner adapter."""
        return self.runner.parse_results(run_dir, log_name=f"{self.solver_cmd}.log")

    def execute(
        self,
        run_dir: Path,
        params: dict[str, Any],
        geometry_args: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute full pipeline for a single case."""

        # 1. Setup
        self.setup_case(run_dir, params)

        # 2. Geometry
        if geometry_args:
            self.generate_geometry(
                run_dir,
                bore_mm=geometry_args["bore_mm"],
                stroke_mm=geometry_args["stroke_mm"],
                intake_port_area_m2=geometry_args["intake_port_area_m2"],
                exhaust_port_area_m2=geometry_args["exhaust_port_area_m2"],
            )

        # 3. Mesh
        ok, stage = self.run_meshing(run_dir)
        if not ok:
            return {"error": 1.0, "stage": stage, "ok": False}

        # 4. Solve
        ok, stage = self.run_solver(run_dir)
        if not ok:
            return {"error": 1.0, "stage": stage, "ok": False}

        # 5. Parse
        metrics = self.parse_results(run_dir)
        return {**metrics, "ok": True}
