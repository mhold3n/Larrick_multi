import json
import subprocess
import tempfile
import time
import os
import shutil
import pytest
from pathlib import Path

def _find_dotnet() -> str:
    if shutil.which("dotnet"): return "dotnet"
    for p in ["/usr/local/share/dotnet/dotnet", "/opt/homebrew/bin/dotnet", "/usr/local/bin/dotnet"]:
        if Path(p).exists() and os.access(p, os.X_OK): return p
    return "dotnet"

@pytest.mark.skipif(os.environ.get("LARRAK_PICOGK_ORACLE") != "1", reason="Requires LARRAK_PICOGK_ORACLE=1")
def test_oracle_reachability():
    """
    Minimal connectivity test for PicoGK Oracle.
    """
    print("Running minimal PicoGK reachability test...")
    
    profile = {
        "outer": [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0]
        ],
        "holes": [],
        "process": {
            "wire_d_mm": 0.2,
            "overcut_mm": 0.05,
            "corner_margin_mm": 0.0,
            "min_ligament_mm": 0.35
        },
        "metadata": {"test": "minimal_reachability"},
        "units": "mm"
    }
    
    oracle_project = Path("tools/picogk_manufact").resolve()
    if not oracle_project.exists():
        pytest.skip(f"Oracle project not found at {oracle_project}")

    resolutions = [0.01] 
    
    for voxel_size in resolutions:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "profile.json"
            with open(input_path, "w") as f:
                json.dump(profile, f)
            
            cmd = [
                _find_dotnet(), "run", 
                "--project", str(oracle_project), 
                "--",
                "--input", str(input_path),
                "--voxel-size", str(voxel_size),
                "--slab-thickness", "5.0"
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                print(f"Stderr: {proc.stderr[:500]}")
            
            assert proc.returncode == 0, f"Oracle failed with rc={proc.returncode}"
            
            res = json.loads(proc.stdout)
            assert "passed" in res

if __name__ == "__main__":
    test_oracle_reachability()
