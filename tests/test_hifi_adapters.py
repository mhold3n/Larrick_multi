"""Unit tests for High-Fidelity Physics Adapters."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.adapters.calculix import CalculiXRunner


@pytest.fixture
def temp_dirs(tmp_path):
    tpl = tmp_path / "template"
    tpl.mkdir()
    (tpl / "constant").mkdir()
    (tpl / "constant" / "pistonMeshDict").write_text("compression {{compression_ratio}};")
    
    run_dir = tmp_path / "run"
    return tpl, run_dir


def test_openfoam_setup(temp_dirs):
    tpl, run = temp_dirs
    runner = OpenFoamRunner(tpl)
    
    params = {"compression_ratio": 15.5}
    runner.setup_case(run, params)
    
    assert run.exists()
    assert (run / "constant" / "pistonMeshDict").exists()
    content = (run / "constant" / "pistonMeshDict").read_text()
    assert "compression 15.5;" in content


def test_openfoam_parse(temp_dirs):
    _, run = temp_dirs
    run.mkdir()
    runner = OpenFoamRunner(Path("dummy"))
    
    log = run / "solver.log"
    log.write_text("""
    Time = 0.5
    Scavenging Efficiency = 0.92
    Trapped Mass = 1.25e-3
    End
    """)
    
    res = runner.parse_results(run)
    assert res["scavenging_efficiency"] == 0.92
    assert res["trapped_mass"] == 1.25e-3


@patch("subprocess.run")
def test_openfoam_execution(mock_run, temp_dirs):
    tpl, run = temp_dirs
    runner = OpenFoamRunner(tpl)
    
    run.mkdir()
    runner.run(run)
    
    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "pisoFoam"


def test_calculix_gen_inp(temp_dirs):
    tpl, run = temp_dirs
    # Setup template
    tpl_inp = tpl / "gear.inp"
    tpl_inp.write_text("*NODE\n1, {{base_radius}}, 0, 0")
    
    runner = CalculiXRunner(tpl_inp)
    
    params = {"base_radius": 50.0}
    runner.generate_inp(run, "job1", params)
    
    assert (run / "job1.inp").exists()
    content = (run / "job1.inp").read_text()
    assert "1, 50.0, 0, 0" in content


@patch("subprocess.run")
def test_calculix_execution(mock_run, temp_dirs):
    tpl, run = temp_dirs
    runner = CalculiXRunner(Path("dummy"))
    
    run.mkdir()
    runner.run(run, "job1")
    
    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "ccx"
    assert args[1] == "job1"


def test_calculix_parse(temp_dirs):
    _, run = temp_dirs
    run.mkdir()
    runner = CalculiXRunner(Path("dummy"))
    
    dat = run / "job1.dat"
    dat.write_text("""
    NODE OUTPUT
    MAXIMUM S Mises 2.50E+02
    """)
    
    res = runner.parse_results(run, "job1")
    assert res["max_stress"] == 250.0
