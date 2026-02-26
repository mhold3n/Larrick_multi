# Dress Rehearsal Process (Pre-Analysis Gate)

This workflow is the required gate before any optimization-quality analysis.

## Command

```bash
larrak-run dress-rehearsal --pop 16 --gen 5 --cem-top 10
```

## Stage 0: Surrogate Pre-Job (Required)

Train NN surrogates before dress rehearsal:

```bash
larrak-run train-surrogates --single-condition \
  --openfoam-data data/openfoam_doe/results.jsonl \
  --calculix-data data/calculix_doe/train.npz
```

OpenFOAM/CalculiX training data are real-data only:

- Provide `--openfoam-data` and `--calculix-data`, or
- Provide `--openfoam-template` and `--calculix-template` so the pre-job generates DOE-backed datasets before training.

## Stage 1: Surrogate Verification

Dress rehearsal verifies required surrogate artifacts are present:

- OpenFOAM NN (required for `fidelity>=2`)
- CalculiX NN (required when `--calculix-stress-mode nn`)
- Gear-loss NN directory (required when `--gear-loss-mode nn`)

## Stage 2: Unit Test Gate

The workflow runs unit tests (`pytest`) before optimization.

- Default: enabled (`--run-unit-tests`)
- Optional bypass for smoke runs: `--skip-unit-tests`

## Stage 3: Optimization Run (Condition Sweep)

Runs `run_pareto` over a low-load coarse operating grid by default:

- RPM sweep: `--rpm-min/--rpm-max/--rpm-step`
- Torque sweep: `--torque-min/--torque-max/--torque-step`
- Disable sweep and use a single point with `--single-condition`
- Strategy/stress mode is explicit:
  - `--calculix-stress-mode nn` (default, strict) or `--calculix-stress-mode analytical` (manual bypass)
  - `--gear-loss-mode physics` (default) or `--gear-loss-mode nn` (manual NN path)

Per-condition outputs are written under `optimization/rpm*_tq*/`, then aggregated to:

- `optimization/pareto_X.npy`
- `optimization/pareto_F.npy`
- `optimization/pareto_G.npy`
- `optimization/final_pop_X.npy`
- `optimization/final_pop_F.npy`

## Stage 4: CEM Completion Gate

Runs full CEM validation on top Pareto candidates (`--cem-top`) and writes:

- `cem_validation_report.txt`
- `cem_validation_report.json`

Gate criterion:

- `n_feasible >= --cem-min-feasible`
- and optimization produced at least `--min-pareto` candidates
- and unit tests passed

## Final Gate Artifact

The canonical gate file is:

- `dress_rehearsal_manifest.json`

Use `ready_for_quality_analysis` as the single decision flag for whether analysis can proceed.

By default, the command exits non-zero if the gate fails.
Use `--allow-gate-failure` to continue without failing the command.
