# Larrak2 — Multi-Objective Optimization Framework

**Single Source of Truth for Larrick Development**

## 1. Project Overview

**Larrak2** is a modular multi-objective optimization framework for thermo-gear systems. It balances fidelity and speed using tiered physics models.

### Fidelity Levels

| Fidelity | Content | Use Case |
| :--- | :--- | :--- |
| **0** | Toy ideal-gas + friction | Fast prototyping, unit tests |
| **1** | Wiebe heat release (Thermo) + Litvin synthesis (Gear) | Realistic forward-eval |
| **2** | NN Surrogates (OpenFOAM/FEA) | Optimization Loop (High Speed) **(Experimental / In Progress)** |

### Core Interface

All physics evaluations conform to:

```python
evaluate_candidate(x: np.ndarray, ctx: EvalContext) -> EvalResult
```

- **Returns**: `F` (Objectives), `G` (Constraints ≤ 0), `diag` (Diagnostics).

---

## 2. Repository Architecture

We enforce a strict layout to maintain modularity.

### Directory Structure

- **`src/larrak2/`**: Application source code.
  - `core/`: Types, encoding, evaluator.
  - `thermo/`: Thermodynamic physics.
  - `gear/`: Gear synthesis and loss models.
  - `surrogate/`: Neural Network interfaces.
  - `adapters/`: Interfaces for solvers (pymoo, OpenFOAM).
- **`tests/`**: Test suite (see Section 3).
- **`scripts/`**: Operational tools and data generators.
- **`data/`**: Large datasets (Git-ignored usually).
- **`outputs/`**: Run artifacts (Git-ignored).

### External Python packages

`larrak-runtime` (from the **`larrak-core`** repo), **`larrak-simulation`**, **`larrak-optimization`**, and **`larrak-orchestration`** are **not** vendored in this repository.

Install the pinned git URLs in [`requirements-external.txt`](requirements-external.txt) via [`scripts/install_external_larrak.sh`](scripts/install_external_larrak.sh) (also symlinks monorepo **`data/`** into the venv for wheel-installed `larrak_runtime`), then install this project:

```bash
bash scripts/install_external_larrak.sh
pip install -e ".[dev]" --no-deps
```

Set **`LARRICK_MULTI_ROOT`** to the root of this checkout when code needs monorepo-relative paths (CI sets it to the workspace root). PicoGK and LEAP71 ShapeKernel C# sources are not submodules here; use upstream clones if you need them for .NET tooling.

Orchestration note: `larrak2.orchestration` remains as the monorepo integration surface (adapters, context construction), but the legacy run-loop core is extracted into **`larrak-orchestration`** and consumed as a pinned dependency.

### Rules

- Do not create root-level folders like `logs`, `results`, `temp`. Use `outputs/` or `data_temp/`.
- All source code must live in `src/`.
- Terminal-generated artifacts should be written under `outputs/`, categorized by source/purpose:
  - `outputs/artifacts/surrogates/openfoam_nn/`
  - `outputs/artifacts/surrogates/calculix_nn/`
  - `outputs/artifacts/surrogates/gear_loss_nn/`
  - `outputs/artifacts/surrogates/v1_gbr/`
  - `outputs/artifacts/surrogates/hifi/`
  - `outputs/artifacts/surrogates/initialization_voxel/`
- Legacy `models/` paths are deprecated. Runtime model reads/writes are strict outputs-only and will fail if pointed at `models/`.
- Artifact layout/migration policy lives in `src/larrak2/artifacts/model_layout.py`.

---

## 3. Testing Strategy

We follow a **"Dev -> Contract CI"** split after package extraction.

### Test Groups

1. **CI gate**: `tests/ci_contract/`
    - **Goal**: Repo contract checks (docs, import hygiene, shim policy).
    - **Rules**: Fast, deterministic, no heavy integration with extracted packages.
    - **Execution**: GitHub Actions runs default `pytest` (see `pyproject.toml` `testpaths`).
2. **Dev (Local)**: `tests/dev/`
    - **Goal**: Proving ground, validation, heavy workloads.
    - **Rules**: Can be slow/flaky. Manual execution only.

### Workflow

1. **Incubate**: Write new tests in `tests/dev/` or in the owning extracted package.
2. **CI**: Keep the default suite small; add contract tests under `tests/ci_contract/` when the monorepo needs guardrails.
3. **Verify**: Commit and push triggers CI.

### Running Tests

- **CI Suite (Default)**: `pytest` (collects `tests/ci_contract` only)
- **Dev Suite**: `pytest tests/dev`

---

## 4. Development Standards

### Coding Style

- **Formatter**: Ruff (Black-compatible). Auto-format before commit.
- **Linting**: Ruff (Standard errors enabled).
- **Typing**: Mypy (Python 3.11+ syntax).

### CI/CD

- **Trigger**: Pushing to `main`.
- **Workflows**:
  - `ci.yml`: Runs `pytest` (Standard suite).

### Git Conventions

- **Commit Messages**: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
- **Branches**: Short-lived feature branches → Merge to `main`.

---

## 5. Constraint & Objective Conventions

- **Objectives (F)**: Always **MINIMIZED**. (Negate maximization targets).
- **Constraints (G)**: **G ≤ 0 is Feasible**.
  - $G > 0$: Violation magnitude.

---

## 6. How to Run

### Installation

```bash
pip install -e ".[dev]"
```

### Running Optimization

```bash
# Fast smoke only (toy physics, never for pass/fail decisions)
python -m larrak2.cli.run_pareto --pop 64 --gen 50

# V1 Physics
python -m larrak2.cli.run_pareto --fidelity 1 --pop 64 --gen 50
```

### Dress Rehearsal (Pre-Analysis Gate)

```bash
# 1) Train NN surrogates (standalone pre-job)
larrak-run train-surrogates --single-condition \
  --openfoam-data outputs/openfoam_doe/results.jsonl \
  --calculix-data data/calculix_doe/train.npz

# 2) Run dress rehearsal (no NN training stage)
larrak-run dress-rehearsal --pop 16 --gen 5 --cem-top 10
```

Outputs include:
- `outputs/dress_rehearsal/dress_rehearsal_manifest.json`
- `outputs/dress_rehearsal/cem_validation_report.txt`
- `outputs/dress_rehearsal/cem_validation_report.json`

Canonical process and hard-first gates: `Docs/main-development-plan.md`

---

## 7. System Vision

For the active roadmap and execution priorities, see `Docs/main-development-plan.md`.
