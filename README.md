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

### Rules

- Do not create root-level folders like `logs`, `results`, `temp`. Use `outputs/` or `data_temp/`.
- All source code must live in `src/`.

---

## 3. Testing Strategy

We follow a **"Dev -> Robust -> CI"** promotion pipeline.

### Test Groups

1. **Robust (CI)**: `tests/ci/`
    - **Goal**: Regression testing.
    - **Rules**: Fast (<1s), Deterministic, Mocks external tools.
    - **Execution**: Ran automatically by GitHub Actions on every push.
2. **Dev (Local)**: `tests/dev/`
    - **Goal**: Proving ground, validation, heavy workloads.
    - **Rules**: Can be slow/flaky. Manual execution only.

### Workflow

1. **Incubate**: Write new tests in `tests/dev/`. Verify logic.
2. **Refine**: Mock heavy dependencies. Ensure determinism.
3. **Promote**: Move to `tests/ci/` when stable.
4. **Verify**: Commit and push triggers CI.

### Running Tests

- **CI Suite (Default)**: `pytest`
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
# Toy Physics
python -m larrak2.cli.run_pareto --pop 64 --gen 50

# V1 Physics
python -m larrak2.cli.run_pareto --fidelity 1 --pop 64 --gen 50
```

---

## 7. System Vision

For detailed high-level architecture diagrams and future roadmap, see `outline.md`.
