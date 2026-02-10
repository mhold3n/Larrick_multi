# Testing Strategy & Contribution Workflow

Larrick follows a structured "Promotion Pipeline" for tests to balance rapid local development with a robust, fast CI environment.

## Test Groups

We separate tests into two primary groups:

| Group | Directory | Purpose | Characteristics |
| :--- | :--- | :--- | :--- |
| **Robust (CI)** | `tests/ci/` | Regression testing on every commit. | Fast (<1s), deterministic, mocks external tools, fully automated. |
| **Dev (Local)** | `tests/dev/` | Proof-of-concept, validation, heavy workloads. | Can be slow, flaky, require local setup/files, manual execution. |

### CI Workflow (`tests/ci`)

- Run automatically by GitHub Actions on Push/PR.
- **MUST** depend only on repository files (or standard pip packages).
- **MUST** use mocks for heavy physics solvers (OpenFOAM, FEA) or large NNs.
- **MUST** be deterministic (fix random seeds).

### Dev Workflow (`tests/dev`)

- Run manually by developers (`pytest tests/dev`).
- Can rely on local files, generated datasets, or live solvers.
- Use this space to "proves code during development".
- Validation scripts and "one-off" experiments live here.

## The Promotion Pipeline

New tests should follow this lifecycle:

1. **Incubate (Dev)**
    - Create a new test or script in `tests/dev/`.
    - Iterate rapidly. Hardcode paths, use print debugging, run against local heavy data.
    - *Goal:* Verify the code works functionally.

2. **Refine (Robustness)**
    - Once the feature logic is stable, refactor the test for CI.
    - **Mock Dependencies:** Replace live solver calls with mocks (e.g., `MockOpenFoamSurrogate`).
    - **Assert Logic:** Replace prints with `assert`.
    - **Slim Down:** Use minimal data/inputs to verify logic, not performance.

3. **Deduplicate**
    - Check if an existing CI test already covers this path.
    - If redundant: **Delete** the Dev test. Do not promote.
    - If distinct/valuable coverage: Proceed to Step 4.

4. **Promote (CI)**
    - Move the refined test file to `tests/ci/`.
    - Run `pytest tests/ci/test_my_feature.py` to verify it passes in isolation.
    - Commit and Push.

## Running Tests

- **Run CI Suite (Default):**

  ```bash
  pytest
  # or
  pytest tests/ci
  ```

- **Run Dev Suite (Manual):**

  ```bash
  pytest tests/dev
  ```
