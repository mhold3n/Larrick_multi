# Larrak2 — Multi-Objective Optimization Framework

Larrak2 is a modular optimization framework for thermo-gear systems using multi-objective Pareto optimization.

## Evaluation Contract

The canonical interface between physics and optimizers:

```python
evaluate_candidate(x: np.ndarray, ctx: EvalContext) -> EvalResult
```

- **x**: Flat parameter vector (encode/decode via `core.encoding`)
- **ctx**: `EvalContext(rpm, torque, fidelity, seed)`
- **Returns**: `EvalResult(F, G, diag)`
  - `F`: Objectives (minimize)
  - `G`: Constraints with convention **G ≤ 0 feasible**
  - `diag`: Diagnostics dict

## Fidelity Levels

Control physics model complexity via `ctx.fidelity`:

| Fidelity | Thermo Model | Gear Model | Use Case |
|----------|--------------|------------|----------|
| 0 | Toy ideal-gas | Toy friction | Fast prototyping, tests |
| 1 | **Wiebe heat release** | **Litvin synthesis** | Realistic forward-eval |
| 2+ | Reserved | Reserved | Future high-fidelity |

**Fidelity 1** ports forward-evaluation logic from Larrak v1:

- Wiebe function for cumulative burn fraction
- Litvin synthesis for conjugate gear profiles
- Enhanced mesh loss using osculating radius

## Constraint Convention

All constraints follow: **G ≤ 0 → feasible**

## Installation

```bash
pip install -e .
# With dev tools:
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Pareto Optimization

### Basic Usage (Toy Physics)

```bash
python -m larrak2.cli.run_pareto --pop 64 --gen 50
```

### With V1 Physics (Fidelity 1)

```bash
python -m larrak2.cli.run_pareto --fidelity 1 --pop 64 --gen 50 --seed 123 --outdir ./results
```

### All Options

```bash
python -m larrak2.cli.run_pareto \
    --pop 64 \          # Population size
    --gen 100 \         # Generations
    --rpm 3000 \        # Engine speed
    --torque 200 \      # Torque demand
    --fidelity 1 \      # Model fidelity (0=toy, 1=v1)
    --seed 42 \         # Random seed
    --outdir ./output \ # Output directory
    --verbose           # Show progress
```

### Outputs

- `pareto_X.npy` — Decision vectors
- `pareto_F.npy` — Objective values (`[-efficiency, loss]`)
- `pareto_G.npy` — Constraint values (7 constraints)
- `summary.json` — Run metadata including:
  - `n_pareto`: Number of Pareto solutions
  - `feasible_fraction`: Fraction satisfying all constraints
  - `best_efficiency`: Best thermal efficiency achieved
  - `best_loss`: Minimum mesh friction loss (W)

## Project Structure

```
src/larrak2/
├── core/           # Types, encoding, evaluator, utilities
├── thermo/         # Thermodynamic physics models
├── gear/           # Gear synthesis and losses
├── ports/          # Ported code from external sources
│   └── larrak_v1/  # V1 forward-evaluation (Wiebe, Litvin)
├── adapters/       # Optimizer adapters (pymoo, deap, casadi)
└── cli/            # Command-line runners
```
