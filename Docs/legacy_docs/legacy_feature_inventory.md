# Legacy Capability Inventory (Surrogates + HiFi)

This file lists notable legacy components that appear mature and are good extraction candidates for `larrak_multi`.

## Surrogates / NN

- `src/truthmaker/surrogates/models/ensemble.py`
  - Ensemble MLP surrogate with uncertainty estimation and threshold calibration.
- `src/truthmaker/surrogates/models/hifi_surrogates.py`
  - Domain-specific thermal + structural surrogate wrappers with physical output bounds.
- `src/truthmaker/surrogates/training/trainer.py`
  - Centralized training loop, validation metrics, and uncertainty logging.
- `src/truthmaker/surrogates/inference/gated.py`
  - Gated inference path that falls back when uncertainty/conditions fail.
- `src/campro/validation/cem_gates.py`
  - Surrogate-backed feasibility checks for thermal and structural constraints.

## OpenFOAM / CalculiX / HiFi Adapters

- `src/Simulations/hifi/base.py`
  - Common external-solver adapter interface.
- `src/Simulations/hifi/docker_solvers.py`
  - Docker wrappers for OpenFOAM and CalculiX execution.
- `src/Simulations/hifi/result_parsers.py`
  - Structured parsers for CalculiX and OpenFOAM outputs.
- `src/Simulations/hifi/structural_fea.py`
  - Structural FEA adapter path.
- `src/Simulations/hifi/combustion_cfd.py`
  - Combustion CFD adapter path.
- `src/Simulations/hifi/conjugate_ht.py`
  - Conjugate heat transfer adapter path.
- `src/Simulations/hifi/port_flow_cfd.py`
  - Port-flow CFD adapter path.
- `src/Simulations/hifi/gear_contact.py`
  - Gear-contact FEA adapter path.

## Orchestration / Integration

- `src/campro/orchestration/adapters/hifi_adapter.py`
  - Surrogate-to-HiFi escalation logic based on uncertainty.
- `src/campro/orchestration/budget.py`
  - Simulation budget allocator with uncertainty-aware prioritization.
- `src/campro/orchestration/trust_region.py`
  - Trust-region controller coupled to surrogate uncertainty.
- `src/campro/client/orchestrator_client.py`
  - API client with explicit `run_openfoam` and `run_calculix` endpoints.

## Data / Training Schemas

- `src/Simulations/hifi/training_schema.py`
  - Shared dataset and normalization schema for surrogate training.
- `src/Simulations/hifi/train_hifi_surrogates.py`
  - End-to-end training entrypoint for thermal/structural surrogates.
