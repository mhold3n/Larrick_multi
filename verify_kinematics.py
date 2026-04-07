import numpy as np
from larrak_engines.evaluator import evaluate_candidate
from larrak_runtime.core.encoding import decode_candidate, mid_bounds_candidate
from larrak_runtime.core.types import EvalContext


def main():
    x = mid_bounds_candidate()
    _candidate = decode_candidate(x)

    # Fidelity 0
    print("Evaluating Fidelity 0...")
    ctx0 = EvalContext(rpm=1000.0, torque=50.0, fidelity=0)
    res0 = evaluate_candidate(x, ctx0)

    # Fidelity 1
    print("\nEvaluating Fidelity 1...")
    ctx1 = EvalContext(rpm=1000.0, torque=50.0, fidelity=1)
    res1 = evaluate_candidate(x, ctx1)

    print("\n--- Diagnostic Check ---")
    diag0 = res0.diag.get("gear", {})
    diag1 = res1.diag.get("gear", {})

    keys_to_check = [
        "hertz_stress_max",
        "sliding_speed_max",
        "entrainment_velocity_mean",
        "hertz_stress_profile",
        "sliding_speed_profile",
        "entrainment_velocity_profile",
        "fn_profile",
    ]

    print("\nFidelity 0 Keys Present:")
    for k in keys_to_check:
        v = diag0.get(k)
        if v is not None:
            if isinstance(v, np.ndarray):
                print(f"  {k}: Array of shape {v.shape}, range [{v.min():.2f}, {v.max():.2f}]")
            else:
                print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: MISSING")

    print("\nFidelity 1 Keys Present:")
    for k in keys_to_check:
        v = diag1.get(k)
        if v is not None:
            if isinstance(v, np.ndarray):
                print(f"  {k}: Array of shape {v.shape}, range [{v.min():.2f}, {v.max():.2f}]")
            else:
                print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: MISSING")

    # Check if life damage is in realworld diag
    rw0 = res0.diag.get("realworld", {})
    rw1 = res1.diag.get("realworld", {})

    print(f"\nPhase-resolved life damage evaluated in FID 0? {'life_damage' in rw0}")
    print(f"Phase-resolved life damage evaluated in FID 1? {'life_damage' in rw1}")


if __name__ == "__main__":
    main()
