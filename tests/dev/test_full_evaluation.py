import numpy as np
from larrak_runtime.core.constraints import get_constraint_names
from larrak_runtime.core.evaluator import evaluate_candidate
from larrak_runtime.core.types import EvalContext


def test_full_eval():
    print("Testing Full Evaluation with Machining Constraints...")

    # 1. Setup Context
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)

    # 2. Mock Decision Vector (Random)
    # n_vars = 61 (encoding.N_TOTAL)
    n_vars = 61
    x = np.random.uniform(0.1, 0.9, n_vars)

    # 3. Evaluate
    print("Running evaluate_candidate...")
    res = evaluate_candidate(x, ctx)

    print("\n--- Results ---")
    print(f"F (Objectives): {res.F}")
    print(f"G (Constraints): {res.G}")

    # Check Constraint Length
    expected_names = get_constraint_names(ctx.fidelity)
    print(f"Expected Constraints: {len(expected_names)}")
    print(f"Actual Constraints: {len(res.G)}")

    if len(res.G) != len(expected_names):
        print(f"FAIL: Constraint length mismatch! {len(res.G)} != {len(expected_names)}")
    else:
        print("PASS: Constraint length matches.")

    # Check Diagnostics
    if "machining" in res.diag:
        print("PASS: Machining diagnostics present.")
        m = res.diag["machining"]
        print(f"  Tooling Cost: {m['tooling_cost']:.2f}")
        print(f"  Tolerance Penalty: {m['tol_penalty']:.2f}")
        print(f"  TMin Proxy: {m['t_min_proxy_mm']:.4f}")
    else:
        print("FAIL: Machining diagnostics missing!")


if __name__ == "__main__":
    test_full_eval()
