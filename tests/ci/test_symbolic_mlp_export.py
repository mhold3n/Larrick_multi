"""Unit test for torch->artifact->CasADi symbolic export consistency."""

from __future__ import annotations

import numpy as np
import pytest

from larrak2.surrogate.stack.symbolic import symbolic_forward
from larrak2.surrogate.stack.train import Normalization, StackMLP, export_torch_mlp_artifact


def test_symbolic_forward_matches_torch_output() -> None:
    torch = pytest.importorskip("torch")
    ca = pytest.importorskip("casadi")

    rng = np.random.default_rng(5)
    X_ref = rng.normal(size=(64, 4))
    Y_ref = rng.normal(size=(64, 3))
    norm = Normalization.fit(X_ref, Y_ref)

    model = StackMLP(
        input_dim=4,
        hidden_layers=(5,),
        output_dim=3,
        activation="relu",
    )
    model.eval()

    with torch.no_grad():
        for module in model.net:
            if hasattr(module, "weight"):
                module.weight[:] = torch.tensor(
                    rng.normal(scale=0.2, size=tuple(module.weight.shape)),
                    dtype=torch.float32,
                )
            if hasattr(module, "bias"):
                module.bias[:] = torch.tensor(
                    rng.normal(scale=0.1, size=tuple(module.bias.shape)),
                    dtype=torch.float32,
                )

    artifact = export_torch_mlp_artifact(
        model=model,
        normalization=norm,
        feature_names=("x_000", "x_001", "rpm", "torque"),
        objective_names=("obj0", "obj1"),
        constraint_names=("g0",),
        fidelity=1,
    )

    x_feat = np.array([0.25, -0.4, 3100.0, 180.0], dtype=np.float64)
    x_norm = (x_feat - norm.x_mean) / np.where(np.abs(norm.x_std) > 0.0, norm.x_std, 1.0)
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32)).cpu().numpy().astype(np.float64)
    y_torch = y_norm * np.where(np.abs(norm.y_std) > 0.0, norm.y_std, 1.0) + norm.y_mean

    x_sym = ca.MX.sym("x", x_feat.size)
    y_sym = symbolic_forward(artifact, x_sym)
    f = ca.Function("f", [x_sym], [y_sym])
    y_casadi = np.asarray(f(x_feat), dtype=np.float64).reshape(-1)

    assert np.allclose(y_casadi, y_torch, atol=1e-6, rtol=1e-6)
