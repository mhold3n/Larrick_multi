"""Symbolic CasADi graph construction for stack surrogate artifacts."""

from __future__ import annotations

import numpy as np

from .artifact import StackSurrogateArtifact
from .runtime import parse_feature_index


def _import_casadi():
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover - exercised by caller handling
        raise ImportError(f"CasADi is required for symbolic stack evaluation: {exc}") from exc
    return ca


def assemble_symbolic_feature_vector(
    *,
    artifact: StackSurrogateArtifact,
    x_full_sym,
    rpm: float,
    torque: float,
):
    """Build symbolic feature vector from full symbolic design variable."""
    ca = _import_casadi()
    feats = []
    for name in artifact.feature_names:
        if name == "rpm":
            feats.append(ca.DM([float(rpm)])[0])
            continue
        if name == "torque":
            feats.append(ca.DM([float(torque)])[0])
            continue
        idx = parse_feature_index(name)
        if idx is None:
            raise ValueError(f"Unsupported stack feature '{name}'")
        feats.append(x_full_sym[int(idx)])
    return ca.vertcat(*feats)


def _activation(z, activation: str, slope: float):
    ca = _import_casadi()
    if activation == "relu":
        return ca.fmax(z, 0.0)
    if activation == "leaky_relu":
        a = float(slope)
        return ca.if_else(z >= 0.0, z, a * z)
    raise ValueError(f"Unsupported activation '{activation}'")


def symbolic_forward(artifact: StackSurrogateArtifact, x_features_sym):
    """Evaluate exported MLP symbolically and return de-normalized output vector."""
    ca = _import_casadi()
    x_mean = ca.DM(np.asarray(artifact.x_mean, dtype=np.float64))
    x_std = ca.DM(np.where(np.abs(artifact.x_std) > 0.0, artifact.x_std, 1.0))
    y_mean = ca.DM(np.asarray(artifact.y_mean, dtype=np.float64))
    y_std = ca.DM(np.where(np.abs(artifact.y_std) > 0.0, artifact.y_std, 1.0))

    h = (x_features_sym - x_mean) / x_std
    for i, layer in enumerate(artifact.layers):
        W = ca.DM(np.asarray(layer.weight, dtype=np.float64))
        b = ca.DM(np.asarray(layer.bias, dtype=np.float64))
        h = W @ h + b
        if i < len(artifact.layers) - 1:
            h = _activation(h, artifact.activation, artifact.leaky_relu_slope)

    y = h * y_std + y_mean
    return y


def symbolic_objectives_constraints(
    artifact: StackSurrogateArtifact,
    x_features_sym,
):
    """Return tuple(F_hat, G_hat) from symbolic features."""
    y = symbolic_forward(artifact, x_features_sym)
    ca = _import_casadi()
    n_obj = len(artifact.objective_names)
    F_hat = y[:n_obj]
    G_hat = y[n_obj:]
    if not isinstance(F_hat, ca.MX) and not isinstance(F_hat, ca.SX):
        F_hat = ca.MX(F_hat)
    if not isinstance(G_hat, ca.MX) and not isinstance(G_hat, ca.SX):
        G_hat = ca.MX(G_hat)
    return F_hat, G_hat
