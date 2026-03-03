"""Global stack surrogate package for symbolic CasADi refinement."""

from .artifact import DenseLayer, StackSurrogateArtifact, load_stack_artifact, save_stack_artifact
from .runtime import StackSurrogateRuntime, default_feature_names, load_stack_runtime
from .symbolic import (
    assemble_symbolic_feature_vector,
    symbolic_forward,
    symbolic_objectives_constraints,
)
from .train import StackMLP, export_torch_mlp_artifact, train_stack_surrogate

__all__ = [
    "DenseLayer",
    "StackMLP",
    "StackSurrogateArtifact",
    "StackSurrogateRuntime",
    "assemble_symbolic_feature_vector",
    "default_feature_names",
    "export_torch_mlp_artifact",
    "load_stack_artifact",
    "load_stack_runtime",
    "save_stack_artifact",
    "symbolic_forward",
    "symbolic_objectives_constraints",
    "train_stack_surrogate",
]

