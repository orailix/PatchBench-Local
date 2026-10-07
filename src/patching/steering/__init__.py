"""Shared steering interfaces and lazy backend loading for benchmark methods."""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import (
    BatchContext,
    InferenceContext,
    OfflineValidationResult,
    PreparedSteeringModel,
    SteeringArtifacts,
    SteeringModel,
    TrainingContext,
    ValidationContext,
    ValidationPlan,
)

if TYPE_CHECKING:
    from .adasteer import AdaSteeringModel
    from .alphasteer import AlphaSteeringModel
    from .ast import ASTSteeringModel
    from .cast import CASTSteeringModel
    from .unsteered import UnsteeredSteeringModel

_STEERING_CLASS_SPECS = {
    "adasteer": ("src.patching.steering.adasteer", "AdaSteeringModel"),
    "alphasteer": ("src.patching.steering.alphasteer", "AlphaSteeringModel"),
    "ast": ("src.patching.steering.ast", "ASTSteeringModel"),
    "cast": ("src.patching.steering.cast", "CASTSteeringModel"),
    "unsteered": ("src.patching.steering.unsteered", "UnsteeredSteeringModel"),
}

_EXPORTED_CLASS_NAMES = {
    class_name: module_path
    for module_path, class_name in _STEERING_CLASS_SPECS.values()
}


def _load_class(module_path: str, class_name: str) -> type[SteeringModel]:
    module = import_module(module_path)
    steering_cls = getattr(module, class_name)
    if not issubclass(steering_cls, SteeringModel):
        raise TypeError(f"{module_path}.{class_name} is not a SteeringModel subclass.")
    return steering_cls


def _load_method_class(method_name: str) -> type[SteeringModel]:
    try:
        module_path, class_name = _STEERING_CLASS_SPECS[method_name]
    except KeyError as exc:
        available = ", ".join(sorted(_STEERING_CLASS_SPECS))
        raise ValueError(f"Unknown steering method {method_name!r}. Available: {available}.") from exc
    return _load_class(module_path, class_name)


def get_steering_method_names() -> list[str]:
    """Return the supported steering method names."""
    return sorted(_STEERING_CLASS_SPECS)


def get_steering_model(method_name: str) -> SteeringModel:
    """Instantiate the steering model for a benchmark method."""
    return _load_method_class(method_name)()


def __getattr__(name: str):
    if name == "STEERING_MODEL_TYPES":
        return {
            method_name: _load_method_class(method_name)
            for method_name in sorted(_STEERING_CLASS_SPECS)
        }
    module_path = _EXPORTED_CLASS_NAMES.get(name)
    if module_path is not None:
        return getattr(import_module(module_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdaSteeringModel",
    "AlphaSteeringModel",
    "ASTSteeringModel",
    "BatchContext",
    "CASTSteeringModel",
    "InferenceContext",
    "OfflineValidationResult",
    "PreparedSteeringModel",
    "SteeringArtifacts",
    "SteeringModel",
    "STEERING_MODEL_TYPES",
    "TrainingContext",
    "UnsteeredSteeringModel",
    "ValidationContext",
    "ValidationPlan",
    "get_steering_model",
    "get_steering_method_names",
]
