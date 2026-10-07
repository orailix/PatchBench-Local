
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class SteeringArtifacts:
    """Files and metadata produced by training."""

    paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingContext:
    """Inputs shared by all training methods."""

    method_name: str
    model_name: str
    snapshot_path: Path
    artifacts_dir: Path
    prompt_settings: Any
    data_paths: dict[str, Path] = field(default_factory=dict)
    method_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceContext:
    """Inputs shared by all inference methods."""

    method_name: str
    model_name: str
    snapshot_path: Path
    dtype: torch.dtype | str
    prompt_settings: Any
    artifacts: SteeringArtifacts = field(default_factory=SteeringArtifacts)
    method_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationContext:
    """Inputs shared by offline and inference-based validation."""

    method_name: str
    model_name: str
    snapshot_path: Path
    dtype: torch.dtype | str
    prompt_settings: Any
    validation_fixed: dict[str, Any] = field(default_factory=dict)
    validation_selected: dict[str, Any] = field(default_factory=dict)
    selected_defaults: dict[str, Any] = field(default_factory=dict)
    artifacts: SteeringArtifacts = field(default_factory=SteeringArtifacts)
    method_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfflineValidationResult:
    """Method-specific outputs produced before generation search."""

    selected_params: dict[str, Any] = field(default_factory=dict)
    report_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationPlan:
    """Inference validation combinations and metadata."""

    param_sets: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSteeringModel:
    """Runtime model returned by prepare_model."""

    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase


@dataclass
class BatchContext:
    """Per-batch data passed to steering hooks."""

    rows: list[dict[str, Any]]
    prompt_texts: list[str]
    inputs: dict[str, torch.Tensor]
    prompt_lengths: list[int] | None = None


def decimal_range(start: float, end: float, step: float) -> list[float]:
    """Expand a numeric validation range without float drift."""

    if step <= 0:
        raise ValueError("Validation range step must be > 0.")

    current = Decimal(str(start))
    end_decimal = Decimal(str(end))
    step_decimal = Decimal(str(step))
    values: list[float] = []

    while current <= end_decimal:
        values.append(float(current))
        current += step_decimal
    return values


def expand_validation_spec(spec: Any) -> list[Any]:
    """Expand one compact validation config spec into candidate values."""

    if spec is None:
        return []
    if isinstance(spec, dict):
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list):
                raise TypeError("Validation 'values' entries must be lists.")
            return list(values)
        if {"min", "max", "step"} <= set(spec):
            return decimal_range(
                start=float(spec["min"]),
                end=float(spec["max"]),
                step=float(spec["step"]),
            )
        if "default" in spec:
            return [spec["default"]]
        return []
    return [spec]


def build_param_grid(
    validation_selected: dict[str, Any],
    selected_defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the Cartesian product for inference validation."""

    param_options: dict[str, list[Any]] = {}
    for name, spec in validation_selected.items():
        values = expand_validation_spec(spec)
        if not values and name in selected_defaults:
            values = [selected_defaults[name]]
        if values:
            param_options[name] = values

    for name, value in selected_defaults.items():
        param_options.setdefault(name, [value])

    if not param_options:
        return [{}]

    names = list(param_options)
    return [
        dict(zip(names, combo_values))
        for combo_values in product(*(param_options[name] for name in names))
    ]


class SteeringModel(ABC):
    """Base interface for AST, CAST, AdaSteer, AlphaSteer, etc."""

    method_name: str = ""

    @abstractmethod
    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """Train the method and return the saved artifacts i.e. refusal and condition vectors artifacts for CAST for instance"""

    @abstractmethod
    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Do one-time setup before the batch loop starts."""

    def offline_validation(
        self,
        context: ValidationContext,
        records: list[dict[str, Any]],
        batch_size_override: int | None = None,
    ) -> OfflineValidationResult:
        """Optional offline phase run before generation validation."""
        return OfflineValidationResult()

    def build_validation_plan(
        self,
        context: ValidationContext,
        offline_result: OfflineValidationResult,
    ) -> ValidationPlan:
        """Build the generation-validation combinations for this method."""
        merged_defaults = {
            **context.selected_defaults,
            **offline_result.selected_params,
        }
        return ValidationPlan(
            param_sets=build_param_grid(
                validation_selected=context.validation_selected,
                selected_defaults=merged_defaults,
            )
        )

    def validation_requires_fresh_model_per_combo(self) -> bool:
        """Whether validation should reload the base model for each combo."""
        return False

    def before_batch(
        self,
        prepared: PreparedSteeringModel,
        batch: BatchContext,
    ) -> None:
        """Optional hook run right before generation."""

    def generate_batch(
        self,
        prepared: PreparedSteeringModel,
        batch: BatchContext,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        """Optional override for methods with custom generation logic."""
        return prepared.model.generate(
            **batch.inputs,
            generation_config=generation_config,
        )

    def after_batch(
        self,
        prepared: PreparedSteeringModel,
        batch: BatchContext,
        outputs: torch.Tensor | None = None,
    ) -> None:
        """Optional hook run right after generation."""
