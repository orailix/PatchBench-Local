
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CONFIG_DIR = REPO_ROOT / "benchmark_configs"


def _resolve_fixed_value(value: Any, model_name: str | None) -> Any:
    """Resolve a parameter value that may specify 'fixed_by_model' and/or 'default'."""
    if isinstance(value, dict) and ("fixed_by_model" in value or "default" in value):
        fixed_by_model = value.get("fixed_by_model")
        if (
            model_name is not None
            and isinstance(fixed_by_model, dict)
            and model_name in fixed_by_model
        ):
            return fixed_by_model[model_name]
        if "default" in value:
            return value["default"]
        return None
    return value


def _resolve_section_dict(raw: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """Resolve section dictionary values for a specific model."""
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        resolved[key] = _resolve_fixed_value(value, model_name)
    return resolved


@dataclass(frozen=True)
class BenchmarkMethodConfig:
    """Human-readable benchmark config for one steering method."""

    method_name: str
    path: Path
    training_fixed: dict[str, Any] = field(default_factory=dict)
    validation_fixed: dict[str, Any] = field(default_factory=dict)
    validation_selected: dict[str, Any] = field(default_factory=dict)
    inference_fixed: dict[str, Any] = field(default_factory=dict)

    def fixed_params(self, phase_name: str, model_name: str | None = None) -> dict[str, Any]:
        if phase_name == "training":
            raw = self.training_fixed
        elif phase_name == "validation":
            raw = self.validation_fixed
        elif phase_name == "inference":
            raw = self.inference_fixed
        else:
            raise ValueError(
                f"Unsupported phase {phase_name!r}. Expected 'training', 'validation', or 'inference'."
            )
        return _resolve_section_dict(raw, model_name=model_name)

    def selected_params(self, model_name: str | None = None) -> dict[str, Any]:
        return _selection_defaults(self.validation_selected, model_name=model_name)

    def for_model(self, model_name: str) -> "BenchmarkMethodConfig":
        """Return a copy of this config specialized for model_name."""
        training_resolved = _resolve_section_dict(self.training_fixed, model_name=model_name)
        validation_fixed_resolved = _resolve_section_dict(self.validation_fixed, model_name=model_name)
        inference_resolved = _resolve_section_dict(self.inference_fixed, model_name=model_name)

        selected_resolved: dict[str, Any] = {}
        for name, spec in self.validation_selected.items():
            if isinstance(spec, dict):
                fixed_values = spec.get("fixed_by_model")
                if isinstance(fixed_values, dict) and model_name in fixed_values:
                    val = fixed_values[model_name]
                    spec = {**spec, "default": val, "values": [val]}
            selected_resolved[name] = spec

        return BenchmarkMethodConfig(
            method_name=self.method_name,
            path=self.path,
            training_fixed=training_resolved,
            validation_fixed=validation_fixed_resolved,
            validation_selected=selected_resolved,
            inference_fixed=inference_resolved,
        )


def default_benchmark_config_path(method_name: str) -> Path:
    """Return the default YAML config path for a steering method."""

    return BENCHMARK_CONFIG_DIR / f"{method_name}.yaml"


def load_benchmark_method_config(
    method_name: str,
    config_path: str | Path | None = None,
    model_name: str | None = None,
) -> BenchmarkMethodConfig:
    """Load one benchmark config YAML."""

    path = default_benchmark_config_path(method_name) if config_path is None else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")

    payload = _load_config_payload(path)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Benchmark config must be a YAML mapping: {path}")

    config_method_name = str(payload.get("method", method_name))
    if config_method_name != method_name:
        raise ValueError(
            f"Benchmark config {path} declares method={config_method_name!r}, "
            f"but {method_name!r} was requested."
        )

    config = BenchmarkMethodConfig(
        method_name=config_method_name,
        path=path,
        training_fixed=_dict_section(payload.get("training_fixed"), section_name="training_fixed"),
        validation_fixed=_dict_section(payload.get("validation_fixed"), section_name="validation_fixed"),
        validation_selected=_dict_section(payload.get("validation_selected"), section_name="validation_selected"),
        inference_fixed=_dict_section(payload.get("inference_fixed"), section_name="inference_fixed"),
    )
    if model_name is not None:
        return config.for_model(model_name)
    return config


def load_benchmark_method_config_from_path(
    config_path: str | Path,
    model_name: str | None = None,
) -> BenchmarkMethodConfig:
    """Load one benchmark config and infer the method from its contents."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")

    payload = _load_config_payload(path)
    if payload is None or not isinstance(payload, dict):
        raise TypeError(f"Benchmark config must be a YAML mapping: {path}")

    method_name = payload.get("method")
    if not isinstance(method_name, str) or not method_name:
        raise ValueError(f"Benchmark config must define a non-empty 'method' field: {path}")
    return load_benchmark_method_config(method_name=method_name, config_path=path, model_name=model_name)


def _dict_section(payload: Any, *, section_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Section {section_name!r} must be a mapping.")
    return dict(payload)


def _selection_defaults(specs: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, spec in specs.items():
        if isinstance(spec, dict):
            fixed_by_model = spec.get("fixed_by_model")
            if (
                model_name is not None
                and isinstance(fixed_by_model, dict)
                and model_name in fixed_by_model
            ):
                defaults[name] = fixed_by_model[model_name]
            elif "default" in spec:
                defaults[name] = spec["default"]
            continue
        if spec is not None:
            defaults[name] = spec
    return defaults


def _load_config_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)
