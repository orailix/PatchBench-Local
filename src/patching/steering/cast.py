import gc
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.utils import add_padding_token, get_model, get_tokenizer, safe_name
from src.patching.steering.ast import (
    ASTSteeringModel,
    LeashLayer,
    REPO_ROOT,
    SteeringDataset,
    SteeringVector,
    _saved_svec,
    _vector_method,
    build_malleable_model,
    load_steering_vector,
    reset_condition_runtime_state,
    set_runtime_prompt_lengths,
)
from src.patching.steering.base import (
    InferenceContext,
    OfflineValidationResult,
    PreparedSteeringModel,
    SteeringArtifacts,
    TrainingContext,
    ValidationContext,
    ValidationPlan,
    build_param_grid,
    decimal_range,
    expand_validation_spec,
)
from src.patching.utils import format_user_prompt
from src.patching.utils.utils import get_model_layer_list

CAST_TRAIN_DIR = REPO_ROOT / "benchmark_data" / "training_data" / "CAST"
CAST_CONDITION_DIR = CAST_TRAIN_DIR / "condition_vector"
DEFAULT_VECTOR_ROOT = REPO_ROOT / "vectors" / "CAST"

POSITIVE_GATE_CATEGORIES = {"original", "close_harmful"}
NEGATIVE_GATE_CATEGORIES = {"benign_harmful_words", "benign_same_structure"}
GATE_PARAM_NAMES = {"condition_layers", "condition_threshold", "condition_direction"}


def _load_condition_pairs(
    condition_json: Path,
    split: str | None,
    max_examples: int | None,
) -> tuple[list[str], list[str]]:
    # CAST condition data can be stored either as a plain list of records or as a dict of dataset splits.
    payload = json.loads(condition_json.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        if split is not None and split in payload:
            records = payload[split]
        elif split is None and "train" in payload:
            records = payload["train"]
        else:
            raise ValueError(f"Unable to resolve split={split!r} in condition JSON: {condition_json}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise TypeError("Condition JSON must be either a list or a dict of split -> records.")

    positives = [record["unsafe"] for record in records]
    negatives = [record["safe"] for record in records]
    count = min(len(positives), len(negatives))
    if max_examples is not None:
        count = min(count, max_examples)
    return positives[:count], negatives[:count]


def train_condition_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    positive_strings: list[str],
    negative_strings: list[str],
    batch_size: int = 32,
    method: str = "pca_pairwise",
    accumulate_last_x_tokens: int | str = "all",
    save_analysis: bool = False,
    output_dir: str = "analysis/cast_condition",
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> SteeringVector:
    system_message = None if system_prompt is None else (system_prompt, system_prompt)

    steering_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=list(zip(positive_strings, negative_strings)),
        suffixes=None,
        disable_suffixes=True,
        use_chat_template=use_chat_template,
        system_message=system_message,
    )

    return SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=steering_dataset,
        batch_size=batch_size,
        method=method,
        save_analysis=save_analysis,
        output_dir=output_dir,
        accumulate_last_x_tokens=accumulate_last_x_tokens,
    )


def _split_gate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int], dict[str, int]]:
    filtered_records: list[dict[str, Any]] = []
    labels: list[int] = []
    counts: Counter[str] = Counter()

    for row in records:
        category = row.get("category")
        if category in POSITIVE_GATE_CATEGORIES:
            filtered_records.append(row)
            labels.append(1)
            counts[str(category)] += 1
        elif category in NEGATIVE_GATE_CATEGORIES:
            filtered_records.append(row)
            labels.append(0)
            counts[str(category)] += 1

    if not filtered_records:
        raise ValueError("No CAST gate-search records were found in the validation input.")
    return filtered_records, labels, dict(counts)


def _search_condition_point(
    similarity_rows: list[dict[str, Any]],
    labels: list[int],
    candidate_layer_sets: list[list[int]],
    thresholds: list[float],
    directions: list[str],
    sample_weights: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(similarity_rows) != len(labels):
        raise ValueError("Gate-search similarities and labels must have the same length.")
    if len(sample_weights) != len(labels):
        raise ValueError("Gate-search sample weights and labels must have the same length.")

    best_result: dict[str, Any] | None = None
    search_results: list[dict[str, Any]] = []

    for layer_set in candidate_layer_sets:
        for threshold in thresholds:
            for direction in directions:
                y_pred = []
                for row in similarity_rows:
                    sims = row["similarities"]
                    condition_met = any(
                        (float(sims[str(layer)]) > threshold) == (direction == "smaller")
                        for layer in layer_set
                    )
                    y_pred.append(1 if condition_met else 0)

                precision = float(precision_score(labels, y_pred, zero_division=0))
                recall = float(recall_score(labels, y_pred, zero_division=0))
                f1 = float(f1_score(labels, y_pred, zero_division=0))
                weighted_precision = float(
                    precision_score(labels, y_pred, sample_weight=sample_weights, zero_division=0)
                )
                weighted_recall = float(
                    recall_score(labels, y_pred, sample_weight=sample_weights, zero_division=0)
                )
                weighted_f1 = float(f1_score(labels, y_pred, sample_weight=sample_weights, zero_division=0))
                result = {
                    "params": {
                        "condition_layers": list(layer_set),
                        "condition_threshold": threshold,
                        "condition_direction": direction,
                    },
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "weighted_precision": weighted_precision,
                        "weighted_recall": weighted_recall,
                        "weighted_f1": weighted_f1,
                    },
                }
                search_results.append(result)
                if best_result is None or weighted_f1 > best_result["metrics"]["weighted_f1"]:
                    best_result = result

    if best_result is None:
        raise RuntimeError("CAST gate search did not evaluate any combinations.")
    return best_result, search_results


def _build_gate_sample_weights(
    records: list[dict[str, Any]],
    validation_fixed: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    weighting_mode = str(validation_fixed.get("gate_weighting", "manual_category")).lower()
    category_counts = Counter(str(row.get("category")) for row in records)
    if not category_counts:
        raise ValueError("Cannot build CAST gate sample weights without gate records.")

    if weighting_mode == "category_balanced":
        target_category_weight = len(records) / len(category_counts)
        category_weights = {
            category: target_category_weight / count
            for category, count in category_counts.items()
        }
    elif weighting_mode == "manual_category":
        configured_weights = validation_fixed.get("gate_category_weights")
        if not isinstance(configured_weights, dict):
            raise ValueError("manual_category gate weighting requires validation_fixed.gate_category_weights.")
        missing_categories = sorted(category for category in category_counts if category not in configured_weights)
        if missing_categories:
            raise ValueError(f"Missing manual CAST gate weights for categories: {missing_categories}")
        category_weights = {
            category: float(configured_weights[category])
            for category in category_counts
        }
    else:
        raise ValueError("validation_fixed.gate_weighting must be one of: manual_category, category_balanced.")

    weights = [category_weights[str(row.get("category"))] for row in records]
    metadata = {
        "enabled": True,
        "strategy": weighting_mode,
        "category_weights": category_weights,
    }
    return weights, metadata


class CASTSteeringModel(ASTSteeringModel):
    """CAST steering with refusal and condition vectors."""

    method_name = "cast"

    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """Train and save CAST artifacts."""
        train_behavior = bool(context.method_kwargs.get("train_behavior", True))
        model, tokenizer = self._load_runtime_model_and_tokenizer(context)

        if train_behavior:
            behavior_artifacts = self._train_refusal_artifacts(
                model=model,
                tokenizer=tokenizer,
                context=context,
            )
        else:
            behavior_artifacts = SteeringArtifacts(
                paths={"behavior_vector": self._default_behavior_vector_out(context)},
                metadata={"vector_method": _vector_method(context.method_kwargs)},
            )
        condition_artifacts = self._train_condition_artifacts(
            model=model,
            tokenizer=tokenizer,
            context=context,
        )
        return SteeringArtifacts(
            paths={**behavior_artifacts.paths, **condition_artifacts.paths},
            metadata={**behavior_artifacts.metadata, **condition_artifacts.metadata},
        )

    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Load CAST artifacts and wrap the model."""
        behavior_vector, behavior_layer_ids = self._load_behavior_vector_and_layers(
            model=model,
            context=context,
        )
        condition_vector, condition_layer_ids, threshold, direction = (
            self._load_condition_runtime_config(model=model, context=context)
        )

        strength = float(context.method_kwargs.get("strength", 1.5))
        comparison_mode = str(context.method_kwargs.get("condition_comparison_mode", "mean"))
        apply_on_first_call = bool(context.method_kwargs.get("apply_on_first_call", True))

        steering_model = self._build_ast_model(
            model=model,
            tokenizer=tokenizer,
            behavior_vector=behavior_vector,
            behavior_layer_ids=behavior_layer_ids,
            strength=strength,
            apply_on_first_call=apply_on_first_call,
        )
        steering_model.reset_leash_to_default()
        steering_model.steer(
            behavior_vector=behavior_vector,
            behavior_layer_ids=behavior_layer_ids,
            behavior_vector_strength=strength,
            condition_vector=condition_vector,
            condition_layer_ids=condition_layer_ids,
            condition_vector_threshold=threshold,
            condition_comparator_threshold_is=direction,
            condition_threshold_comparison_mode=comparison_mode,
            apply_behavior_on_first_call=apply_on_first_call,
        )
        return PreparedSteeringModel(model=steering_model, tokenizer=tokenizer)

    def offline_validation(
        self,
        context: ValidationContext,
        records: list[dict[str, Any]],
        batch_size_override: int | None = None,
    ) -> OfflineValidationResult:
        """Run CAST gate-search before generation-based validation."""
        if not bool(context.validation_fixed.get("find_condition_point", True)):
            selected_params = {
                key: context.selected_defaults[key]
                for key in GATE_PARAM_NAMES
                if key in context.selected_defaults
            }
            return OfflineValidationResult(
                selected_params=selected_params,
                report_payload={
                    "type": "cast_gate_search",
                    "enabled": False,
                    "best_gate_params": selected_params,
                    "best_gate_metrics": {},
                    "results": [],
                },
            )

        gate_records, gate_labels, category_counts = _split_gate_records(records)
        candidate_layer_sets = self._expand_condition_layers(context.validation_fixed)
        thresholds = self._expand_thresholds(context.validation_fixed)
        directions = self._expand_directions(context.validation_selected)
        search_layers = sorted({layer for layer_set in candidate_layer_sets for layer in layer_set})

        similarity_rows, runtime_metadata = self._collect_condition_similarities(
            context=context,
            records=gate_records,
            search_layers=search_layers,
            batch_size_override=batch_size_override,
        )
        valid_layer_set = set(runtime_metadata["searched_layers"])
        candidate_layer_sets = [
            layer_set
            for layer_set in candidate_layer_sets
            if all(layer in valid_layer_set for layer in layer_set)
        ]
        if not candidate_layer_sets:
            raise ValueError("No CAST gate-search layer combinations remain after validating layers.")

        sample_weights, sample_weighting_metadata = _build_gate_sample_weights(
            records=gate_records,
            validation_fixed=context.validation_fixed,
        )
        best_gate_result, gate_search_results = _search_condition_point(
            similarity_rows=similarity_rows,
            labels=gate_labels,
            candidate_layer_sets=candidate_layer_sets,
            thresholds=thresholds,
            directions=directions,
            sample_weights=sample_weights,
        )
        selected_params = dict(best_gate_result["params"])
        report_payload = {
            "type": "cast_gate_search",
            "enabled": True,
            "positive_categories": sorted(POSITIVE_GATE_CATEGORIES),
            "negative_categories": sorted(NEGATIVE_GATE_CATEGORIES),
            "category_counts": category_counts,
            "num_records": len(gate_labels),
            "search_space": {
                "layer_range_start": context.validation_fixed.get("layer_range_start"),
                "layer_range_end": context.validation_fixed.get("layer_range_end"),
                "max_layers_to_combine": context.validation_fixed.get("max_layers_to_combine"),
                "threshold_min": context.validation_fixed.get("threshold_min"),
                "threshold_max": context.validation_fixed.get("threshold_max"),
                "threshold_step": context.validation_fixed.get("threshold_step"),
                "directions": directions,
            },
            "runtime": runtime_metadata,
            "selection_metric": "weighted_f1",
            "sample_weighting": sample_weighting_metadata,
            "best_gate_params": selected_params,
            "best_gate_metrics": best_gate_result["metrics"],
            "results": gate_search_results,
        }
        return OfflineValidationResult(
            selected_params=selected_params,
            report_payload=report_payload,
        )

    def build_validation_plan(
        self,
        context: ValidationContext,
        offline_result: OfflineValidationResult,
    ) -> ValidationPlan:
        """Validate only the inference-time CAST params after gate-search."""
        generation_selected = {
            key: value
            for key, value in context.validation_selected.items()
            if key not in GATE_PARAM_NAMES
        }
        generation_defaults = {
            key: value
            for key, value in context.selected_defaults.items()
            if key not in GATE_PARAM_NAMES
        }
        param_grid = build_param_grid(
            validation_selected=generation_selected,
            selected_defaults=generation_defaults,
        )
        gate_params = dict(offline_result.selected_params)
        param_sets = [{**gate_params, **params} for params in param_grid] or [gate_params]
        return ValidationPlan(
            param_sets=param_sets,
            metadata={"offline_validation": offline_result.report_payload},
        )

    def _train_condition_artifacts(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: TrainingContext,
    ) -> SteeringArtifacts:
        prompt_settings = context.prompt_settings
        if prompt_settings.add_generation_prompt:
            raise ValueError("CAST training does not support add_generation_prompt=True.")

        vector_method = _vector_method(context.method_kwargs, key="condition_vector_method")
        batch_size = int(context.method_kwargs.get("batch_size", 32))
        max_examples = context.method_kwargs.get("max_condition_examples")
        if max_examples is not None:
            max_examples = int(max_examples)
        accumulate_last_x_tokens = context.method_kwargs.get(
            "condition_accumulate_last_x_tokens",
            "all",
        )
        save_analysis = bool(context.method_kwargs.get("save_analysis", False))
        condition_split = context.method_kwargs.get("condition_split", "train")

        condition_json = self._resolve_condition_json(context)
        model_version = context.model_name.split("/")[-1]
        vector_out = Path(
            context.method_kwargs.get(
                "condition_vector_out",
                context.artifacts_dir / "condition_vectors" / f"{vector_method}_{model_version}",
            )
        )
        analysis_out_dir = Path(
            context.method_kwargs.get(
                "condition_analysis_out_dir",
                REPO_ROOT / "analysis" / f"cast_condition_{vector_method}_{model_version}",
            )
        )

        positive_strings, negative_strings = _load_condition_pairs(
            condition_json=condition_json,
            split=condition_split,
            max_examples=max_examples,
        )
        vector = train_condition_vector(
            model=model,
            tokenizer=tokenizer,
            positive_strings=positive_strings,
            negative_strings=negative_strings,
            batch_size=batch_size,
            method=vector_method,
            accumulate_last_x_tokens=accumulate_last_x_tokens,
            save_analysis=save_analysis,
            output_dir=str(analysis_out_dir),
            use_chat_template=prompt_settings.use_chat_template,
            system_prompt=prompt_settings.system_prompt,
        )
        vector.save(str(vector_out))

        paths = {"condition_vector": _saved_svec(vector_out)}
        metadata: dict[str, Any] = {"condition_vector_method": vector_method}
        return SteeringArtifacts(paths=paths, metadata=metadata)

    def _resolve_condition_json(self, context: TrainingContext) -> Path:
        if "condition_json" in context.data_paths:
            return Path(context.data_paths["condition_json"])

        model_slug = safe_name(context.model_name)
        candidates = sorted(CAST_CONDITION_DIR.glob(f"{model_slug}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No CAST condition JSON found for model {context.model_name!r} in {CAST_CONDITION_DIR}."
            )
        if len(candidates) > 1:
            joined = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"Ambiguous CAST condition JSON for model {context.model_name!r}: {joined}."
            )
        return candidates[0]

    def _load_condition_runtime_config(
        self,
        model: PreTrainedModel,
        context: InferenceContext,
    ) -> tuple[Any, list[int], float, str]:
        condition_vector_path = self._resolve_condition_vector_path(context)
        condition_vector = load_steering_vector(str(condition_vector_path))

        layer_spec = context.method_kwargs.get("condition_layers")
        if layer_spec is None:
            layer_spec = context.artifacts.metadata.get("condition_layer_ids")
        if layer_spec is None:
            raise ValueError("CAST requires condition layers at inference time.")

        threshold = context.method_kwargs.get("condition_threshold")
        if threshold is None:
            threshold = context.artifacts.metadata.get("condition_threshold")
        if threshold is None:
            raise ValueError("CAST requires a condition threshold at inference time.")

        direction = str(
            context.method_kwargs.get(
                "condition_direction",
                context.artifacts.metadata.get("condition_direction", "larger"),
            )
        )
        condition_layer_ids = self._resolve_layer_ids(
            model=model,
            layer_spec=layer_spec,
            vector_layer_ids=set(condition_vector.directions.keys()),
        )
        return condition_vector, condition_layer_ids, float(threshold), direction

    def _resolve_condition_vector_path(self, context: InferenceContext | ValidationContext) -> Path:
        if "condition_vector" in context.artifacts.paths:
            return context.artifacts.paths["condition_vector"]
        if "condition_vector_path" in context.method_kwargs:
            return Path(context.method_kwargs["condition_vector_path"])

        vector_method = _vector_method(context.method_kwargs, key="condition_vector_method")
        model_version = context.model_name.split("/")[-1]
        return DEFAULT_VECTOR_ROOT / "condition_vectors" / f"{vector_method}_{model_version}.svec"

    def _expand_condition_layers(self, validation_fixed: dict[str, Any]) -> list[list[int]]:
        start_layer = int(validation_fixed.get("layer_range_start", 1))
        end_layer = int(validation_fixed.get("layer_range_end", start_layer))
        max_layers_to_combine = int(validation_fixed.get("max_layers_to_combine", 1))

        if end_layer < start_layer:
            raise ValueError("validation_fixed.layer_range_end must be >= layer_range_start.")
        if max_layers_to_combine < 1:
            raise ValueError("validation_fixed.max_layers_to_combine must be >= 1.")

        available_layers = list(range(start_layer, end_layer + 1))
        all_layer_sets: list[list[int]] = []
        max_group_size = min(max_layers_to_combine, len(available_layers))
        for group_size in range(1, max_group_size + 1):
            for layer_group in itertools.combinations(available_layers, group_size):
                all_layer_sets.append(list(layer_group))
        return all_layer_sets

    def _expand_thresholds(self, validation_fixed: dict[str, Any]) -> list[float]:
        return decimal_range(
            start=float(validation_fixed.get("threshold_min", 0.0)),
            end=float(validation_fixed.get("threshold_max", 0.06)),
            step=float(validation_fixed.get("threshold_step", 0.01)),
        )

    def _expand_directions(self, validation_selected: dict[str, Any]) -> list[str]:
        values = expand_validation_spec(validation_selected.get("condition_direction"))
        directions = [str(value) for value in values]
        if directions:
            return directions
        return ["larger", "smaller"]

    def _collect_condition_similarities(
        self,
        *,
        context: ValidationContext,
        records: list[dict[str, Any]],
        search_layers: list[int],
        batch_size_override: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        comparison_mode = str(context.method_kwargs.get("condition_comparison_mode", "mean"))
        condition_vector_path = self._resolve_condition_vector_path(context)
        if not condition_vector_path.exists():
            raise FileNotFoundError(
                f"CAST condition vector not found: {condition_vector_path}. "
                "Run CAST training before validation."
            )

        tokenizer = get_tokenizer(context.model_name, str(context.snapshot_path))
        tokenizer.padding_side = "left"
        tokenizer = add_padding_token(tokenizer, context.model_name)
        model = get_model(
            context.model_name,
            str(context.snapshot_path),
            dtype=context.dtype,
            device_map="auto",
        )
        steering_model = build_malleable_model(model=model, tokenizer=tokenizer)

        try:
            condition_vector = load_steering_vector(str(condition_vector_path))
            num_layers = len(get_model_layer_list(model))
            valid_layers = sorted(
                layer
                for layer in search_layers
                if layer in condition_vector.directions and 0 <= layer < num_layers
            )
            if not valid_layers:
                raise ValueError(
                    "No valid CAST condition layers remain after intersecting config, vector, and model."
                )

            steering_model.steer(
                behavior_vector=None,
                condition_vector=condition_vector,
                condition_layer_ids=valid_layers,
                condition_vector_threshold=1.0,
                condition_comparator_threshold_is="smaller",
                condition_threshold_comparison_mode=comparison_mode,
                apply_behavior_on_first_call=False,
            )

            batch_size = int(batch_size_override or context.method_kwargs.get("batch_size", 1))
            if batch_size < 1:
                raise ValueError("Validation batch size must be >= 1.")

            similarity_rows: list[dict[str, Any]] = []
            for batch_start in range(0, len(records), batch_size):
                batch = records[batch_start : batch_start + batch_size]
                prompts_for_model = [
                    format_user_prompt(
                        tokenizer=tokenizer,
                        user_prompt=row["prompt"],
                        settings=context.prompt_settings,
                    )
                    for row in batch
                ]
                inputs = tokenizer(
                    prompts_for_model,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(context.method_kwargs.get("max_input_tokens", 4096)),
                )
                prompt_lengths = None
                if "attention_mask" in inputs:
                    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                inputs = inputs.to(steering_model.device)

                reset_condition_runtime_state()
                set_runtime_prompt_lengths(prompt_lengths)
                with torch.no_grad():
                    steering_model(**inputs)

                for row_index, row in enumerate(batch):
                    similarity_rows.append(
                        {
                            "id": row.get("id", batch_start + row_index),
                            "category": row.get("category"),
                            "prompt": row["prompt"],
                            "similarities": {
                                str(layer): float(LeashLayer.condition_similarities[row_index].get(layer, 0.0))
                                for layer in valid_layers
                            },
                        }
                    )
                reset_condition_runtime_state()

            metadata = {
                "condition_vector_path": str(condition_vector_path.resolve()),
                "comparison_mode": comparison_mode,
                "searched_layers": valid_layers,
            }
            return similarity_rows, metadata

        finally:
            reset_fn = getattr(steering_model, "reset_leash_to_default", None)
            if callable(reset_fn):
                reset_fn()
            reset_condition_runtime_state()
            del steering_model
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
