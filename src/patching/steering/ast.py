
import sys
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.utils import add_padding_token, get_model, get_tokenizer
from src.patching.steering.base import (
    BatchContext,
    InferenceContext,
    PreparedSteeringModel,
    SteeringArtifacts,
    SteeringModel,
    TrainingContext,
)
from src.patching.utils.utils import get_model_layer_list, parse_layers, resolve_dtype

REPO_ROOT = Path(__file__).resolve().parents[3]
CAST_TRAIN_DIR = REPO_ROOT / "benchmark_data" / "training_data" / "CAST"
CAST_REFUSAL_DIR = CAST_TRAIN_DIR / "refusal_vector"
DEFAULT_VECTOR_ROOT = REPO_ROOT / "vectors" / "CAST"
DEFAULT_MAX_QUESTIONS = 100
DEFAULT_MAX_SUFFIXES = 100


def _ensure_cast_vendor_path() -> None:
    path = Path(__file__).resolve().parent.parent / "CAST"
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_cast_vendor_path()

from src.patching.CAST.activation_steering.leash_layer import LeashLayer
from src.patching.CAST.activation_steering.malleable_model import MalleableModel
from src.patching.CAST.activation_steering.steering_dataset import SteeringDataset
from src.patching.CAST.activation_steering.steering_vector import SteeringVector


def _saved_svec(path: Path) -> Path:
    if str(path).endswith(".svec"):
        return path
    return path.with_name(f"{path.name}.svec")


def _vector_method(method_kwargs: dict[str, Any], key: str = "vector_method") -> str:
    return str(method_kwargs.get(key, method_kwargs.get("method", "pca_pairwise")))


def _resolve_dtype_value(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return resolve_dtype(str(value))


def _normalize_questions(questions: list[str | dict[str, Any]]) -> list[str]:
    normalized: list[str] = []
    for item in questions:
        if isinstance(item, str):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("question") or item.get("prompt")
            if text is None:
                raise KeyError("Question records must contain either 'question' or 'prompt'.")
            normalized.append(text)
            continue
        raise TypeError("questions must contain strings or dict records.")
    return normalized


def _load_refusal_suffix_pairs(
    behavior_refusal_json: Path,
    max_suffixes: int | None,
) -> list[tuple[str, str]]:
    payload = json.loads(behavior_refusal_json.read_text(encoding="utf-8"))
    refusal = payload["non_compliant_responses"]
    compliant = payload["compliant_responses"]
    count = min(len(refusal), len(compliant))
    if max_suffixes is not None:
        count = min(count, max_suffixes)
    return list(zip(refusal[:count], compliant[:count]))


def _default_refusal_paths() -> tuple[Path, Path]:
    return (
        CAST_REFUSAL_DIR / "alpaca_for_refusal.json",
        CAST_REFUSAL_DIR / "behavior_refusal.json",
    )


def build_malleable_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> MalleableModel:
    return MalleableModel(model=model, tokenizer=tokenizer)


def load_steering_vector(file_path: str) -> SteeringVector:
    return SteeringVector.load(file_path)


def set_runtime_prompt_lengths(prompt_lengths: Sequence[int] | None) -> None:
    # `LeashLayer` uses prompt lengths during generation to distinguish the original prompt tokens from newly generated tokens on a per-example basis.
    # AST updates this shared runtime state once per batch, right before calling `generate`, so each steering layer can apply its logic at the correct time.
    if prompt_lengths is None:
        LeashLayer.prompt_lengths = None
        return
    LeashLayer.prompt_lengths = [int(length) for length in prompt_lengths]


def reset_condition_runtime_state() -> None:
    # `LeashLayer` stores several pieces of mutable state across forward passes (whether a condition has already triggered, how many forwards happened, and the last computed similarities). 
    # We clear everything between batches so one request never leaks steering state into the next one.
    LeashLayer.condition_met.clear()
    LeashLayer.forward_calls.clear()
    LeashLayer.condition_similarities = defaultdict(lambda: defaultdict(float))
    LeashLayer.prompt_lengths = None


def train_refusal_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[str],
    refusal_suffix_pairs: Sequence[tuple[str, str]],
    batch_size: int = 32,
    method: str = "pca_pairwise",
    accumulate_last_x_tokens: int | str = "suffix-only",
    save_analysis: bool = False,
    output_dir: str = "analysis/refusal",
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> SteeringVector:
    # AST learns a single refusal direction by taking the same benign prompt on
    # both sides, then appending a refusal suffix to one side and a compliant
    # suffix to the other. When a system prompt is used, we pass the same prompt
    # to both sides of the contrast so the behavioral suffix is the main source
    # of difference between the paired examples.
    system_message = None if system_prompt is None else (system_prompt, system_prompt)

    # The dataset repeats each question on both sides of the pair and changes only  the suffixes.
    # This mirrors the original activation-steering training setup where the vector should capture the refusal/compliance behavioral difference, not differences in the user prompt itself.
    steering_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(question, question) for question in questions],
        suffixes=list(refusal_suffix_pairs),
        disable_suffixes=False,
        use_chat_template=use_chat_template,
        system_message=system_message,
    )

    # `SteeringVector.train` returns the learned refusal vector object, which can then be saved and reused at inference time to steer the refusal behavior in the model.
    return SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=steering_dataset,
        method=method,
        accumulate_last_x_tokens=accumulate_last_x_tokens,
        batch_size=batch_size,
        save_analysis=save_analysis,
        output_dir=output_dir,
    )


class ASTSteeringModel(SteeringModel):
    """AST steering with one refusal vector."""

    method_name = "ast"

    def validation_requires_fresh_model_per_combo(self) -> bool:
        """AST mutates model layers in place, so validation should start each combo from a fresh model."""
        return True

    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """Train and save the AST refusal vector."""
        # The framework may call `train` in modes where vectors were already produced earlier. In that case we simply report the expected output path and metadata without recomputing the refusal vector.
        if not bool(context.method_kwargs.get("train_behavior", True)):
            return SteeringArtifacts(
                paths={"behavior_vector": self._default_behavior_vector_out(context)},
                metadata={"vector_method": _vector_method(context.method_kwargs)},
            )

        # AST training : load the base model/tokenizer, build the refusal/compliance dataset, and learn refusal steering vector so  later validation/inference stages can consume it through the common API.
        model, tokenizer = self._load_runtime_model_and_tokenizer(context)
        return self._train_refusal_artifacts(
            model=model,
            tokenizer=tokenizer,
            context=context,
        )

    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Load the refusal vector and wrap the model."""
        # At inference time the base model is already available. This step only restores AST-specific artifacts i.e refusal vector and inserts the steering layers around the requested transformer blocks.
        behavior_vector, behavior_layer_ids = self._load_behavior_vector_and_layers(
            model=model,
            context=context,
        )
        strength = float(context.method_kwargs.get("strength", 1.5))
        apply_on_first_call = bool(context.method_kwargs.get("apply_on_first_call", True))

        # The returned `PreparedSteeringModel` keeps the framework interface uniform: downstream inference code can call `.model.generate(...)`
        # without caring whether the model is unsteered or AST-wrapped.
        steering_model = self._build_ast_model(
            model=model,
            tokenizer=tokenizer,
            behavior_vector=behavior_vector,
            behavior_layer_ids=behavior_layer_ids,
            strength=strength,
            apply_on_first_call=apply_on_first_call,
        )
        return PreparedSteeringModel(model=steering_model, tokenizer=tokenizer)

    def before_batch(
        self,
        prepared: PreparedSteeringModel,
        batch: BatchContext,
    ) -> None:
        """Reset runtime state before generation."""
        # AST stores batch-local state globally inside `LeashLayer`, so each new generation batch must start from a clean slate before the first forward.
        reset_condition_runtime_state()
        # Prompt lengths let the steering layers know where prompt tokens end for each example, which matters when condition checks should only react to generated continuation tokens.
        set_runtime_prompt_lengths(batch.prompt_lengths)

    def after_batch(
        self,
        prepared: PreparedSteeringModel,
        batch: BatchContext,
        outputs: torch.Tensor | None = None,
    ) -> None:
        """Clear runtime state after generation."""
        # Clean up again after generation so no condition flags or similarity values survive into the next batch if the same prepared model is reused.
        reset_condition_runtime_state()

    def _load_runtime_model_and_tokenizer(
        self,
        context: TrainingContext,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        dtype = _resolve_dtype_value(context.method_kwargs.get("dtype", "auto"))
        snapshot_path = str(context.snapshot_path)
        tokenizer = get_tokenizer(context.model_name, snapshot_path)
        tokenizer.padding_side = "left"
        tokenizer = add_padding_token(tokenizer, context.model_name)
        model = get_model(context.model_name, snapshot_path, dtype=dtype, device_map="auto")
        return model, tokenizer

    def _train_refusal_artifacts(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: TrainingContext,
    ) -> SteeringArtifacts:
        # AST training expects plain prompt formatting during vector extraction.
        # Adding a generation prompt would change the token layout seen by the dataset/training code, so we reject that configuration here.
        prompt_settings = context.prompt_settings
        if prompt_settings.add_generation_prompt:
            raise ValueError("AST training does not support add_generation_prompt=True.")

        # Use repo defaults unless the caller overrides them through `data_paths`.
        # `alpaca_json` provides benign prompts, while `behavior_refusal_json`provides aligned refusal/compliant suffix pairs that are appended those prompts to build the contrastive training pairs.
        # This training part of the refusal behavior is indetical to both AST and CAST as aligned in the 'activation-steering' official implementation
        alpaca_json, behavior_refusal_json = _default_refusal_paths()
        alpaca_json = Path(context.data_paths.get("alpaca_json", alpaca_json))
        behavior_refusal_json = Path(
            context.data_paths.get("behavior_refusal_json", behavior_refusal_json)
        )

        # Collect AST-specific training parameters from the generic framework context.
        # These control dataset size, the vector training method, and optional analysis outputs without changing the pipeline API.
        vector_method = _vector_method(context.method_kwargs)
        max_questions = context.method_kwargs.get("max_questions", DEFAULT_MAX_QUESTIONS)
        max_suffixes = context.method_kwargs.get("max_suffixes", DEFAULT_MAX_SUFFIXES)
        batch_size = int(context.method_kwargs.get("batch_size", 32))
        accumulate_last_x_tokens = context.method_kwargs.get(
            "refusal_accumulate_last_x_tokens",
            context.method_kwargs.get("accumulate_last_x_tokens", "suffix-only"),
        )
        save_analysis = bool(context.method_kwargs.get("save_analysis", False))

        # Keep artifact/vectors locations configurable, but fall back to deterministic paths so later validation/inference steps can rediscover the saved refusal vector from the same method/model combination.
        vector_out = Path(
            context.method_kwargs.get(
                "vector_out",
                context.artifacts_dir / "refusal_vectors" / f"{vector_method}_{context.model_name.split('/')[-1]}",
            )
        )
        analysis_out_dir = Path(
            context.method_kwargs.get(
                "refusal_analysis_out_dir",
                REPO_ROOT / "analysis" / f"refusal_PCA_{vector_method}_{context.model_name.split('/')[-1]}",
            )
        )

        # Load and normalize the benign prompt set. 
        # The helper accepts either raw strings or records with `question`/`prompt` fields so AST can reuse slightly different dataset formats.
        alpaca_data = json.loads(alpaca_json.read_text(encoding="utf-8"))
        raw_questions = alpaca_data["train"] if isinstance(alpaca_data, dict) else alpaca_data
        questions = _normalize_questions(raw_questions)
        if max_questions is not None:
            questions = questions[: int(max_questions)]

        # Refusal and compliant suffixes are paired positionally. 
        refusal_suffix_pairs = _load_refusal_suffix_pairs(
            behavior_refusal_json=behavior_refusal_json,
            max_suffixes=None if max_suffixes is None else int(max_suffixes),
        )

        # Delegate the actual vector estimation to the shared AST helper, then save the resulting steering vector in the format expected by `SteeringVector.load` during inference.
        vector = train_refusal_vector(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            refusal_suffix_pairs=refusal_suffix_pairs,
            batch_size=batch_size,
            method=vector_method,
            accumulate_last_x_tokens=accumulate_last_x_tokens,
            save_analysis=save_analysis,
            output_dir=str(analysis_out_dir),
            use_chat_template=prompt_settings.use_chat_template,
            system_prompt=prompt_settings.system_prompt,
        )
        vector.save(str(vector_out))

        # Return the artifact/vector location through the generic framework container so downstream stages do not need AST-specific path conventions.
        return SteeringArtifacts(
            paths={"behavior_vector": _saved_svec(vector_out)},
            metadata={"vector_method": vector_method},
        )

    def _resolve_behavior_vector_path(self, context: InferenceContext) -> Path:
       
        if "behavior_vector" in context.artifacts.paths:
            return context.artifacts.paths["behavior_vector"]
        # Fall back to a direct method-specific override when the vector is given as a raw path rather than wrapped in `SteeringArtifacts`.
        if "vector_path" in context.method_kwargs:
            return Path(context.method_kwargs["vector_path"])

        # Otherwise reconstruct the default training output path from the same method/model settings so inference can rediscover vectors produced by the training stage without hard-coding AST-specific paths elsewhere.
        training_context = TrainingContext(
            method_name=context.method_name,
            model_name=context.model_name,
            snapshot_path=Path(context.snapshot_path),
            artifacts_dir=DEFAULT_VECTOR_ROOT,
            prompt_settings=context.prompt_settings,
            method_kwargs=context.method_kwargs,
        )
        return self._default_behavior_vector_out(training_context)

    def _load_behavior_vector_and_layers(
        self,
        model: PreTrainedModel,
        context: InferenceContext,
    ) -> tuple[Any, list[int]]:
        # Load the saved AST refusal vector first; its stored directions tell us which layer IDs are actually available inside the artifact.
        vector_path = self._resolve_behavior_vector_path(context)
        behavior_vector = load_steering_vector(str(vector_path))
        
        # "auto" = middle layers of the model (minus first quarter and last quarter of the model layers)
        layer_spec = context.method_kwargs.get("layers", "auto")
        behavior_layer_ids = self._resolve_layer_ids(
            model=model,
            layer_spec=layer_spec,
            vector_layer_ids=set(behavior_vector.directions.keys()),
        )
        return behavior_vector, behavior_layer_ids

    def _build_ast_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        behavior_vector: Any,
        behavior_layer_ids: list[int],
        strength: float,
        apply_on_first_call: bool,
    ):
        # Wrap the base Hugging Face model with class `MalleableModel`, which exposes the `.steer(...)` API used to inject AST behavior vectors into selected transformer layers.
        steering_model = build_malleable_model(model=model, tokenizer=tokenizer)
        
        # AST uses only a refusal vector, with no separate condition vector. `strength` scales the learned direction, and `apply_on_first_call` controls whether steering is already active on the
        # first forward pass during generation.
        steering_model.steer(
            behavior_vector=behavior_vector,
            behavior_layer_ids=behavior_layer_ids,
            behavior_vector_strength=strength,
            apply_behavior_on_first_call=apply_on_first_call,
        )
        return steering_model

    def _default_behavior_vector_out(self, context: TrainingContext) -> Path:
        # Reconstruct the canonical save path from the training configuration so both training and inference agree on where the AST refusal vector lives when no explicit  path is provided.
        vector_method = _vector_method(context.method_kwargs)
        model_version = context.model_name.split("/")[-1]
        vector_root = Path(context.artifacts_dir)
        return _saved_svec(vector_root / "refusal_vectors" / f"{vector_method}_{model_version}")

    def _resolve_layer_ids(
        self,
        model: PreTrainedModel,
        layer_spec: str | Sequence[int],
        vector_layer_ids: set[int],
    ) -> list[int]:
        
        # Layer selection depends on both the model depth and the layers that were actually stored in the learned steering vector.
        num_layers = len(get_model_layer_list(model))
        
        if isinstance(layer_spec, str):
            # String specs such as `"auto"`, `"all"`, `"15-23"`, or comma-separated
            # lists are handled by the shared parser used across steering methods.
            return parse_layers(
                layer_spec=layer_spec,
                num_layers=num_layers,
                vector_layer_ids=vector_layer_ids,
            )
        
        resolved = [int(layer_id) for layer_id in layer_spec]
        invalid = [layer_id for layer_id in resolved if layer_id not in vector_layer_ids]
        if invalid:
            raise ValueError(f"Layer IDs are missing from the loaded steering vector: {invalid}")
        return resolved
