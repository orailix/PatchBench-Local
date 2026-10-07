import argparse
import typing
from dataclasses import dataclass

import torch

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Optional

# (alpha_layer ~25%, beta_layer ~50%) per model
_ADASTEER_LAYER_SPECS: dict[str, tuple[int, int]] = {
    "Qwen/Qwen2.5-3B-Instruct":                   (9,  18),  # 36 layers
    "Qwen/Qwen2.5-14B-Instruct":                  (12, 24),  # 48 layers
    "mistralai/Mistral-7B-Instruct-v0.3":         (8,  16),  # 32 layers
    # "mistralai/Ministral-3-14B-Instruct-2512":    (10, 20),  # ~40 layers
    "google/gemma-3-4b-it":                       (9,  18),  # 34 layers
    "google/gemma-3-12b-it":                      (11,  23),  # 46 layers
    "meta-llama/Llama-3.1-8B-Instruct":           (8,  16),  # 32 layers
    "meta-llama/Llama-4-Scout-17B-16E-Instruct":  (12, 24),  # 48 layers
}


def get_adasteer_layers(model_name: str) -> tuple[int, int]:
    """Return (alpha_layer, beta_layer) for the given model (~25% and ~50% depth)."""
    if model_name not in _ADASTEER_LAYER_SPECS:
        raise ValueError(
            f"No AdaSteer layer spec for model '{model_name}'. "
            f"Add it to _ADASTEER_LAYER_SPECS in src/patching/utils/utils.py."
        )
    return _ADASTEER_LAYER_SPECS[model_name]


@dataclass(frozen=True)
class PromptTemplateSettings:
    # Whether prompts are formatted with tokenizer.apply_chat_template.
    use_chat_template: bool
    # Optional system message prepended before the user message.
    system_prompt: str | None
    # Passed directly to tokenizer.apply_chat_template.
    add_generation_prompt: bool


def add_prompt_template_cli_args(
    parser: argparse.ArgumentParser,
    default_use_chat_template: bool,
    default_system_prompt: str | None,
    default_add_generation_prompt: bool,
) -> None:
    parser.add_argument(
        "--use-chat-template",
        dest="use_chat_template",
        action="store_true",
        default=default_use_chat_template,
        help=f"Format prompts with tokenizer chat template (default: {default_use_chat_template}).",
    )
    parser.add_argument(
        "--no-chat-template",
        dest="use_chat_template",
        action="store_false",
        help="Do not use tokenizer chat template; use raw prompt text.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=default_system_prompt,
        help=f"System prompt text (default: {default_system_prompt!r}).",
    )
    parser.add_argument(
        "--no-system-prompt",
        dest="system_prompt",
        action="store_const",
        const=None,
        help="Disable system prompt.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        dest="add_generation_prompt",
        action="store_true",
        default=default_add_generation_prompt,
        help=f"Use add_generation_prompt=True in chat templating (default: {default_add_generation_prompt}).",
    )
    parser.add_argument(
        "--no-add-generation-prompt",
        dest="add_generation_prompt",
        action="store_false",
        help="Use add_generation_prompt=False in chat templating.",
    )


def resolve_prompt_template_settings(args: argparse.Namespace) -> PromptTemplateSettings:
    return PromptTemplateSettings(
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
        add_generation_prompt=args.add_generation_prompt,
    )


def format_user_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    settings: PromptTemplateSettings,
) -> str:
    if not settings.use_chat_template:
        return user_prompt

    messages = []
    if settings.system_prompt:
        messages.append({"role": "system", "content": settings.system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=settings.add_generation_prompt,
    )

def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def parse_accumulate_last_x_tokens(value: str) -> int | str:
    if value in {"all", "suffix-only"}:
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "accumulate_last_x_tokens must be 'all', 'suffix-only', or a positive integer."
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("accumulate_last_x_tokens integer value must be >= 1.")
    return parsed


def _resolve_attr_path(root: object, path: str) -> object | None:
    # Resolve nested attributes like "model.decoder.layers" and return None when any step is missing.
    current = root
    for attr in path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _coerce_layer_container(candidate: object) -> typing.Sequence[torch.nn.Module] | None:
    # Normalize supported container types into a plain list of torch modules.
    if isinstance(candidate, torch.nn.ModuleList):
        return list(candidate)
    if isinstance(candidate, (list, tuple)):
        if len(candidate) > 0 and isinstance(candidate[0], torch.nn.Module):
            return list(candidate)
    return None


def get_model_layer_list(model: PreTrainedModel) -> typing.Sequence[torch.nn.Module]:
    # Return the main transformer block list for many model wrappers and architectures.
    roots = [
        model,
        getattr(model, "base_model", None),
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "text_model", None),
    ]
    paths = [
        "layers",
        "h",
        "blocks",
        "transformer.h",
        "transformer.layers",
        "model.layers",
        "decoder.layers",
        "model.decoder.layers",
    ]

    for root in roots:
        if root is None:
            continue
        for path in paths:
            layers = _coerce_layer_container(_resolve_attr_path(root, path))
            if layers is not None:
                return layers

    # Fallback: scan named modules for typical layer container names.
    for module_name, module in model.named_modules():
        if module_name.split(".")[-1] in {"layers", "h", "blocks"}:
            if "vision" in module_name or "visual" in module_name:
                continue
            layers = _coerce_layer_container(module)
            if layers is not None:
                return layers

    raise AttributeError(f"Unable to locate transformer layers on model type: {type(model)}")


# Deduce which layers to steer based on the provided layer spec, number of model layers, and available vector layers. The layer spec can be "auto", "all", a range like "15-23", or a comma-separated list of layer indices.
# The function returns a list of valid layer indices to apply the steering vector to.
def parse_layers(
    layer_spec: str,
    num_layers: int,
    vector_layer_ids: set[int],
) -> list[int]:

    shared_layers = sorted([layer for layer in vector_layer_ids if 0 <= layer < num_layers])
    if not shared_layers:
        raise ValueError("No overlapping layer IDs between model and loaded steering vector.")

    if layer_spec == "auto":
        if num_layers <= 2:
            return shared_layers

        # Keep the middle 1/2 of layers (drop ~1/4 at the start and ~1/4 at the end),
        # then keep only layers that exist in the loaded vector.
        trim = int(round(num_layers / 4))
        start = max(1, trim)
        end = min(num_layers - 1, num_layers - trim)

        if end <= start:
            auto_candidates = list(range(1, num_layers - 1))
        else:
            auto_candidates = list(range(start, end))

        auto_layers = [layer for layer in auto_candidates if layer in shared_layers]
        if auto_layers:
            return auto_layers
        return shared_layers

    if layer_spec == "all":
        return shared_layers

    parsed: list[int] = []
    chunks = [chunk.strip() for chunk in layer_spec.split(",") if chunk.strip()]
    # Parse each chunk of the layer spec. 
    # If it contains a "-", treat it as a range; otherwise, treat it as a single layer index. 
    # This allows for flexible specifications like "15-23, 30, 35-37".
    for chunk in chunks:
        if "-" in chunk:
            left, right = chunk.split("-", maxsplit=1)
            start = int(left)
            end = int(right)
            if start <= end:
                parsed.extend(range(start, end + 1))
            else:
                parsed.extend(range(start, end - 1, -1))
        else:
            parsed.append(int(chunk))

    normalized: list[int] = []
    # Normalize the parsed layer indices, handling negative values and filtering out invalid or non-overlapping layers. 
    # Negative indices are treated as offsets from the end of the layer list.
    for layer in parsed:
        candidate = layer if layer >= 0 else num_layers + layer
        if 0 <= candidate < num_layers and candidate in vector_layer_ids and candidate not in normalized:
            normalized.append(candidate)

    if not normalized:
        raise ValueError("No valid layers selected. Check --layers and vector/model compatibility.")
    
    return normalized

def get_last_valid_token_index(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    Return the index of the last non-PAD token for each sample in the batch.

    Supports 2-D padding masks (1 = valid, 0 = pad) and 4-D additive masks
    (0 = valid, -inf = masked) as produced by recent HuggingFace models.
    Falls back to the last position when no mask is provided.
    """
    if attention_mask is None:
        return torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)

    if attention_mask.dim() == 4:
        # 4-D causal/additive mask: shape (B, heads, T, T).
        # The last row of the last head encodes which keys are attended to from
        # the final query position; 0 = valid, large negative = masked.
        last_row = attention_mask[:, 0, -1, :]
        valid_mask = last_row == 0
    elif attention_mask.dim() == 2:
        valid_mask = attention_mask != 0
    else:
        raise ValueError(f"Unexpected attention_mask.dim={attention_mask.dim()}, expected 2 or 4.")

    has_valid = valid_mask.any(dim=-1)
    flipped = torch.flip(valid_mask.to(dtype=torch.long), dims=[1])
    inv_idx = flipped.argmax(dim=-1)
    last_idx = (seq_len - 1) - inv_idx
    last_idx = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))
    return last_idx