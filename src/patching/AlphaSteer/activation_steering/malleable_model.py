import typing, os
import json
import torch

import numpy as np

from typing import List, Union, Tuple, Optional
from itertools import combinations
from collections import defaultdict
from typing import Any, Tuple
from sklearn.metrics import f1_score
from transformers import PretrainedConfig, PreTrainedModel

from rich.progress import track
from rich.table import Table
from activation_steering.config import log

from src.patching.utils import get_model_layer_list, get_last_valid_token_index

if typing.TYPE_CHECKING:
    from .extract import SteeringVector

class _AlphaSteerLayerHook:
    """
    Pre-forward hook for a single transformer decoder layer.

    Registered with ``layer.register_forward_pre_hook``.  The hook fires
    *before* the layer's own forward pass and modifies ``hidden_states`` by
    adding a dynamic steering vector computed as:

        steering_vector = last_hidden @ steering_matrix * strength

    where ``last_hidden`` is the hidden state of the last non-PAD token.
    Single-token autoregressive decode steps are skipped so that steering
    is applied only during the prefill (prompt) pass.
    """
    def __init__(self, steering_matrix: torch.Tensor, strength: float):
        self.steering_matrix = steering_matrix
        self.strength = strength

    def __call__(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        hidden_states: torch.Tensor = args[0]

        # Skip single-token autoregressive decode steps.
        if hidden_states.shape[1] <= 1:
            return args

        attention_mask: Optional[torch.Tensor] = args[1] if len(args) > 1 else None

        B, T, _ = hidden_states.shape
        device = hidden_states.device

        if self.steering_matrix.device != device:
            self.steering_matrix = self.steering_matrix.to(device)

        last_idx = get_last_valid_token_index(
            attention_mask=attention_mask,
            seq_len=T,
            batch_size=B,
            device=device,
        )  # (B,)

        batch_idx = torch.arange(B, device=device)
        last_hidden = hidden_states[batch_idx, last_idx, :]           # (B, D)
        steering_vector = last_hidden @ self.steering_matrix * self.strength  # (B, D)
        steering_vector = steering_vector.unsqueeze(1)                 # (B, 1, D)

        hidden_states = hidden_states + steering_vector
        return (hidden_states,) + args[1:]


class MalleableModel(torch.nn.Module):
    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizerBase'):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list = []
        
        self._layers = get_model_layer_list(self.model)
        self._steering_matrix: Optional[torch.Tensor] = None

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log(f"... The target model type is [cyan]{model.config.model_type}[/cyan].", style="magenta", class_name="MalleableModel")

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    @staticmethod
    def _normalise_layer_index(layer_idx: int, num_layers: int) -> int:
        normalised = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if normalised < 0 or normalised >= num_layers:
            raise IndexError(f"Layer {layer_idx} is out of range for model with {num_layers} layers.")
        return normalised

    def _build_layer_mask(
        self,
        num_layers: int,
        selected_layers: Optional[list[int] | tuple[int, ...] | set[int] | dict[int, bool]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a boolean mask over model layers.

        selected_layers can be either:
        - list/tuple/set of layer indices to enable, or
        - dict[layer_idx] -> bool to explicitly enable/disable each layer.
        """
        if selected_layers is None:
            return torch.ones(num_layers, dtype=torch.bool, device=device)

        mask = torch.zeros(num_layers, dtype=torch.bool, device=device)

        if isinstance(selected_layers, dict):
            for layer_idx, enabled in selected_layers.items():
                normalised = self._normalise_layer_index(layer_idx, num_layers)
                mask[normalised] = bool(enabled)
            return mask

        for layer_idx in selected_layers:
            normalised = self._normalise_layer_index(layer_idx, num_layers)
            mask[normalised] = True
        return mask
    
    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor] = None,
        strength: Optional[list] = None,
        selected_layers: Optional[list[int] | tuple[int, ...] | set[int] | dict[int, bool]] = None,
    ) -> None:
        """
        Attach (or refresh) alpha steering hooks on the model's decoder layers.

        The steering matrix is cached: pass it once, then call with only
        ``strength`` to update the per-layer strengths without reloading.

        Args:
            steering_matrix: Tensor of shape ``(num_layers, D, D)``.
            strength:         Per-layer strength list of length ``num_layers``.
                              Layers with strength ``0.0`` are skipped.
            selected_layers:  Optional layer selector to apply an additional
                              mask on top of ``strength``. Accepts either a
                              collection of layer indices or a
                              ``dict[layer_idx] -> bool``.
        """
        self._remove_hooks()

        if steering_matrix is not None:
            self._steering_matrix = steering_matrix

        if self._steering_matrix is None or strength is None:
            return

        device = self.device
        num_layers = len(self._layers)
        layer_mask = self._build_layer_mask(
            num_layers=num_layers,
            selected_layers=selected_layers,
            device=device,
        )
        strength = torch.tensor(strength, device=device)
        for layer_idx, layer in enumerate(self._layers):
            if not layer_mask[layer_idx]:
                continue

            layer_strength = strength[layer_idx] if layer_idx < len(strength) else 0.0
            if layer_strength == 0.0:
                continue

            layer_matrix = self._steering_matrix[layer_idx].to(device)
            if not torch.any(layer_matrix):
                continue

            hook = _AlphaSteerLayerHook(layer_matrix, layer_strength)
            handle = layer.register_forward_pre_hook(hook)
            self._hooks.append(handle)
            
    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        
        for name, module in self.model.named_modules():
            if module._forward_hooks:
                module._forward_hooks.clear()
            if module._forward_pre_hooks:
                module._forward_pre_hooks.clear()
            if module._backward_hooks:
                module._backward_hooks.clear()
        self._hooks.clear()
    
    def unwrap(self) -> PreTrainedModel:
        """
        Remove steering modifications and return the original model.
        """
        for handle in self._hooks:
            handle.remove()
            
        self._hooks.clear()
        return self.model

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def respond(self, prompt, settings=None, use_chat_template=True,reset_after_response=True):
        # Force model to CPU or GPU to ensure weights are in the correct device
        self.model.to(self.device)
        
        if use_chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{prompt}"}],
                tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        if settings is None:
            settings = {
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": False,
                "max_new_tokens": 50,
                "repetition_penalty": 1.1,
            }
        
        with torch.no_grad():  # Ensure we're not tracking gradients during inference
            output = self.model.generate(**input_ids, **settings)
        
        response = self.tokenizer.decode(output.squeeze()[input_ids['input_ids'].shape[1]:])

        # if reset_after_response:
        #     # reset for each call
        #     LeashLayer.reset_class()

        return response


    def respond_batch_sequential(self, prompts, settings=None, use_chat_template=True):
        self.model.to(self.device)
        responses = []
        for prompt in prompts:
            response = self.respond(prompt, settings, use_chat_template)
            responses.append(response)

        return responses
    

    def find_best_condition_point(
        self, 
        positive_strings: List[str], 
        negative_strings: List[str], 
        condition_vector: 'SteeringVector', 
        layer_range: Optional[Tuple[int, int]] = None, 
        max_layers_to_combine: int = 1, 
        threshold_range: Tuple[float, float] = (0.0, 1.0), 
        threshold_step: float = 0.01, save_analysis: bool = False, 
        file_path: Optional[str] = None, 
        condition_threshold_comparison_mode: str = "mean"
    ) -> Tuple[List[int], float, str, float]:
        raise NotImplementedError("find_best_condition_point is not implemented yet.")


    def _save_analysis_results(
        self, 
        analysis_results, 
        best_layers, 
        best_threshold, 
        best_direction, 
        best_f1, 
        file_path
    ):
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # If no file name is provided, generate a default one
        if not os.path.basename(file_path):
            file_name = f"condition_point_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(directory, file_name)

        summary = {
            "best_layers": best_layers,
            "best_threshold": best_threshold,
            "best_direction": best_direction,
            "best_f1_score": best_f1,
            "analysis": analysis_results
        }

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        log(f"Analysis results saved to {file_path}", style="bold blue", class_name="MalleableModel")


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

