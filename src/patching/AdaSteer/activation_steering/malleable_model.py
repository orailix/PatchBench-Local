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

def _normalise(direction: torch.Tensor, divisor: Optional[float] = None) -> torch.Tensor:
    if divisor is not None:
        return direction / divisor
    return direction / direction.norm()


def _project_and_scale(
    hs_last: torch.tensor,
    harmful_anchor: torch.tensor,
    harmless_anchor: torch.tensor,
    direction: torch.tensor,
    standard: float = 100.0,
) -> torch.tensor:
    """
    Project the last-token hidden state relative to both class anchors onto
    ``direction``, normalise by the harmful/harmless separation, and return
    the scaled harmful-class projection (shape [B]).
    """
    proj_harmful  = (hs_last - harmful_anchor)  @ direction
    proj_harmless = (hs_last - harmless_anchor) @ direction
    scale = (proj_harmful - proj_harmless).mean()
    return proj_harmful / scale * standard

class _AdaSteerLayerHook:
    def __init__(self, steering_matrix: torch.Tensor, strength: float):
        self.steering_matrix = steering_matrix
        self.strength = strength

    def __call__(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        raise NotImplementedError("This is a placeholder hook implementation. The actual logic will depend on the specific model architecture and how the steering vector should be applied to the hidden states.")
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
        
        # Steering tensors (set by get_steer)
        self.refusal_vector: Optional[torch.Tensor] = None
        self.steer_vector_2: Optional[torch.Tensor] = None
        self._scalar_alpha: Optional[float] = None

        # AdaSteer dynamic state
        self.alpha_list: Optional[torch.Tensor] = None
        self.beta_list: Optional[torch.Tensor] = None
        self.harmful_anchors: Optional[np.ndarray] = None
        self.harmless_anchors: Optional[np.ndarray] = None
        self.pseudo_harmful_anchors: Optional[np.ndarray] = None
        self.pseudo_harmless_anchors: Optional[np.ndarray] = None
        self.acceptance_direction: Optional[np.ndarray] = None
        self.pseudo_acceptance_direction: Optional[np.ndarray] = None

        # Activation collection
        self.all_activations: List = []
        self._hs_last_cpu: List = []   # reset at the start of each inner-model forward
        self._is_phase2: bool = False  # latched in the pre-forward hook
        self.standard: float = 100.0

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log(f"... The target model type is [cyan]{model.config.model_type}[/cyan].", style="magenta", class_name="MalleableModel")

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device
     
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

    # def generate(self, *args, **kwargs):
    #     return self.model.generate(*args, **kwargs)
    
    def generate(self, *args, max_new_tokens: int = 128, **kwargs):
        """
        Two-pass AdaSteer generate.

        Returns (tokens, alpha_list) where alpha_list is the per-sample adaptive
        steering strength tensor computed during the probe pass (shape [B]).
        """
        self.reset_alpha()
        self.model.generate(*args, max_new_tokens=2, **kwargs)
        tokens = self.model.generate(*args, max_new_tokens=max_new_tokens, **kwargs)
        if self.validating:
            return tokens, self.pos_list_alpha, self.pos_list_beta
        return tokens
    
    def reset_alpha(self) -> None:
        """Reset to phase 1 so that the next generate call re-detects alpha/beta."""
        self.alpha_list = None
        self.beta_list = None

    def get_steer(
        self, 
        refusal_vector: torch.tensor, 
        projection: torch.tensor,
        harmful_accept_vector: torch.tensor,
        harmful_refuse_vector: torch.tensor,
        benign_accept_vector: torch.tensor,
        alpha_layer: int,
        beta_layer: int,
        w_r: float = -0.02, 
        b_r: float = -1.2, 
        w_c: float = 0.017, 
        b_c: float = 0.25,
        lambda_r: float = 0.1,
        lambda_c: float = 0.,
        alpha_min: float = .0,
        alpha_max: float = 0.3,
        beta_min: float = -0.3,
        beta_max: float = 0.3,
    ) -> None:
        self.steer_vector = -refusal_vector # the acceptance direction is the negative of the refusal vector
        self.alpha_layer  = alpha_layer
        self.beta_layer   = beta_layer
    
        self.all_activations = []
        self.alpha_list      = None
        self.beta_list       = None
        if w_r == 0. and b_r == 0.:
            self.validating = True
            self.alpha_formula = lambda x: torch.ones_like(x) * lambda_r
        else:
            self.validating = False
            self.alpha_formula=lambda x: torch.clamp((w_r) * (x - b_r), alpha_min, alpha_max)
        if w_c == 0. and b_c == 0.:
            self.validating = True 
            self.beta_formula = lambda x: torch.ones_like(x) * lambda_c
        else:
            self.validating = False or self.validating and lambda_c != 0.
            self.beta_formula=lambda x:  torch.clamp((w_c) * (x - b_c), beta_min, beta_max)
       # print(self.validating, lambda_r, lambda_c, w_r, b_r, w_c, b_c)
        self.projection = projection

        self.harmful_anchors  = harmful_accept_vector
        self.harmless_anchors = harmful_refuse_vector
        self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
        self.acceptance_direction[alpha_layer] = _normalise(
            self.acceptance_direction[alpha_layer]
        )
        self.acceptance_direction[beta_layer] = _normalise(
            self.acceptance_direction[beta_layer]
        )

        self.pseudo_harmful_anchors  = harmful_accept_vector
        self.pseudo_harmless_anchors = benign_accept_vector
        self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
        self.pseudo_acceptance_direction[alpha_layer] = _normalise(
            self.pseudo_acceptance_direction[alpha_layer]
        )
        self.pseudo_acceptance_direction[beta_layer] = _normalise(
            self.pseudo_acceptance_direction[beta_layer]
        )

        self._install_hooks()
    
    def _install_hooks(self) -> None:
        self._remove_hooks()
        layers = self._get_layers()

        # ── Pre-forward: latch phase and reset per-pass buffer ────────────────
        def pre_hook(module, args):
            self._hs_last_cpu = []
            # Capture phase at the START of this forward pass so that layer hooks
            # are consistent even after alpha_list is set mid-pass (in phase 1).
            self._is_phase2 = self.alpha_list is not None

        self._hooks.append(self.model.register_forward_pre_hook(pre_hook))

        # ── Post-forward: flush activation buffer into all_activations ────────
        # def post_hook(module, args, output):
        #     if self._is_phase2 and self._hs_last_cpu:
        #         _B, _L, _D = output[0].shape
        #         if _L > 1:
        #             stacked = torch.stack(self._hs_last_cpu, dim=1)  # (B, ?, D)
        #             self.all_activations.extend(stacked.tolist())

        # self._hooks.append(self.model.model.register_forward_hook(post_hook))

        # ── Per-layer hook ────────────────────────────────────────────────────
        for layer_idx, layer in enumerate(layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_hook(layer_idx))
            )
            
    def _get_layers(self):
        return get_model_layer_list(self.model)
    
    def _make_layer_hook(self, layer_idx: int):
        def hook(module, args, output):
            # Some architectures (e.g. Gemma3) return a tuple; extract the hidden states tensor.
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            _B, _L, _D = hidden_states.shape

            # ── AdaSteer mode: phase 1 (no injection, probe sentinel layers) ──
            if not self._is_phase2:
                if layer_idx == self.alpha_layer:
                    hs = hidden_states[:, -1, :]#.to(torch.float).cpu().numpy()
                    proj = _project_and_scale(
                        hs,
                        self.harmful_anchors[self.alpha_layer],
                        self.harmless_anchors[self.alpha_layer],
                        self.acceptance_direction[self.alpha_layer],
                        self.standard,
                    )
                    self.alpha_list = self.alpha_formula(proj) # torch.ones_like(self.alpha_formula(proj))*1e-2#/self.alpha_formula(proj) * 1e-2
                    # self.alpha_list = torch.ones_like(self.alpha_formula(proj))*1e-1
                    # print(self.alpha_list)
                    self.pos_list_alpha = proj


                if layer_idx == self.beta_layer:
                    hs = hidden_states[:, -1, :]#.to(torch.float).cpu().numpy()
                    proj = _project_and_scale(
                        hs,
                        self.pseudo_harmful_anchors[self.beta_layer],
                        self.pseudo_harmless_anchors[self.beta_layer],
                        self.pseudo_acceptance_direction[self.beta_layer],
                        self.standard,
                    )
                    self.beta_list = self.beta_formula(proj)
                    # print(self.beta_list)
                    self.pos_list_beta = proj

                return output  # no modification in phase 1

            # ── AdaSteer mode: phase 2 (inject + collect activations) ─────────
            if _L > 1:
                # Per-sample scaling: [B] outer [D] → [B, D] → [B, 1, D]
                hidden_states = hidden_states + (
                    torch.outer(self.alpha_list[:_B], self.steer_vector[layer_idx]).unsqueeze(1)
                    + torch.outer(self.beta_list[:_B], self.pseudo_acceptance_direction[layer_idx]).unsqueeze(1)
                )

            self._hs_last_cpu.append(
                hidden_states[:, -1, :].to(torch.float).cpu()
            )

            return (hidden_states,) + output[1:] if is_tuple else hidden_states

        return hook
