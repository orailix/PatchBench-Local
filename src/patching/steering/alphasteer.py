
import sys
import torch
import json
import logging
import os
import random

import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Any
from pathlib import Path
from tqdm import tqdm

from .base import (
    InferenceContext,
    PreparedSteeringModel,
    SteeringArtifacts,
    SteeringModel,
    TrainingContext,
)

from src.models.utils import add_padding_token, get_model, get_tokenizer
from src.patching.utils import get_model_layer_list, parse_layers
from src.patching.utils.utils import resolve_dtype
from src.models.utils import safe_name


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
ALPHA_VECTOR_DIR = REPO_ROOT / "vectors" / "alphasteer"

DEFAULT_LAYER_SPEC_BY_MODEL = {
    "meta-llama/Llama-3.1-8B-Instruct": "8,9,10,11,12,13,14,16,18,19",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24",
    "google/gemma-2-12b-it": "11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23",
    "google/gemma-3-4b-it": "9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19",
    "Qwen/Qwen2.5-3B-Instruct": "9, 10, 11, 12, 13, 14, 15, 16, 18, 19",
    "Qwen/Qwen2.5-14B-Instruct": "12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24",
    "mistralai/Mistral-7B-Instruct-v0.3": "8, 9, 10, 11, 12, 13, 14, 15, 16",
    # "mistralai/Ministral-3-14B-Instruct-2512": "10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22"
}

AlphaSteer_CALCULATION_CONFIG = {
    "meta-llama/Llama-3.1-8B-Instruct": [(8, 0.6), (9, 0.6), (10, 0.6), (11, 0.6), (12, 0.4), (13, 0.5), (14, 0.6), (15, 0.6), (16, 0.6), (17, 0.6), (18, 0.6), (19, 0.6)],
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": [(12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6), (17, 0.6), (18, 0.6), (19, 0.6), (20, 0.6), (21, 0.6), (22, 0.6), (23, 0.6), (24, 0.6)],
    "Qwen/Qwen2.5-3B-Instruct": [(9, 0.6), (10, 0.6), (11, 0.6), (12, 0.6), (13, 0.5), (14, 0.6), (15, 0.5), (16, 0.5), (17, 0.5), (18, 0.3), (19, 0.3)],
    "Qwen/Qwen2.5-14B-Instruct": [(12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6), (17, 0.6), (18, 0.6), (19, 0.6), (20, 0.6), (21, 0.6), (22, 0.6), (23, 0.6), (24, 0.6)],
    # "google/gemma-2-9b-it": [(6, 0.5), (8, 0.4), (10, 0.6), (11, 0.6), (12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6), (18, 0.6), (22, 0.5)],
    "google/gemma-3-4b-it": [(9, 0.5), (10, 0.6), (11, 0.5), (12, 0.5), (13, 0.5), (14, 0.3), (15, 0.3), (16, 0.5), (17, 0.5), (18, 0.5), (19, 0.6)],
    "google/gemma-3-12b-it": [(11, 0.5), (12, 0.5), (13, 0.5), (14, 0.3), (15, 0.3), (16, 0.5), (17, 0.5), (18, 0.5), (19, 0.6), (20, 0.6), (21, 0.6), (22, 0.6), (23, 0.6)],
    "mistralai/Mistral-7B-Instruct-v0.3": [(8, 0.6), (9, 0.6), (10, 0.6), (11, 0.6), (12, 0.4), (13, 0.5), (14, 0.6), (16, 0.6), (17, 0.6), (18, 0.6), (19, 0.6)],
    # "mistralai/Ministral-3-14B-Instruct-2512": [(10, 0.6), (11, 0.6), (12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6), (17, 0.6), (18, 0.6), (19, 0.6), (20, 0.6), (21, 0.6), (22, 0.6)],
}

DATA_DIR = REPO_ROOT / "data" / "raw"

BENIGN_TRAIN_JSON = DATA_DIR / "benign_train.json"

SAFE = 0
UNSAFE = 1


def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    Q = null_space_l(A, min_null_space_ratio, abs_nullspace_ratio)
    P = Q @ Q.T
    return P

def null_space_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    device = A.device
    A32 = A.to(torch.float32)
    # _, S, Vh = torch.linalg.svd(A32.T @ A32)
    _, S, Vh = torch.linalg.svd(A32.cpu(), full_matrices=False) # Use less memory
    S = S.to(A.dtype).to(device)
    Vh = Vh.to(A.dtype).to(device)
    M, N = A.shape[0], A.shape[1]
    
    if abs_nullspace_ratio > 0:
        num = int(N * abs_nullspace_ratio)
    
    else:    
        S_ = torch.sqrt(S)
        rcond = torch.finfo(S.dtype).eps * max(M, N)
        tol = torch.amax(S_) * rcond
        num = torch.sum(S_ < tol)
    
        if num / N < min_null_space_ratio:
            num = int(N * min_null_space_ratio)
    
    print(f"final null space ratio: {num / N}")
    Q = Vh[-num:,:].T.conj()
    return Q

def cal_tilde_delta_with_regularization_l(
    H_h_layer, P_layer, refusal_vector, lambda_reg, device="cuda:0"):
    X = H_h_layer @ P_layer
    A = X.T @ X + lambda_reg * (P_layer.T @ P_layer)
    b = X.T @ refusal_vector.repeat(X.shape[0], 1)
    device = A.device
    tilde_delta_layer = torch.linalg.pinv(A.to(torch.float32).cpu()) @ b.to(torch.float32).cpu()
    tilde_delta_layer = tilde_delta_layer.to(A.dtype).to(device)
    result = X @ tilde_delta_layer
    avg_reconstruction_error = torch.norm(result - refusal_vector) / X.shape[0]
    print(f"avg_reconstruction_error: {avg_reconstruction_error}", end="\t")
    print(f"refusal_vector norm: {torch.norm(refusal_vector)}")
    return tilde_delta_layer

def cal_steering_matrix_l(P_layer, tilde_delta_layer, device="cuda:0"):
    P_layer = P_layer.to(device)
    tilde_delta_layer = tilde_delta_layer.to(device)
    steering_matrix_layer = P_layer @ tilde_delta_layer
    return steering_matrix_layer

def _ensure_alphasteer_vendor_path() -> None:
    path = Path(__file__).resolve().parent.parent / "AlphaSteer"
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_alphasteer_vendor_path()

from activation_steering.malleable_model import MalleableModel

def _resolve_dtype_value(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return resolve_dtype(str(value))

class AlphaSteeringModel(SteeringModel):
    """AlphaSteer integration placeholder following the shared steering interface."""

    method_name = "alphasteer"

    @torch.no_grad()
    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """Train AlphaSteer and return the saved artifacts."""
        model, tokenizer = self._load_runtime_model_and_tokenizer(context)
        num_layers = len(get_model_layer_list(model))

        data_dir = REPO_ROOT / "data" / "raw"
        
        REFUSAL_JSON = Path("benchmark_data") / "training_data" / f"AlphaSteer/{safe_name(context.model_name, dir=False, remove_first_caps=True)}" / "row_data.json"
        
        BENIGN_TRAIN_PT = ALPHA_VECTOR_DIR / context.model_name / "benign_train.pt" # Alpaca
        REFUSAL_PT = ALPHA_VECTOR_DIR / context.model_name / "refusal.pt"
        COCONOT_PREF_PT = ALPHA_VECTOR_DIR / context.model_name / "coconot_pref.pt"
        COCONOT_ORIGINAL_PT = ALPHA_VECTOR_DIR / context.model_name / "coconot_original.pt"
        HARMFUL_1000_US_PT = ALPHA_VECTOR_DIR / context.model_name / "harmful_1000_us.pt"

        os.makedirs(ALPHA_VECTOR_DIR / context.model_name, exist_ok=True)
        if not (BENIGN_TRAIN_PT).exists():
            benign_emb = self.extract_embeddings(
                model=model,
                tokenizer=tokenizer,
                prompts=[record["query"] for record in json.loads(BENIGN_TRAIN_JSON.read_text(encoding="utf-8"))],
                batch_size=16,
                layers=list(range(num_layers)),
            )
            torch.save(benign_emb.cpu(), BENIGN_TRAIN_PT)
        else:
            benign_emb = torch.load(BENIGN_TRAIN_PT).to(model.device)
        if not (COCONOT_PREF_PT).exists():
            coconot_pref_emb = self.extract_embeddings(
                model=model,
                tokenizer=tokenizer,
                prompts=[record["prompt"] for record in json.loads((data_dir / "coconot_pref.json").read_text(encoding="utf-8"))],
                batch_size=16,
                layers=list(range(num_layers)),
            )
            torch.save(coconot_pref_emb.cpu(), COCONOT_PREF_PT)
        else:
            coconot_pref_emb = torch.load(COCONOT_PREF_PT).to(model.device)
        if not (COCONOT_ORIGINAL_PT).exists():
            coconot_orig_emb = self.extract_embeddings(
                model=model,
                tokenizer=tokenizer,
                prompts=[record["prompt"] for record in json.loads((data_dir / "coconot_original.json").read_text(encoding="utf-8"))],
                batch_size=4,
                layers=list(range(num_layers)),
            )
            torch.save(coconot_orig_emb.cpu(), COCONOT_ORIGINAL_PT)
        else:
            coconot_orig_emb = torch.load(COCONOT_ORIGINAL_PT).to(model.device)
        if not (HARMFUL_1000_US_PT).exists() or True:
            data = json.loads(REFUSAL_JSON.read_text(encoding="utf-8"))

            safe = random.sample([r for r in data if r["label"] == "safe"], 1200)
            unsafe = random.sample([r for r in data if r["label"] == "unsafe"], 300)

            prompts = [r["prompt"] for r in safe + unsafe]
            
            harmful_emb_us = self.extract_embeddings(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=4,
                layers=list(range(num_layers)),
            ).to(model.device)
            torch.save(harmful_emb_us.cpu(), HARMFUL_1000_US_PT)
        else:
            harmful_emb_us = torch.load(HARMFUL_1000_US_PT).to(model.device)
        # Sample a subset of borderline examples to balance the dataset
        indices_borderline = torch.randperm(coconot_orig_emb.size(0))[:4000 - coconot_pref_emb.size(0)]
        # Combine all benign embeddings
        benign_emb_train = torch.cat([
            benign_emb, coconot_orig_emb[indices_borderline], coconot_pref_emb], dim=0).to(model.device)
        
        device = model.device
        torch.cuda.empty_cache()
        
        # Combine all harmful embeddings
        harmful_emb_train = torch.cat([harmful_emb_us], dim=0)
        # Free memory
        torch.cuda.empty_cache()
        
        if not (REFUSAL_PT).exists():
            refusal_emb = self._generate_refusal_vector(tokenizer, model, context, path=REFUSAL_JSON)
            torch.save(refusal_emb.cpu(), REFUSAL_PT)
        else:
            refusal_emb = torch.load(REFUSAL_PT).to(device)
        
        del model; del tokenizer
        
        num_layer = refusal_emb.shape[0]
        d_model = refusal_emb.shape[1]
        P = torch.zeros(num_layer, d_model, d_model, device=device)
        tilde_delta = torch.zeros(num_layer, d_model, d_model, device=device)
        steering_matrix = torch.zeros(num_layer, d_model, d_model, device=device)
        
        layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[context.model_name]
        for layer, ratio in layers_ratio_list:            
            # Calculate null space projection matrix for benign embeddings
            P_layer = null_space_projection_l(benign_emb_train[:, layer, :], abs_nullspace_ratio=ratio)
            P[layer] = P_layer

            # Calculate delta matrix with regularization to align with refusal vectors
            tilde_delta_layer = cal_tilde_delta_with_regularization_l(
                harmful_emb_train[:, layer, :], P_layer, refusal_emb[layer], lambda_reg=10.0, device=device)
            tilde_delta[layer] = tilde_delta_layer

            # Calculate final steering matrix by combining projection and delta matrices
            steering_matrix_layer = cal_steering_matrix_l(
                P_layer, tilde_delta_layer, device=device)
            steering_matrix[layer] = steering_matrix_layer

        (context.artifacts_dir / context.model_name).mkdir(parents=True, exist_ok=True)
        torch.save(steering_matrix.cpu(), context.artifacts_dir / context.model_name / "steering_matrix.pt")
        
        steering_artifacts = SteeringArtifacts(
            paths={"steering_matrix": context.artifacts_dir / context.model_name / "steering_matrix.pt"},
            metadata={"layer_spec": layers_ratio_list, "refusal_vector_path": context.artifacts_dir / context.model_name / "refusal.pt"},
        )
        
        return steering_artifacts

    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Prepare the AlphaSteer runtime model."""
        if isinstance(model, MalleableModel):
            model._remove_hooks()
        steering_model = MalleableModel(model=model, tokenizer=tokenizer)

        layer_spec = DEFAULT_LAYER_SPEC_BY_MODEL.get(context.model_name)
        num_layers = len(get_model_layer_list(model))
        if layer_spec is None:
            raise ValueError(f"No default layer spec found for model {context.model_name}.")
         
        refusal_matrix_path = ALPHA_VECTOR_DIR / context.model_name / "steering_matrix.pt"
        
        # Check if they exist; if not, raise an error with instructions to run the training script.
        if not refusal_matrix_path.exists():
            raise FileNotFoundError(
                f"Steering matrix not found for model {context.model_name} at expected paths: {refusal_matrix_path}. \n"
                f"Please run the training script to generate these artifacts: `python src/patching/train/train_alphasteer.py --model {context.model_name} --config <path_to_config>`."
            )
        
        steering_matrix = torch.load(refusal_matrix_path, map_location=steering_model.device)   # TODO CHANGE THIS
        steering_matrix = steering_matrix.to(context.dtype)
        
        nonzero_layer_ids = {
            i for i in range(steering_matrix.shape[0])
            if torch.any(steering_matrix[i])
        }
        
        layer_ids = parse_layers(
            layer_spec=layer_spec,
            num_layers=num_layers,
            vector_layer_ids=nonzero_layer_ids,
        )    
        
        strength_scalar = float(context.method_kwargs.get("strength", -5))
        strength = [0.0] * num_layers
        for layer in layer_ids:
            strength[layer] = strength_scalar
            
        steering_model.set_steering_parameters(
            steering_matrix = steering_matrix,
            strength = strength,
            selected_layers = layer_ids,
        )

        return PreparedSteeringModel(model=steering_model, tokenizer=tokenizer)

    def extract_embeddings(self, model, tokenizer, prompts, batch_size, layers):
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        
        resid_pre_cache = {i: [] for i in layers}
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            
            batch_prompts = formatted_prompts[i:i+batch_size]
            batch_inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**batch_inputs, output_hidden_states=True)

            for layer_idx in layers:
                resid_pre_cache[layer_idx].append(
                    outputs.hidden_states[layer_idx][:, -1, :].detach().to('cpu')) 
            
            outputs = None
            torch.cuda.empty_cache()
            
        resid_pre_benign_embs = {
            layer: torch.cat(resid_pre_cache[layer], dim=0)
            for layer in layers}
        
        H = torch.stack(list(resid_pre_benign_embs.values()), dim=1)
        return H

    def _generate_refusal_vector(self, tokenizer, model, context, path):
        lang_data = self._load_preference_data(tokenizer, path)

        # source_lan_emb = {}
        # for lang, data in tqdm(lang_data.items(), desc='Computing sentence embeddings'):
        #     sent_embs = self._get_hidden_sentence_embeddings(model, data)  # (L, N, D)
            # source_lan_emb[lang] = sent_embs

        source_lan_emb = self._get_hidden_sentence_embeddings(model, lang_data["prompt"]).transpose(0, 1)
        # temp_harmful = source_lan_emb["first"]
        # temp_harmless = source_lan_emb["second"]
        labels = torch.tensor(np.array(lang_data["label"]))
        
        temp_harmful  = source_lan_emb[labels == UNSAFE*torch.ones_like(labels)]
        temp_harmless = source_lan_emb[labels ==   SAFE*torch.ones_like(labels)]

        refusal_vector = temp_harmful.mean(dim=0) - temp_harmless.mean(dim=0)

        return refusal_vector.to(device=model.device)

    def _load_preference_data(self, tokenizer, path, keys=None):
        filepath = Path(path)

        if not filepath.exists():
            logging.error(f'File not found at: {filepath}')
            return {}

        # Load all records from JSON array or JSONL
        records = []
        if filepath.suffix.lower() == '.jsonl':
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
        
        if not records:
            logging.warning(f'No records loaded from: {filepath}')
            return {}

        # Resolve keys via auto-detection when not supplied
        if keys is None:
            sample = records[0]
            if 'first' in sample and 'second' in sample:
                keys = ['first', 'second']
            elif 'query' in sample:
                keys = ['query']
            elif 'prompt' in sample:
                keys = ['prompt']
            else:
                logging.error(f'Cannot auto-detect keys from fields: {list(sample.keys())}')
                return {}
        elif isinstance(keys, str):
            keys = [keys]

        lang_data: dict[str, list[str]] = {k: [] for k in keys + (["label"] if "label" in sample else [])}
        label_count = {
            "safe"  : 0,
            "unsafe": 0
        }
        for record in records:
            if "label" in record:
                if label_count[record["label"]] >= 339:
                    continue
                label_count[record["label"]] += 1
                lang_data["label"].append(SAFE if record["label"] == 'safe' else UNSAFE)
            for k in keys:
                if k == 'label':
                    continue
                if k in record:
                    text = str(record[k]).strip()
                    messages = [
                        {"role": "system", "content": ''},
                        {"role": "user", "content": text},
                    ]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    lang_data[k].append(formatted)

        for k, v in lang_data.items():
            logging.info(f'Loaded {len(v)} {k} samples.')

        return {
            k: (tokenizer(
                v,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ) if k != "label" else v)
            for k, v in lang_data.items()
            if v
        }
    
    def _get_hidden_sentence_embeddings(self, model, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        batch_size = min(1, input_ids.size(0))
        num_batches = input_ids.size(0) // batch_size
        sent_embs = []

        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f'Batch {i + 1}/{num_batches} of size {batch_input_ids.size(0)}')

            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids.to(model.device), 
                    attention_mask=batch_attention_mask.to(model.device), 
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
            del outputs
            hidden_states = hidden_states[1:]  # Remove the input layer embeddings
            hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
            hidden_sent_embs = hidden_states[:, :, -1, :]
            sent_embs.append(hidden_sent_embs.detach().to('cpu'))
            del hidden_sent_embs, hidden_states
            torch.cuda.empty_cache()

        # sent_embs is a list of tensors of shape (L, N, D). Concatenate them along the batch dimension
        hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
        del sent_embs
        logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
        torch.cuda.empty_cache()
        return hidden_sent_embs
    
    
    def _load_runtime_model_and_tokenizer(self, context: TrainingContext) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        dtype = _resolve_dtype_value(context.method_kwargs.get("dtype", "auto"))
        snapshot_path = str(context.snapshot_path)
        tokenizer = get_tokenizer(context.model_name, snapshot_path)
        tokenizer.padding_side = "left"
        tokenizer = add_padding_token(tokenizer, context.model_name)
        model = get_model(context.model_name, snapshot_path, dtype=dtype, device_map="auto")
        return model, tokenizer