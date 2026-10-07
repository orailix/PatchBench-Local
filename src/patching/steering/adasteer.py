import sys
import os
import json
import torch
import logging

import numpy as np

from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Any
from tqdm import tqdm

from src.models.utils import add_padding_token, get_model, get_tokenizer
from src.patching.utils.utils import resolve_dtype
from src.patching.utils import get_adasteer_layers, get_model_layer_list
from src.models.utils import safe_name
from .base import (
    InferenceContext,
    PreparedSteeringModel,
    SteeringArtifacts,
    SteeringModel,
    TrainingContext,
)

def _ensure_adasteer_vendor_path() -> None:
    path = Path(__file__).resolve().parent.parent / "AdaSteer"
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_ensure_adasteer_vendor_path()

from activation_steering.malleable_model import MalleableModel

SAFE = 0
UNSAFE = 1

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
DATA_DIR = REPO_ROOT / "data" / "raw"
ADA_VECTOR_DIR = REPO_ROOT / "vectors" / "alphasteer"

BENIGN_TRAIN_JSON = DATA_DIR / "benign_train.json"
COCONOT_PREF_JSON = DATA_DIR / "coconot_pref.json"
COCONOT_ORIGINAL_JSON = DATA_DIR / "coconot_original.json"

def _resolve_dtype_value(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return resolve_dtype(str(value))

def orthogonalize(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Remove the component of w along r (Gram-Schmidt step)."""
    r_unit = r / r.norm()
    return w - (w @ r_unit) * r_unit

class AdaSteeringModel(SteeringModel):
    """AdaSteer integration placeholder following the shared steering interface."""

    method_name = "adasteer"

    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """Train AdaSteer and return the saved artifacts."""
        model, tokenizer = self._load_runtime_model_and_tokenizer(context)
        device = next(model.parameters()).device
        num_layers = len(get_model_layer_list(model))

        data_dir = REPO_ROOT / "data" / "raw"
        
        REFUSAL_JSON = Path("benchmark_data") / "training_data" / f"AdaSteer/{safe_name(context.model_name, dir=False, remove_first_caps=True)}" / "row_data.json"
        
        REFUSAL_PT  = ADA_VECTOR_DIR / context.model_name / "refusal.pt"
        HARMFUL_ACCEPT_PT  = ADA_VECTOR_DIR / context.model_name / "harmful_accept.pt"
        HARMFUL_REFUSE_PT = ADA_VECTOR_DIR / context.model_name / "harmful_refuse.pt"
        
        HARMFULNESS_PT = ADA_VECTOR_DIR / context.model_name / "harmfulness.pt"
        
        BENIGN_TRAIN_PT = ADA_VECTOR_DIR / context.model_name / "benign_train.pt"
        COCONOT_PREF_PT = ADA_VECTOR_DIR / context.model_name / "coconot_pref.pt"
        COCONOT_ORIGINAL_PT = ADA_VECTOR_DIR / context.model_name / "coconot_original.pt"
        
        PROJECTION_PT = ADA_VECTOR_DIR / context.model_name / "projection.pt"
        
        
        os.makedirs(ADA_VECTOR_DIR / context.model_name, exist_ok=True)
        
        if not (REFUSAL_PT).exists() or not (HARMFUL_ACCEPT_PT).exists() or not (HARMFUL_REFUSE_PT).exists():
            refusal_emb, harmful_accept_emb, harmful_refuse_emb = self._generate_refusal_vector(tokenizer, model, context, path=REFUSAL_JSON)
            torch.save(refusal_emb.cpu(),  REFUSAL_PT)
            torch.save(harmful_accept_emb.cpu(),  HARMFUL_ACCEPT_PT)
            torch.save(harmful_refuse_emb.cpu(), HARMFUL_REFUSE_PT)
        else:
            refusal_emb  = torch.load(REFUSAL_PT).to(device)
            harmful_accept_emb  = torch.load(HARMFUL_ACCEPT_PT).to(device)
            harmful_refuse_emb = torch.load(HARMFUL_REFUSE_PT).to(device)
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
                prompts=[record["prompt"] for record in json.loads(COCONOT_PREF_JSON.read_text(encoding="utf-8"))],
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
                prompts=[record["prompt"] for record in json.loads(COCONOT_ORIGINAL_JSON.read_text(encoding="utf-8"))],
                batch_size=4,
                layers=list(range(num_layers)),
            )
            torch.save(coconot_orig_emb.cpu(), COCONOT_ORIGINAL_PT)
        else:
            coconot_orig_emb = torch.load(COCONOT_ORIGINAL_PT).to(model.device) 
            
            
        # Sample a subset of borderline examples to balance the dataset
        indices_borderline = torch.randperm(coconot_orig_emb.size(0))[:4000 - coconot_pref_emb.size(0)]
        # Combine all benign embeddings
        benign_emb_train = torch.cat([
            benign_emb, coconot_orig_emb[indices_borderline], coconot_pref_emb], dim=0).to(model.device).transpose(0, 1)
        
        harmfulness_emb = benign_emb_train.mean(dim=1) - harmful_accept_emb.mean(dim=1)
        torch.save(harmfulness_emb.cpu(), HARMFULNESS_PT)
        
        projection = []
        for l in range(refusal_emb.shape[0]):
            # I used -refusal_emb because they're implementation uses acceptance instead 
            proj_l = orthogonalize(harmfulness_emb[l].float(), -refusal_emb[l].float())
            projection.append(proj_l)
            
        projection = torch.stack(projection)  # (L, D)
        
        torch.save(projection.cpu(), PROJECTION_PT)

        steering_artifacts = SteeringArtifacts(
            paths={
                "refusal": REFUSAL_PT, 
                "harmful_accept": HARMFUL_ACCEPT_PT, 
                "harmful_refuse": HARMFUL_REFUSE_PT,
                "harmfulness": HARMFULNESS_PT,
                "benign_train": BENIGN_TRAIN_PT,
                "projection": PROJECTION_PT,
            },
            metadata={ 
                "refusal_vector_path": str(REFUSAL_PT), 
                "harmful_vector_path": str(HARMFUL_ACCEPT_PT), 
                "harmful_refuse_vector_path": str(HARMFUL_REFUSE_PT)
            },
        )
        
        return steering_artifacts

    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Prepare the AdaSteer runtime model."""
        REFUSAL_PT         = ADA_VECTOR_DIR / context.model_name / "refusal.pt"
        PROJECTION_PT      = ADA_VECTOR_DIR / context.model_name / "projection.pt"
        HARMFUL_ACCEPT_PT  = ADA_VECTOR_DIR / context.model_name / "harmful_accept.pt"
        HARMFUL_REFUSE_PT  = ADA_VECTOR_DIR / context.model_name / "harmful_refuse.pt"
        BENIGN_TRAIN_PT    = ADA_VECTOR_DIR / context.model_name / "benign_train.pt"
        
        if isinstance(model, MalleableModel):
            model._remove_hooks()

        steering_model = MalleableModel(model=model, tokenizer=tokenizer)
        steering_model.get_steer(
            refusal_vector = torch.load(REFUSAL_PT).to(steering_model.device),
            projection = torch.load(PROJECTION_PT).to(steering_model.device),
            harmful_accept_vector=torch.load(HARMFUL_ACCEPT_PT).to(steering_model.device).mean(dim=1),
            harmful_refuse_vector=torch.load(HARMFUL_REFUSE_PT).to(steering_model.device).mean(dim=1),
            benign_accept_vector=torch.load(BENIGN_TRAIN_PT).to(steering_model.device).mean(dim=0),
            alpha_layer = get_adasteer_layers(context.model_name)[0],
            beta_layer = get_adasteer_layers(context.model_name)[1],
            w_r = float(context.method_kwargs.get("w_r")),
            b_r = float(context.method_kwargs.get("b_r")),
            w_c = float(context.method_kwargs.get("w_c")),
            b_c = float(context.method_kwargs.get("b_c")),
            lambda_r = float(context.method_kwargs.get("lambda_r")),
            lambda_c = float(context.method_kwargs.get("lambda_c")),
        )
       
        return PreparedSteeringModel(model=steering_model, tokenizer=tokenizer)

    def _load_runtime_model_and_tokenizer(self, context: TrainingContext) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        dtype = _resolve_dtype_value(context.method_kwargs.get("dtype", "auto"))
        snapshot_path = str(context.snapshot_path)
        tokenizer = get_tokenizer(context.model_name, snapshot_path)
        tokenizer.padding_side = "left"
        tokenizer = add_padding_token(tokenizer, context.model_name)
        model = get_model(context.model_name, snapshot_path, dtype=dtype, device_map="auto")
        return model, tokenizer
    
    def _generate_refusal_vector(self, tokenizer, model, context, path):
        lang_data = self._load_preference_data(tokenizer, path)

        source_lan_emb = self._get_hidden_sentence_embeddings(model, lang_data["prompt"]).transpose(0, 1)
        labels = torch.tensor(np.array(lang_data["label"]))
        
        harmful  = source_lan_emb[labels == UNSAFE*torch.ones_like(labels)]
        harmless = source_lan_emb[labels ==   SAFE*torch.ones_like(labels)]

        refusal_vector = harmful.mean(dim=0) - harmless.mean(dim=0)

        return refusal_vector.to(device=model.device), harmful.transpose(0, 1).to(device=model.device), harmless.transpose(0, 1).to(device=model.device)


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