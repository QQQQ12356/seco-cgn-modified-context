import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List
from peft import get_peft_model, inject_adapter_in_model
from safetensors.torch import load_file
from icecream import ic as pprint
from compression_modules import MultiscaleCompressor, resolve_budget, BUDGET_MODES
from context_compression import ContextCompressor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The generation settings the evaluation actually uses. They are module
# constants rather than arguments because ``forward`` calls ``generate``
# directly, and the checkpoint manifest has to record the values that ran --
# reading ``model_args.max_new_tokens`` instead would have written 20 while the
# decoder was capped at 15.
GENERATION_MAX_NEW_TOKENS = 15
GENERATION_REPETITION_PENALTY = 1.3

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    compress_ratio: int = field(default=16, metadata={"help": "compression ratio for the clustering"})
    # defaults follow the experimentally-selected route (see experiments/实验过程记录.docx):
    # multi-seed prompt queries + sharpened cluster weighting (learnable temperature)
    pool_method: str = field(default="multi_seed", metadata={"help": "prompt pooling method: seed | mean | last | multi_seed"})
    num_pool_seeds: int = field(default=8, metadata={"help": "number of learnable seed queries when pool_method=multi_seed"})
    pool_temp_init: float = field(default=0.05, metadata={"help": "init for learnable temperature applied to cluster softmax weights; 1.0 keeps original (near-uniform) weighting, ~0.05 sharpens"})
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    train: bool = field(default=False, metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"})
    max_new_tokens: int = field(default=20, metadata={"help": "Maximum tokens generated during inference."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty for generation."})
    temperature: float = field(default=0.0, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.9, metadata={"help": "Top-p for generation when sampling is enabled."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling during generation."})

    # --- compression route (see Docx/SECO压缩优化方案与实施交接.md) ---
    compressor_version: str = field(default="context_adaptive_v1", metadata={"help": "legacy | multiscale_budget_v1 | context_adaptive_v1"})
    context_boundary_mode: str = field(default="semantic")
    context_budget_mode: str = field(default="adaptive")
    context_merge_mode: str = field(default="hybrid")
    context_novelty_weight: float = field(default=0.5)
    context_boundary_radius: float = field(default=0.25)
    context_anchor_weight: float = field(default=0.35)
    budget_mode: str = field(default="strict_context", metadata={"help": f"one of {BUDGET_MODES}; ignored by the legacy compressor"})
    relevance_dim: int = field(default=256, metadata={"help": "low-dimensional shared relevance space width"})
    query_slots: int = field(default=8, metadata={"help": "number of learnable query slots"})
    query_phrase_widths: List[int] = field(default_factory=lambda: [2, 4], metadata={"help": "phrase pooling window widths"})
    context_window_widths: List[int] = field(default_factory=lambda: [8, 32], metadata={"help": "context local summary widths"})
    allocation_block_width: int = field(default=128, metadata={"help": "target context block width for budget allocation"})
    raw_memory_fraction: float = field(default=0.25, metadata={"help": "fraction of memory slots kept as raw evidence"})
    query_fusion_temperature: float = field(default=0.1, metadata={"help": "init for the slot fusion temperature"})
    assignment_temperature: float = field(default=0.1, metadata={"help": "init for the token-to-anchor assignment temperature"})
    merge_temperature: float = field(default=1.0, metadata={"help": "init for the merge weight temperature"})
    anchor_residual_init: float = field(default=0.7, metadata={"help": "init weight on the anchor's own hidden state"})
    coverage_weight: float = field(default=0.2, metadata={"help": "weight of the un-covered-query-aspect gain"})
    redundancy_weight: float = field(default=0.2, metadata={"help": "weight of the selected-anchor redundancy penalty"})
    position_penalty: float = field(default=0.2, metadata={"help": "weight of the distance penalty in the merge assignment"})
    diversity_loss_weight: float = field(default=0.01, metadata={"help": "weight of the query slot diversity loss"})
    selection_mode: str = field(default="budgeted", metadata={"help": "budgeted (block allocation + coverage) | topk (global top-k, ablation)"})

    # --- SecO v2: calibrated query attention + aspect-aware aggregation ---
    query_attention_mode: str = field(default="dot", metadata={"help": "dot (historical Q.K/sqrt(p)) | cosine_tau (normalised slots/keys over a learnable temperature)"})
    query_attn_temperature: float = field(default=0.1, metadata={"help": "init for the query attention temperature; only used by cosine_tau"})
    query_mode: str = field(default="multi_slot", metadata={"help": "multi_slot (learned slots) | mean (single mean-query control)"})
    diversity_mode: str = field(default="attention", metadata={"help": "attention (historical slot-attention overlap) | slot_rep (squared cosine between slot vectors)"})
    merge_gate_mode: str = field(default="plain", metadata={"help": "plain (anchor+summary gate) | query (also conditioned on the anchor's query context)"})
    legacy_budget_mode: str = field(default="legacy", metadata={"help": "budget rule the legacy compressor uses; 'legacy' keeps the historical ceil+min-2, strict_context makes it same-K as v1"})


    
@dataclass
class DataArguments:
    train_file: str = field(default="/path/to/train/file", metadata={"help": "Path to the training data."})
    test_file: str = field(default="/path/to/test/file", metadata={"help": "Path to the test data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset"})
    test_sample_index: int = field(default=0, metadata={"help": "Index of the sample to test in the dataset."})
    train_samples: int = field(default=24000, metadata={"help": "Maximum number of training examples."})
    eval_samples: int = field(default=500, metadata={"help": "Maximum number of evaluation examples."})
    shuffle_eval: bool = field(default=False, metadata={"help": "Randomly sample the evaluation set."})
    eval_output_dir: str = field(default="./eval_result", metadata={"help": "Directory used to save inference results."})
    
@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length."},
    )
    report_to: Optional[str] = field(default="wandb")
    project_name: Optional[str] = field(default="cluster")
    max_steps: int = field(default=10000, metadata={"help": "Maximum number of training steps."})
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=10000, metadata={"help": "Interval between two checkpoints saved."})
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=200000, metadata={"help": "Interval between two evaluations."})
    num_train_epochs: int = field(default=1)
    add_special_token_for_lm: bool = field(default=False)
    restore_from: str = field(default="", metadata={"help": "checkpoint to restore from"})
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=100)
    deepspeed: str = field(default=None)
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16"})
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)
    pooler_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    encoder_last_hidden_only: bool = field(default=False, metadata={"help": "call the encoder backbone directly instead of the CausalLM head path"})
    new_module_lr: Optional[float] = field(default=None, metadata={"help": "separate learning rate for freshly initialised compressor modules; None keeps the single global rate"})

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

class SECO(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            trust_remote_code=True
        )
        self.training_mode = model_args.train
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            trust_remote_code=True
        )
        # the top pretrained layer, captured before LoRA wrapping, serves as the prompt pooler
        pooler_layer = copy.deepcopy(self.decoder.model.layers[-1])
        self.decoder = get_peft_model(self.decoder, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.encoder.config.vocab_size
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.dim = self.encoder.config.hidden_size
        self.compress_ratio = model_args.compress_ratio
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # prompt pooler: top pretrained Transformer layer (frozen) + LoRA + learnable seed queries,
        # replaces mean pooling for fusing prompt tokens into representative vector(s).
        # pool_method: seed (1 learnable query, original), mean (token mean, pre-pooler baseline),
        # last (last-token hidden state), multi_seed (m learnable queries, relevance = max over seeds)
        self.pool_method = model_args.pool_method
        self.num_pool_seeds = model_args.num_pool_seeds
        self.pooler = pooler_layer
        self.pooler.requires_grad_(False)
        pooler_lora_config = copy.deepcopy(lora_config)
        pooler_lora_config.target_modules = training_args.pooler_target_modules
        inject_adapter_in_model(pooler_lora_config, self.pooler)
        num_seeds = self.num_pool_seeds if self.pool_method == "multi_seed" else 1
        self.pooler_seed = nn.Parameter(
            self.decoder.get_input_embeddings().weight[self.bos_id]
            .detach().clone().float().unsqueeze(0).repeat(num_seeds, 1)
        )
        # learnable temperature for the cluster softmax weights: raw cosine similarities
        # live in a narrow band so softmax(cos) is near-uniform (see experiments);
        # a temperature < 1 sharpens the relevance weighting and restores gradient flow
        self.pooler_temperature = nn.Parameter(torch.tensor(float(model_args.pool_temp_init)))

        # compression route: "legacy" keeps the top-k centre / hard-cluster behaviour
        # above unchanged; "multiscale_budget_v1" delegates to compression_modules.
        self.compressor_version = model_args.compressor_version
        self.budget_mode = model_args.budget_mode
        self.diversity_loss_weight = float(model_args.diversity_loss_weight)
        if self.compressor_version == "context_adaptive_v1":
            if self.budget_mode not in BUDGET_MODES:
                raise ValueError(f"unknown budget_mode: {self.budget_mode!r}")
            self.compressor = ContextCompressor(
                self.dim, relevance_dim=model_args.relevance_dim,
                num_slots=model_args.query_slots,
                phrase_widths=model_args.query_phrase_widths,
                window_widths=model_args.context_window_widths,
                block_width=model_args.allocation_block_width,
                fusion_tau_init=model_args.query_fusion_temperature,
                attention_mode=model_args.query_attention_mode,
                attn_tau_init=model_args.query_attn_temperature,
                query_mode=model_args.query_mode,
                diversity_mode=model_args.diversity_mode,
                boundary_mode=model_args.context_boundary_mode,
                context_budget_mode=model_args.context_budget_mode,
                merge_mode=model_args.context_merge_mode,
                novelty_weight=model_args.context_novelty_weight,
                boundary_radius=model_args.context_boundary_radius,
                anchor_weight=model_args.context_anchor_weight,
            )
        elif self.compressor_version != "legacy":
            if self.compressor_version != "multiscale_budget_v1":
                raise ValueError(f"unknown compressor_version: {self.compressor_version!r}")
            if self.budget_mode not in BUDGET_MODES:
                raise ValueError(f"unknown budget_mode: {self.budget_mode!r}")
            self.compressor = MultiscaleCompressor(
                self.dim,
                relevance_dim=model_args.relevance_dim,
                num_slots=model_args.query_slots,
                phrase_widths=model_args.query_phrase_widths,
                window_widths=model_args.context_window_widths,
                block_width=model_args.allocation_block_width,
                raw_fraction=model_args.raw_memory_fraction,
                fusion_tau_init=model_args.query_fusion_temperature,
                assignment_tau_init=model_args.assignment_temperature,
                merge_tau_init=model_args.merge_temperature,
                position_penalty=model_args.position_penalty,
                anchor_residual_init=model_args.anchor_residual_init,
                coverage_weight=model_args.coverage_weight,
                redundancy_weight=model_args.redundancy_weight,
                selection_mode=model_args.selection_mode,
                attention_mode=model_args.query_attention_mode,
                attn_tau_init=model_args.query_attn_temperature,
                query_mode=model_args.query_mode,
                diversity_mode=model_args.diversity_mode,
                merge_gate_mode=model_args.merge_gate_mode,
            )
        else:
            self.compressor = None
        if model_args.legacy_budget_mode not in BUDGET_MODES:
            raise ValueError(
                f"unknown legacy_budget_mode: {model_args.legacy_budget_mode!r}"
            )
        self.legacy_budget_mode = model_args.legacy_budget_mode

        if self.training_mode:
            self.init()

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            if self.compressor_version == "context_adaptive_v1":
                from compressor_config import check_route
                check_route(self.training_args.restore_from, self.model_args)
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if self.compressor is not None:
                # new compressor weights must round-trip; silently training a
                # randomly initialised compressor would invalidate the run
                compressor_missing = [k for k in missing if k.startswith("compressor.")]
                if compressor_missing:
                    raise RuntimeError(
                        f"checkpoint {self.training_args.restore_from} is missing "
                        f"{len(compressor_missing)} compressor parameters, e.g. "
                        f"{compressor_missing[:3]}; a legacy checkpoint cannot warm-start "
                        f"the multiscale compressor"
                    )
            if unexpected:
                print(f"Ignoring {len(unexpected)} unexpected keys in the checkpoint")
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def tokens_to_embeddings(self, token_ids):
        base_embeds = self.encoder.get_base_model().model.embed_tokens(token_ids)
        return base_embeds

    def _encode(self, inputs_embeds, attention_mask):
        """Run the encoder backbone and return only its final hidden states.

        The LM head (vocab-sized) and the per-layer hidden states are both
        unneeded here, so call the backbone directly. LoRA adapters stay active
        because they are injected into the modules this call traverses.
        """
        if self.training_args.encoder_last_hidden_only:
            backbone = self.encoder.get_base_model().model
            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            return outputs.last_hidden_state
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _pool_prompt(self, prompt_tokens):
        # fuse a set of token hidden states into representative vector(s), shape (m, d);
        # m = 1 for seed/mean/last, m = num_pool_seeds for multi_seed
        if self.pool_method == "mean":
            return prompt_tokens.mean(dim=0, keepdim=True)
        if self.pool_method == "last":
            return prompt_tokens[-1:].clone()
        # seed / multi_seed: seeds are appended last so each seed attends to every token
        seeds = self.pooler_seed.to(prompt_tokens.dtype)                  # (m, d)
        x = torch.cat([prompt_tokens, seeds], dim=0).unsqueeze(0)         # (1, n+m, d)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out = self.pooler(x, position_ids=position_ids, use_cache=False)[0]
        return out[0, -seeds.size(0):]                             # (m, d)
    # seco的消融3 直接选择Uniform Sampling
    # def _compress_one(self, hidden_states, prompt_mask, attention_mask):
    #     valid_mask = attention_mask.bool()
    #     prompt_mask = prompt_mask.bool() & valid_mask
    #     input_mask = (~prompt_mask) & valid_mask

    #     prompt_tokens = hidden_states[prompt_mask]
    #     input_tokens = hidden_states[input_mask]

    #     if input_tokens.size(0) == 0:
    #         return hidden_states[valid_mask], torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool)
        
    #     if prompt_tokens.size(0) > 0:
    #         prompt_repr = self._pool_prompt(prompt_tokens)
    #     else:
    #         prompt_repr = self._pool_prompt(input_tokens)
    #     if prompt_repr.size(0) == 1:
    #         sim_to_prompt = self.cos(input_tokens, prompt_repr[0])
    #     else:
    #         sim_to_prompt = self.cos(input_tokens.unsqueeze(1), prompt_repr.unsqueeze(0)).max(dim=1).values
    #     # sim_to_prompt = self.cos(input_tokens, prompt_repr)
    #     # pprint("sim_to_prompt:", sim_to_prompt)

    #     # Uniform sampling for cluster centers: pick middle of each block
    #     R = self.compress_ratio
    #     L = input_tokens.size(0)
    #     center_indices = []
    #     for start in range(0, L, R):
    #         end = min(start + R, L)
    #         center_indices.append(start + (end - start - 1) // 2)
    #     center_indices = torch.tensor(center_indices, device=hidden_states.device, dtype=torch.long)
    #     K = center_indices.size(0)
    #     # pprint("K (uniform):", K)
    #     # pprint("center_indices:", center_indices)
    #     cluster_centers = input_tokens[center_indices]
    #     non_center_mask = torch.ones(input_tokens.size(0), dtype=torch.bool, device=hidden_states.device)
    #     non_center_mask[center_indices] = False
    #     non_center_tokens = input_tokens[non_center_mask]

    #     if non_center_tokens.size(0) > 0:
    #         sim_to_centers = self.cos(non_center_tokens.unsqueeze(1), cluster_centers.unsqueeze(0))
    #         best_cluster = sim_to_centers.argmax(dim=1)
    #         sim_prompt_non = sim_to_prompt[non_center_mask].unsqueeze(1)
    #         scores = sim_to_centers * sim_prompt_non
    #         clustered = []
    #         for k in range(K):
    #             members = [cluster_centers[k:k+1]]
    #             weights = [sim_to_prompt[center_indices[k]].unsqueeze(0)]

    #             idx = (best_cluster == k).nonzero(as_tuple=True)[0]
    #             if idx.numel() > 0:
    #                 members.append(non_center_tokens[idx])
    #                 weights.append(scores[idx, k])

    #             vecs = torch.cat(members, dim=0)
    #             weight = F.softmax(torch.cat(weights), dim=0).unsqueeze(-1)
    #             clustered.append((vecs * weight).sum(dim=0, keepdim=True))
    #         compressed_input = torch.cat(clustered, dim=0)
    #     else:
    #         compressed_input = cluster_centers

    #     combined = torch.cat([compressed_input, prompt_tokens], dim=0)
    #     # pprint("combined shape:", combined.shape)
    #     return combined, torch.ones(combined.size(0), device=hidden_states.device, dtype=torch.bool)

    def _compress_one(self, hidden_states, prompt_mask, attention_mask,
                      return_trace=False):
        valid_mask = attention_mask.bool()
        prompt_mask = prompt_mask.bool() & valid_mask
        input_mask = (~prompt_mask) & valid_mask

        prompt_tokens = hidden_states[prompt_mask]
        input_tokens = hidden_states[input_mask]

        length = input_tokens.size(0)
        if length == 0:
            return hidden_states[valid_mask], torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool), None

        # the historical legacy rule ignores --budget_mode entirely (ceil, floor
        # of two); legacy_budget_mode makes the same-K comparison possible
        # without silently changing what an already-trained legacy checkpoint
        # computes, since "legacy" is and stays the default
        K = resolve_budget(
            input_tokens.size(0), self.compress_ratio, self.legacy_budget_mode
        )[0]
        if K <= 0:
            return (
                hidden_states[valid_mask],
                torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool),
                None,
            )
        if prompt_tokens.size(0) > 0:
            prompt_repr = self._pool_prompt(prompt_tokens)
        else:
            prompt_repr = self._pool_prompt(input_tokens)

        # per-token relevance to the prompt: max over the m prompt vectors
        if prompt_repr.size(0) == 1:
            sim_to_prompt = self.cos(input_tokens, prompt_repr[0])
        else:
            sim_to_prompt = self.cos(input_tokens.unsqueeze(1), prompt_repr.unsqueeze(0)).max(dim=1).values
        K = min(K, input_tokens.size(0))
        _, center_indices = torch.topk(sim_to_prompt, K, largest=True)
        cluster_centers = input_tokens[center_indices]
        non_center_mask = torch.ones(input_tokens.size(0), dtype=torch.bool, device=hidden_states.device)
        non_center_mask[center_indices] = False
        non_center_tokens = input_tokens[non_center_mask]

        # positions of the non-center tokens inside the original context, so a
        # probe can report which context tokens each cluster actually absorbed
        non_center_positions = non_center_mask.nonzero(as_tuple=True)[0]
        members_by_cluster = {}
        merge_weights = {}
        merge_entropy_norm = []
        if non_center_tokens.size(0) > 0:
            sim_to_centers = self.cos(non_center_tokens.unsqueeze(1), cluster_centers.unsqueeze(0))
            best_cluster = sim_to_centers.argmax(dim=1)
            sim_prompt_non = sim_to_prompt[non_center_mask].unsqueeze(1)
            scores = sim_to_centers * sim_prompt_non
            clustered = []
            for k in range(K):
                members = [cluster_centers[k:k+1]]
                weights = [sim_to_prompt[center_indices[k]].unsqueeze(0)]

                idx = (best_cluster == k).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    members.append(non_center_tokens[idx])
                    weights.append(scores[idx, k])
                members_by_cluster[k] = [int(center_indices[k])] + [
                    int(p) for p in non_center_positions[idx]
                ]

                vecs = torch.cat(members, dim=0)
                weight = F.softmax(torch.cat(weights) / self.pooler_temperature.clamp_min(1e-3), dim=0)
                if return_trace:
                    column = weight.detach().float().cpu()
                    entropy = -(column * column.clamp_min(1e-12).log()).sum().item()
                    merge_weights[k] = column
                    merge_entropy_norm.append(
                        entropy / math.log(column.numel()) if column.numel() > 1 else 0.0
                    )
                clustered.append((vecs * weight.unsqueeze(-1)).sum(dim=0, keepdim=True))
            compressed_input = torch.cat(clustered, dim=0)
        else:
            compressed_input = cluster_centers
            members_by_cluster = {k: [int(center_indices[k])] for k in range(K)}

        combined = torch.cat([compressed_input, prompt_tokens], dim=0)
        anchors = sorted(int(i) for i in center_indices)
        # legacy keeps centres in top-k score order; record the order too
        trace = {
            "context_tokens": length,
            "memory_tokens": K,
            "selection_mode": "legacy_topk_centres",
            "anchors": anchors,
            "anchor_types": ["merge"] * K,
            "block_ranges": [(0, length)],
            "block_allocation": [K],
            "raw_positions": [],
            "merge_members": members_by_cluster,
            "center_order": [int(i) for i in center_indices],
            "merge_weights": merge_weights,
            "merge_entropy_normalised": merge_entropy_norm,
        }
        return combined, torch.ones(combined.size(0), device=hidden_states.device, dtype=torch.bool), trace

    def _compress_one_v1(self, hidden_states, prompt_mask, attention_mask,
                         return_trace=False):
        """Multiscale budgeted compression: exactly K memory slots plus the query."""
        valid_mask = attention_mask.bool()
        prompt_mask = prompt_mask.bool() & valid_mask
        input_mask = (~prompt_mask) & valid_mask

        prompt_tokens = hidden_states[prompt_mask]
        input_tokens = hidden_states[input_mask]

        if input_tokens.size(0) == 0:
            return (
                hidden_states[valid_mask],
                torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool),
                None,
            )

        budget, feasible = resolve_budget(
            input_tokens.size(0), self.compress_ratio, self.budget_mode
        )
        memory, trace = self.compressor(
            input_tokens, prompt_tokens, budget, return_trace=return_trace
        )
        # the compressor reports the budget it was handed, not whether that
        # budget is achievable; strict_context is infeasible for L < R
        trace["budget_feasible"] = feasible
        trace["realized_context_ratio"] = (
            input_tokens.size(0) / budget if budget > 0 else None
        )
        combined = torch.cat([memory, prompt_tokens], dim=0)
        return (
            combined,
            torch.ones(combined.size(0), device=hidden_states.device, dtype=torch.bool),
            trace,
        )

    def _compress_batch(self, hidden_states, prompt_mask, attention_mask,
                        return_traces=False):
        """Compress every sample and return *unpadded* per-sample tensors.

        Padding stays out of this function on purpose. Right-padding a short
        sample's prefix to the batch maximum and appending the answers after
        that maximum puts the first answer token in a padding slot, predicted
        from the padding position instead of the end of the query. The caller
        rebuilds the sequence per sample (:meth:`_build_training_batch`) and pads
        once, at the end.
        """
        batch_compressed = []
        batch_masks = []
        div_losses = []
        traces = []

        use_v1 = self.compressor is not None
        for b in range(hidden_states.size(0)):
            if use_v1:
                _compressed, mask, trace = self._compress_one_v1(
                    hidden_states[b], prompt_mask[b], attention_mask[b],
                    return_trace=return_traces,
                )
            else:
                _compressed, mask, trace = self._compress_one(
                    hidden_states[b], prompt_mask[b], attention_mask[b]
                )
            batch_compressed.append(_compressed)
            batch_masks.append(mask)
            if trace is not None:
                traces.append(trace)
                div_loss = trace.get("div_loss")
                if div_loss is not None:
                    div_losses.append(div_loss)

        div_loss = None
        if div_losses:
            div_loss = torch.stack(div_losses).mean()
        if return_traces:
            return batch_compressed, batch_masks, div_loss, traces
        return batch_compressed, batch_masks, div_loss

    @staticmethod
    def _pad_to(rows, length, value):
        """Stack ``rows`` into a ``(B, length, ...)`` tensor, padding the tail."""
        padded = []
        for row in rows:
            pad = length - row.size(0)
            if pad > 0:
                fill = row.new_full((pad,) + tuple(row.shape[1:]), value)
                row = torch.cat([row, fill], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    def _pad_compressed_batch(self, compressed_embeds, compressed_masks):
        """Dense ``(B, max_len, d)`` + bool mask for the generation path."""
        length = max(row.size(0) for row in compressed_embeds)
        embeds = self._pad_to(compressed_embeds, length, 0.0)
        masks = self._pad_to(compressed_masks, length, False)
        return embeds, masks

    def _build_training_batch(self, compressed_embeds, compressed_masks, labels):
        """Glue each sample's answer to its own valid prefix, then pad the tail.

        Returns ``(inputs_embeds, attention_mask, labels)`` where row ``b`` reads
        ``[prefix_b ; answer_b ; padding]``. Both the position ids and the causal
        shift of the first answer token are therefore about the end of sample
        ``b``'s own query, independently of the other samples in the batch.
        """
        d_model = compressed_embeds[0].size(-1)
        device = compressed_embeds[0].device
        pad_token_id = self.tokenizer.pad_token_id

        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = pad_token_id
        label_embeds = self.decoder.get_input_embeddings()(safe_labels)

        rows = []
        label_rows = []
        answer_lengths = []
        prefix_lengths = []
        for b in range(len(compressed_embeds)):
            valid = labels[b] != -100          # right-padded, so this is a prefix
            prefix = compressed_embeds[b][compressed_masks[b]]
            answer = label_embeds[b][valid]
            rows.append(torch.cat([prefix, answer], dim=0))
            label_rows.append(
                torch.cat(
                    [
                        torch.full((prefix.size(0),), -100, dtype=labels.dtype, device=device),
                        labels[b][valid],
                    ],
                    dim=0,
                )
            )
            prefix_lengths.append(int(prefix.size(0)))
            answer_lengths.append(int(valid.sum()))

        total = max(row.size(0) for row in rows)
        inputs_embeds = self._pad_to(rows, total, 0.0)
        full_labels = self._pad_to(label_rows, total, -100)
        lengths = torch.tensor(
            [p + a for p, a in zip(prefix_lengths, answer_lengths)], device=device
        )
        attention_mask = (
            torch.arange(total, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        assert inputs_embeds.size(-1) == d_model
        return inputs_embeds, attention_mask, full_labels

    def forward(
        self,
        input_ids=None,
        prompt_mask=None,
        attention_mask=None,
        labels=None,
    ):
        inputs_embeds = self.tokens_to_embeddings(input_ids)
        hidden_states = self._encode(inputs_embeds, attention_mask)

        compressed_embeds, compressed_masks, div_loss = self._compress_batch(
            hidden_states, prompt_mask, attention_mask
        )
        if labels is not None:
            full_embeds, full_mask, full_labels = self._build_training_batch(
                compressed_embeds, compressed_masks, labels
            )
            outputs = self.decoder(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=full_labels,
                return_dict=True
            )
            if div_loss is not None and self.diversity_loss_weight > 0:
                outputs.loss = outputs.loss + self.diversity_loss_weight * div_loss
                outputs.div_loss = div_loss.detach()
            return outputs
        else:
            compressed_embeds, compressed_mask = self._pad_compressed_batch(
                compressed_embeds, compressed_masks
            )
            compressed_attention_mask = compressed_mask.long()
            outputs = self.decoder.generate(
                inputs_embeds=compressed_embeds,
                attention_mask=compressed_attention_mask,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_id,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=GENERATION_REPETITION_PENALTY,
                use_cache=True
            )
            return outputs

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)
        # self.decoder.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)
        # self.decoder.gradient_checkpointing_disable(*args, **kwargs)
