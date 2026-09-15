import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer, TrainerCallback
from datasets import Dataset
import safetensors.torch

from compressor_config import write_config
from modeling_seco_cluster import (
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_REPETITION_PENALTY,
)

class InstructFTTokenizeFunction:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        all_input_ids = []
        all_prompt_masks = []
        all_labels = []
        eos_token = self.tokenizer.eos_token
        for i, p, a in zip(examples["input"], examples["prompt"], examples["answer"]):
            input_ids = self.tokenizer.encode(f"{i}", add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(f"{p}", add_special_tokens=False)
            if isinstance(a, list):
                answer_ids = self.tokenizer.encode(f"{a[0]}{eos_token}", add_special_tokens=False)
            else:
                answer_ids = self.tokenizer.encode(f"{a}{eos_token}", add_special_tokens=False)
            # same length policy as inference (ft_inference_all.encode_context_and_query):
            # the query is protected, the context absorbs the truncation. Without
            # this, training would feed the encoder a longer context than the one
            # inference sees, shifting the prompt-mask boundary and evidence set.
            if len(prompt_ids) >= self.max_length:
                raise ValueError(
                    f"query alone is {len(prompt_ids)} tokens, exceeding "
                    f"model_max_length={self.max_length}"
                )
            input_ids = input_ids[: self.max_length - len(prompt_ids)]

            combined_input = input_ids + prompt_ids
            p_mask = [0] * len(input_ids) + [1] * len(prompt_ids)

            all_input_ids.append(torch.tensor(combined_input))
            all_prompt_masks.append(torch.tensor(p_mask))
            all_labels.append(torch.tensor(answer_ids))

        return {
            "input_ids": all_input_ids,
            "prompt_mask": all_prompt_masks,
            "labels": all_labels
        }


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        prompt_mask = [torch.tensor(f["prompt_mask"]) for f in features]

        batch_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        batch_prompt_mask = pad_sequence(prompt_mask, batch_first=True, padding_value=0)
        # explicit length mask, matching inference. Deriving it from
        # `input_ids != pad_token_id` would mask a legitimate EOS wherever it
        # appears, because pad_token_id == eos_token_id in this setup.
        lengths = torch.tensor([ids.size(0) for ids in input_ids]).unsqueeze(1)
        attention_mask = torch.arange(batch_input_ids.size(1)).unsqueeze(0) < lengths

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "prompt_mask": batch_prompt_mask,
            "attention_mask": attention_mask,
        }


# compressor submodules, each logged separately. A single "compressor" norm is
# dominated by whichever submodule has the most parameters, so it cannot show
# that (say) the query slots are frozen while the merger trains.
_COMPRESSOR_SUBMODULES = (
    ("query_proj", ("query_encoder.q_norm", "query_encoder.proj")),
    ("query_slots", ("query_encoder.slots",)),
    ("query_gate", ("query_encoder.scale_gate", "query_encoder.res_proj",
                    "query_encoder.phrase_proj", "query_encoder.global_proj")),
    ("query_temp", ("query_encoder.raw_attn_tau",)),
    ("scorer", ("scorer.",)),
    ("merger", ("merger.",)),
)


def _param_groups(model):
    """Group SECO parameters by module for gradient-health logging.

    Groups: pooler_seed (learnable prompt queries), pooler_lora (pooler layer
    adapters), encoder_lora / decoder_lora (LoRA controls), the whole compressor,
    and one group per compressor submodule.
    """
    groups = {
        "pooler_seed": [], "pooler_lora": [], "encoder_lora": [],
        "decoder_lora": [], "compressor": [],
    }
    for group, _ in _COMPRESSOR_SUBMODULES:
        groups[group] = []
    for name, param in model.named_parameters():
        if name == "pooler_seed" or name == "pooler_temperature":
            groups["pooler_seed"].append(param)
        elif name.startswith("compressor."):
            groups["compressor"].append(param)
            body = name[len("compressor."):]
            for group, prefixes in _COMPRESSOR_SUBMODULES:
                if body.startswith(prefixes):
                    groups[group].append(param)
                    break
            else:
                groups.setdefault("compressor_other", []).append(param)
        elif name.startswith("pooler.") and "lora_" in name:
            groups["pooler_lora"].append(param)
        elif name.startswith("encoder.") and "lora_" in name:
            groups["encoder_lora"].append(param)
        elif name.startswith("decoder.") and "lora_" in name:
            groups["decoder_lora"].append(param)
    return {k: v for k, v in groups.items() if v}


def _group_grad_norm(params):
    """Global L2 gradient norm over a parameter group; None if no grad at all."""
    total = None
    for p in params:
        if p.grad is None:
            continue
        sq = p.grad.detach().float().pow(2).sum()
        total = sq if total is None else total + sq
    return float(total.sqrt().item()) if total is not None else None


def _group_drift(params, snapshot):
    """L2 distance between current weights and the training-start snapshot."""
    total = None
    for p, p0 in zip(params, snapshot):
        sq = (p.detach().float() - p0.to(p.device)).pow(2).sum()
        total = sq if total is None else total + sq
    return float(total.sqrt().item()) if total is not None else 0.0


class PoolerGradientHealthCallback(TrainerCallback):
    """Logs gradient norms and weight drift for the prompt pooler vs. LoRA controls.

    Purpose: verify whether the prompt-pooling module actually receives a
    usable training signal (top-K selection is non-differentiable; gradients
    only flow through the softmax aggregation weights).
    """

    def __init__(self, model, log_path, log_every=10):
        self.model = model
        self.log_path = log_path
        self.log_every = log_every
        self.groups = _param_groups(model)
        self.snapshot = {
            name: [p.detach().float().cpu().clone() for p in params]
            for name, params in self.groups.items()
        }

    def on_optimizer_step(self, args, state, control, **kwargs):
        # fires after optimizer.step() but before zero_grad(): gradients are still intact
        if args.process_index != 0:
            return
        step = state.global_step + 1
        if step % self.log_every != 0 and step > 5:
            return
        record = {"step": step}
        for name, params in self.groups.items():
            record[f"{name}_grad_norm"] = _group_grad_norm(params)
            record[f"{name}_drift"] = _group_drift(params, self.snapshot[name])
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"grad_health/{k}": v for k, v in record.items() if k != "step"},
                          step=state.global_step)
        except ImportError:
            pass


# Parameters that are trained on top of the frozen base models and therefore
# have to survive a checkpoint round-trip. Everything else is restored from the
# pretrained weights at load time.
_TRAINED_KEY_SUFFIXES = (
    "pooler_seed",
    "pooler_temperature",
)


def _is_trained_parameter(name):
    """True for adapter and compressor parameters; false for frozen base weights."""
    if "lora_" in name:
        return True
    if name.startswith("compressor."):
        return True
    return name.endswith(_TRAINED_KEY_SUFFIXES)


class CustomTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        # keep only adapter weights (LoRA), the pooler parameters and the new
        # compressor modules to reduce checkpoint size; base weights stay frozen
        # and are restored from the pretrained model at load time
        state_dict = {k: v for k, v in state_dict.items() if _is_trained_parameter(k)}
        deduped = {}
        seen_data_ptrs = {}
        for k, v in state_dict.items():
            ptr = v.data_ptr()
            if ptr in seen_data_ptrs:
                deduped[k] = v.clone()
            else:
                seen_data_ptrs[ptr] = k
                deduped[k] = v
        safetensors.torch.save_file(deduped, os.path.join(output_dir, "model.safetensors"))
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)
        # record which compression route produced these weights; check_route()
        # refuses to evaluate a checkpoint under a different route
        model_args = getattr(self.model, "model_args", None)
        if model_args is not None:
            compressor = getattr(self.model, "compressor", None)
            state = getattr(self, "state", None)
            tokenizer = getattr(self.model, "tokenizer", None)
            write_config(output_dir, model_args, extra={
                "compressor_parameter_count": (
                    sum(p.numel() for p in compressor.parameters())
                    if compressor is not None else 0
                ),
                "global_step": getattr(state, "global_step", None),
                # without these a checkpoint cannot be reproduced or audited:
                # the same weights under a different schedule/budget are a
                # different experiment
                "learning_rate": self.args.learning_rate,
                "new_module_lr": getattr(self.args, "new_module_lr", None),
                "max_steps": self.args.max_steps,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "lr_scheduler_type": str(self.args.lr_scheduler_type),
                "warmup_ratio": self.args.warmup_ratio,
                "weight_decay": self.args.weight_decay,
                "optim": self.args.optim,
                "seed": self.args.seed,
                "bf16": self.args.bf16,
                "train_samples": getattr(self.args, "train_samples", None),
                "encoder_last_hidden_only": getattr(
                    self.args, "encoder_last_hidden_only", None
                ),
                "tokenizer_policy": "context+query, add_special_tokens=False, context truncated first",
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                # the constants the decoder actually runs with, not the
                # (unused) model_args defaults
                "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
                "repetition_penalty": GENERATION_REPETITION_PENALTY,
                "generation_is_greedy": True,
            })

            
def _build_optimizers(model, training_args):
    """Give the freshly initialised compressor modules their own learning rate.

    The encoder/decoder LoRA adapters start from a pretrained model, while the
    compressor's projections start from noise; sharing one rate either starves
    the new modules or destabilises the adapters. Returns ``(None, None)`` to
    fall back to the Trainer's single-rate default.
    """
    new_lr = getattr(training_args, "new_module_lr", None)
    if new_lr is None:
        return (None, None)

    new_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (new_params if name.startswith("compressor.") else base_params).append(param)

    groups = [{"params": base_params, "lr": training_args.learning_rate}]
    if new_params:
        groups.append({"params": new_params, "lr": float(new_lr)})
    optimizer = torch.optim.AdamW(
        groups, lr=training_args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=training_args.weight_decay,
    )
    print(f"Optimizer groups: base={len(base_params)} @ {training_args.learning_rate}, "
          f"new_modules={len(new_params)} @ {new_lr}")
    # scheduler stays None so the Trainer builds its configured warmup/cosine
    return (optimizer, None)


def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    data_collator = DataCollatorForDynamicPadding(tokenizer.pad_token_id)

    optimizers = _build_optimizers(model, training_args)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=optimizers,
    )

    # gradient-health instrumentation for the prompt pooler (opt-in via env var)
    grad_log_path = os.environ.get("SECO_GRAD_LOG")
    if grad_log_path:
        log_every = int(os.environ.get("SECO_GRAD_LOG_EVERY", "10"))
        trainer.add_callback(PoolerGradientHealthCallback(model, grad_log_path, log_every))

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)