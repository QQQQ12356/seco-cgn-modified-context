import hashlib
import json
import os
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from peft import LoraConfig
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser

from compressor_config import (
    ROUTE_KEYS,
    check_route,
    fatal_missing_compressor_weights,
)
from eval_utils import compute_exact, compute_f1, normalize_answer
from modeling_seco_cluster import (
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_REPETITION_PENALTY,
    SECO,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from sample_ids import stable_sample_id


ood_subsets = {"DROP", "BioASQ", "DuoRC.ParaphraseRC", "TextbookQA", "RelationExtraction", "RACE"}
id_subsets = {"SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"}


class JsonlDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def barrier_if_needed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def load_checkpoint(model, restore_from, model_args=None):
    if not restore_from:
        return

    # refuse to report a number for an algorithm these weights were not trained
    # with; several routes share identical tensor shapes
    if model_args is not None:
        check_route(restore_from, model_args)

    print(f"Loading checkpoint: {restore_from}")
    if restore_from.endswith(".safetensors"):
        state_dict = load_file(restore_from)
    else:
        state_dict = torch.load(restore_from, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if getattr(model, "compressor", None) is not None:
        compressor_missing = fatal_missing_compressor_weights(model.compressor, missing)
        if compressor_missing:
            raise RuntimeError(
                f"{restore_from} has no multiscale compressor weights "
                f"({len(compressor_missing)} missing, e.g. {compressor_missing[:3]}); "
                "evaluating a randomly initialised compressor on this checkpoint "
                "would not measure the trained model"
            )
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected checkpoint keys")


def file_sha256(path, chunk=1 << 20):
    """Content hash of a file, or ``None`` if it is not readable."""
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint():
    """Hash of the source files that decide what an evaluation computes.

    The v1 review notes that an evaluation directory on its own cannot prove
    which code produced it, so "these two runs differ only by X" rests on a
    recollection. Recording this makes that claim checkable, or visibly
    unsupported.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    files = ["modeling_seco_cluster.py", "compression_modules.py",
             "context_compression.py",
             "compressor_config.py", "ft_inference_all.py", "eval_utils.py"]
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update((file_sha256(os.path.join(root, name)) or "missing").encode())
    return digest.hexdigest()


def build_manifest(model_args, data_args, training_args, restore_from,
                   test_file, num_samples, world_size, dataset):
    """Everything needed to say what this number is a number about."""
    route = {key: getattr(model_args, key, None) for key in ROUTE_KEYS}
    return {
        "checkpoint": os.path.abspath(restore_from) if restore_from else None,
        "checkpoint_sha256": file_sha256(restore_from),
        "source_sha256": source_fingerprint(),
        "model_name_or_path": model_args.model_name_or_path,
        # the generation settings the decoder actually used, from the module
        # constants -- not the unused model_args defaults
        "generation": {
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
            "repetition_penalty": GENERATION_REPETITION_PENALTY,
            "do_sample": False,
            "temperature": 0.0,
        },
        "route": route,
        "test_file": os.path.abspath(test_file),
        "test_file_sha256": file_sha256(test_file),
        "eval_samples": num_samples,
        "num_rows_loaded": len(dataset),
        "sample_ids_sha256": hashlib.sha256(
            "".join(sorted(stable_sample_id(row) for row in dataset)).encode()
        ).hexdigest(),
        "seed": training_args.seed,
        "world_size": world_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "dtype": "bfloat16" if training_args.bf16 else "float16",
        "encoder_last_hidden_only": training_args.encoder_last_hidden_only,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_eval_dataset(test_file, max_samples, shuffle_eval, seed):
    with open(test_file, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    if shuffle_eval:
        rng = random.Random(seed)
        rng.shuffle(data)

    if max_samples > 0:
        data = data[: min(max_samples, len(data))]
    return data


def encode_context_and_query(tokenizer, context, question, max_length):
    """Tokenise exactly like training: context then query, no special tokens.

    Training builds the sequence from two independent ``add_special_tokens=False``
    encodes, so inference has to do the same or the encoder sees a different
    token sequence, boundary and prompt mask than the one it was trained on.
    The query is protected first; the context absorbs any truncation.
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    query_ids = tokenizer.encode(question, add_special_tokens=False)
    if len(query_ids) >= max_length:
        raise ValueError(
            f"query alone is {len(query_ids)} tokens, exceeding model_max_length={max_length}; "
            "refusing to silently truncate the question"
        )
    context_ids = context_ids[: max_length - len(query_ids)]
    input_ids = context_ids + query_ids
    prompt_mask = [0] * len(context_ids) + [1] * len(query_ids)
    return input_ids, prompt_mask


def collate_batch(features, tokenizer, max_length):
    encoded_ids = []
    prompt_masks = []
    for item in features:
        input_ids, prompt_mask = encode_context_and_query(
            tokenizer, item["input"], item["prompt"], max_length
        )
        encoded_ids.append(torch.tensor(input_ids, dtype=torch.long))
        prompt_masks.append(torch.tensor(prompt_mask, dtype=torch.long))

    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        encoded_ids, batch_first=True, padding_value=pad_id
    )
    prompt_mask = torch.nn.utils.rnn.pad_sequence(
        prompt_masks, batch_first=True, padding_value=0
    )
    # explicit length mask: deriving it from token ids would drop a legitimate EOS
    # whenever pad_token_id == eos_token_id, which is exactly the current setup
    lengths = torch.tensor([ids.size(0) for ids in encoded_ids])
    attention_mask = (
        torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    ).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": prompt_mask,
        "raw_batch": features,
    }


def compute_metrics(results):
    all_em = []
    all_f1 = []
    subset_stats = defaultdict(lambda: {"em_scores": [], "f1_scores": [], "count": 0})
    ood_em_scores = []
    ood_f1_scores = []
    id_em_scores = []
    id_f1_scores = []
    sample_details = []

    for item in results:
        prediction = item["model_output"]
        ground_truth = item["ground_truth"]
        subset = item["subset"]

        if isinstance(ground_truth, list):
            em_score = max(
                compute_exact(normalize_answer(gt), normalize_answer(prediction))
                for gt in ground_truth
            )
            f1_score = max(
                compute_f1(normalize_answer(prediction), normalize_answer(gt))
                for gt in ground_truth
            )
        else:
            em_score = compute_exact(normalize_answer(ground_truth), normalize_answer(prediction))
            f1_score = compute_f1(normalize_answer(prediction), normalize_answer(ground_truth))

        all_em.append(em_score)
        all_f1.append(f1_score)
        subset_stats[subset]["em_scores"].append(em_score)
        subset_stats[subset]["f1_scores"].append(f1_score)
        subset_stats[subset]["count"] += 1

        if subset in ood_subsets:
            ood_em_scores.append(em_score)
            ood_f1_scores.append(f1_score)
        elif subset in id_subsets:
            id_em_scores.append(em_score)
            id_f1_scores.append(f1_score)

        sample_details.append(
            {
                "subset": subset,
                "sample_id": item.get("sample_id"),
                "prediction": prediction,
                "ground_truth": ground_truth,
                "em": em_score,
                "f1": f1_score,
            }
        )

    subset_summary = {}
    for subset, stats in subset_stats.items():
        em_scores = stats["em_scores"]
        f1_scores = stats["f1_scores"]
        subset_summary[subset] = {
            "count": stats["count"],
            "avg_em": (sum(em_scores) / len(em_scores)) if em_scores else 0,
            "avg_f1": (sum(f1_scores) / len(f1_scores)) if f1_scores else 0,
        }

    return {
        "summary": {
            "total_samples": len(all_em),
            "avg_em": (sum(all_em) / len(all_em)) if all_em else 0,
            "avg_f1": (sum(all_f1) / len(all_f1)) if all_f1 else 0,
            "ood_em": (sum(ood_em_scores) / len(ood_em_scores)) if ood_em_scores else 0,
            "ood_f1": (sum(ood_f1_scores) / len(ood_f1_scores)) if ood_f1_scores else 0,
            "id_em": (sum(id_em_scores) / len(id_em_scores)) if id_em_scores else 0,
            "id_f1": (sum(id_f1_scores) / len(id_f1_scores)) if id_f1_scores else 0,
        },
        "subset_summary": subset_summary,
        "details": sample_details,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.train = False

    distributed, rank, world_size, local_rank, device = setup_distributed()
    random.seed(training_args.seed + rank)
    torch.manual_seed(training_args.seed + rank)

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = SECO(model_args, training_args, lora_config)
    load_checkpoint(model, training_args.restore_from, model_args)
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    dataset = load_eval_dataset(
        data_args.test_file,
        data_args.eval_samples,
        data_args.shuffle_eval,
        training_args.seed,
    )

    if rank == 0:
        print(f"Loaded {len(dataset)} samples for inference.")
        print(f"World size: {world_size}, per-device batch size: {training_args.per_device_eval_batch_size}")

    shard = dataset[rank::world_size]
    dataloader = DataLoader(
        JsonlDataset(shard),
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, training_args.model_max_length),
    )

    model_tag = os.path.basename(model_args.model_name_or_path.rstrip("/"))
    checkpoint_num = int(training_args.restore_from.split("checkpoint-")[1].split("/")[0])
    run_name = (
        f"cr{model_args.compress_ratio}"
        f"_bs{training_args.per_device_eval_batch_size}"
        f"_samples{data_args.eval_samples}"
        f"_cp{checkpoint_num}"
    )
    output_dir = os.path.join(data_args.eval_output_dir, model_tag, run_name)
    os.makedirs(output_dir, exist_ok=True)

    part_path = os.path.join(output_dir, f"predictions.rank{rank}.jsonl")
    results = []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"Inference rank {rank}", disable=rank != 0)
        for batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)

            generated_ids = model(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                labels=None,
            )

            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for sample, prediction in zip(batch["raw_batch"], responses):
                results.append(
                    {
                        "subset": sample.get("subset", ""),
                        "sample_id": stable_sample_id(sample),
                        "context": sample["input"],
                        "question": sample["prompt"],
                        "model_output": prediction.strip(),
                        "ground_truth": sample.get("answer", ""),
                    }
                )

    with open(part_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    barrier_if_needed(distributed)

    if rank == 0:
        merged_results = []
        for shard_rank in range(world_size):
            shard_path = os.path.join(output_dir, f"predictions.rank{shard_rank}.jsonl")
            with open(shard_path, "r", encoding="utf-8") as handle:
                merged_results.extend(json.loads(line) for line in handle)

        metrics = compute_metrics(merged_results)
        merged_path = os.path.join(output_dir, "mrqa_inference_results.jsonl")
        metrics_path = os.path.join(output_dir, "mrqa_inference_metrics.json")

        with open(merged_path, "w", encoding="utf-8") as handle:
            for item in merged_results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

        # a metric without this cannot be reproduced, compared, or audited
        manifest = build_manifest(
            model_args, data_args, training_args, training_args.restore_from,
            data_args.test_file, data_args.eval_samples, world_size, dataset,
        )
        manifest_path = os.path.join(output_dir, "eval_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"Manifest saved to: {manifest_path}")

        print(f"Inference saved to: {merged_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"Final EM: {metrics['summary']['avg_em']:.4f}")
        print(f"Final F1: {metrics['summary']['avg_f1']:.4f}")
        print(f"OOD EM: {metrics['summary']['ood_em']:.4f}")
        print(f"OOD F1: {metrics['summary']['ood_f1']:.4f}")
        print(f"ID EM: {metrics['summary']['id_em']:.4f}")
        print(f"ID F1: {metrics['summary']['id_f1']:.4f}")

    barrier_if_needed(distributed)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
