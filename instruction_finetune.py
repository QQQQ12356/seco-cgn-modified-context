import transformers
import wandb
from peft import LoraConfig
from datasets import load_dataset, load_from_disk
from modeling_seco_cluster import SECO, ModelArguments, DataArguments, TrainingArguments
from training_utils import InstructFTTokenizeFunction, train_model, DataCollatorForDynamicPadding
from transformers import AutoTokenizer

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_name = "seco_train"
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = data_args.train_file
    eval_file = data_args.test_file
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    print("Train size:", len(train_dataset))
    print("Eval size:", len(eval_dataset))

    train_dataset_length = data_args.train_samples
    test_dataset_length = data_args.eval_samples

    if data_args.debug_data:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(32, len(eval_dataset))))
    else:
        train_dataset = train_dataset.select(range(min(train_dataset_length, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(test_dataset_length, len(eval_dataset))))

    if training_args.local_rank <= 0:
        if data_args.debug_data:
            wandb.init(project=training_args.project_name, name=f"seoc_debug_{model_args.compress_ratio}")
        else:
            name = f"seco_{model_args.lora_r}_{training_args.max_steps}_{model_args.compress_ratio}"
            wandb.init(project=training_args.project_name, name=name)
    print(f"Dataset size: train={len(train_dataset)}, eval={len(eval_dataset)}")

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    tokenize_fn = InstructFTTokenizeFunction(tokenizer, training_args.model_max_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, batch_size=1000)

    model = SECO(model_args, training_args, lora_config)

    train_model(model, train_dataset, eval_dataset, training_args, tokenizer)

if __name__ == "__main__":
    main()