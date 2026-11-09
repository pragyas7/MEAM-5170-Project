"""LoRA supervised fine-tune of the task planner.

Designed for a single 24 GB GPU and the default config (Qwen2.5-7B-Instruct,
r=16, 400 steps, sequence length 1024, batch 1, grad accumulation 8, bf16,
gradient checkpointing). That fit is well under two hours on a 24 GB card.

This machine is a Mac. Run the real fit in Colab or on any CUDA box:

    pip install -e '.[train]'
    python train/sft_lora.py --config configs/sft_lora.json

`--dry-run` checks the JSONL and prints the config without loading a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_text_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        if "messages" in rec:
            messages = rec["messages"]
        elif "prompt" in rec and "completion" in rec:
            messages = [
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["completion"]},
            ]
        else:
            raise ValueError("SFT records need messages or prompt+completion")
        rows.append({"messages": messages})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for the TAMP planner")
    parser.add_argument("--config", default="configs/sft_lora.json")
    parser.add_argument("--data", default=None, help="overrides config data_path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-cpu", action="store_true", help="debug only; not the project setting")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.data:
        cfg["data_path"] = args.data
    records = read_records(Path(cfg["data_path"]))
    rows = to_text_rows(records)
    print(f"sft examples: {len(rows)}")
    print(json.dumps({k: cfg[k] for k in cfg if k != "target_modules"}, indent=2))
    if args.dry_run:
        print("dry-run ok")
        return
    _train(cfg, rows, force_cpu=args.force_cpu)


def _train(cfg: dict, rows: list[dict], *, force_cpu: bool) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available() and not force_cpu:
        raise SystemExit(
            "Refusing to fine-tune without CUDA. The project spec runs this on a 24 GB GPU. "
            "Pass --force-cpu only to debug the loop, or --dry-run to validate the data."
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    max_length = int(cfg.get("max_length", 1024))

    def tokenize(batch):
        texts = []
        prompt_lens = []
        for messages in batch["messages"]:
            if hasattr(tokenizer, "apply_chat_template"):
                full = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            else:
                full = "\n".join(m["content"] for m in messages)
                prompt = messages[0]["content"]
            texts.append(full)
            prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        labels = []
        for ids, plen in zip(tokenized["input_ids"], prompt_lens):
            masked = list(ids)
            for i in range(min(plen, len(masked))):
                masked[i] = -100
            labels.append(masked)
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_list(rows).map(tokenize, batched=True, remove_columns=["messages"])
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "runs/sft"),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        max_steps=int(cfg.get("max_steps", 400)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        bf16=bool(torch.cuda.is_available()),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
    # TRL's SFTTrainer is an alternative entry point when the extra dependency is
    # installed. The peft Trainer above is the one this script actually runs:
    # its arguments have been stable for longer than TRL's SFTConfig.
    _ = _trl_available()
    trainer.train()
    trainer.save_model(cfg.get("output_dir", "runs/sft"))
    tokenizer.save_pretrained(cfg.get("output_dir", "runs/sft"))


def _trl_available() -> bool:
    try:
        import trl  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
