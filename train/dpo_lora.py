"""Optional DPO on (feasible, infeasible) plan pairs.

Same hardware story as `sft_lora.py`: a 24 GB CUDA GPU, not the Mac. The
preference file is the `.dpo.jsonl` written by `scripts/generate_data.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA DPO for the TAMP planner")
    parser.add_argument("--config", default="configs/dpo_lora.json")
    parser.add_argument("--data", default=None, help="overrides config data_path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.data:
        cfg["data_path"] = args.data
    rows = []
    with Path(cfg["data_path"]).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        if not {"prompt", "chosen", "rejected"} <= set(row):
            raise SystemExit("DPO rows need prompt, chosen, rejected")
    print(f"dpo pairs: {len(rows)}")
    if args.dry_run:
        print("dry-run ok")
        return
    _train(cfg, rows, force_cpu=args.force_cpu)


def _train(cfg: dict, rows: list[dict], *, force_cpu: bool) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not force_cpu:
        raise SystemExit("DPO fine-tuning requires CUDA. Pass --dry-run to validate the file.")
    try:
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:
        raise SystemExit(f"trl is required for DPO ({exc}). pip install -e '.[train]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    dataset = Dataset.from_list(rows)
    training = DPOConfig(
        output_dir=cfg.get("output_dir", "runs/dpo"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("grad_accum", 8)),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        max_steps=int(cfg.get("max_steps", 200)),
        beta=float(cfg.get("beta", 0.1)),
        max_length=int(cfg.get("max_length", 1024)),
        bf16=True,
        report_to="none",
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model=model,
        args=training,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.get("output_dir", "runs/dpo"))


if __name__ == "__main__":
    main()
