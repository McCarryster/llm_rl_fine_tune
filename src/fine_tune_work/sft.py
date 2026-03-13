from datasets import load_dataset
from typing import List
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import sft_cfg as cfg
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import os


def load_streaming_dataset(dataset_dir: str):
    arrow_files: List[str] = []
    for shard in sorted(os.listdir(dataset_dir)):
        shard_path = os.path.join(dataset_dir, shard)
        for f in os.listdir(shard_path):
            if f.endswith(".arrow"):
                arrow_files.append(os.path.join(shard_path, f))
 
    print(f"[Dataset] Found {len(arrow_files)} arrow shard(s) in {dataset_dir}")
    return load_dataset("arrow", data_files=arrow_files, split="train", streaming=True)


def run_oom_probe(model, batch_size: int, eval_batch_size: int, max_seq_len: int, device: str = "cuda") -> None:
    print(f"\n[OOM Probe] train_batch={batch_size}, eval_batch={eval_batch_size}, seq_len={max_seq_len}")

    def make_batch(bs):
        return (
            torch.ones(bs, max_seq_len, dtype=torch.long, device=device),
            torch.ones(bs, max_seq_len, dtype=torch.long, device=device),
            torch.ones(bs, max_seq_len, dtype=torch.long, device=device),
        )

    try:
        # Training step — worst case memory state
        ids, labels, mask = make_batch(batch_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
        model.zero_grad(set_to_none=True)

        # Eval step — larger batch, no grad, but optimizer states still in VRAM
        ids, labels, mask = make_batch(eval_batch_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=mask, labels=labels)

        torch.cuda.empty_cache()
        print(f"[OOM Probe] ✓ Passed\n")

    except torch.OutOfMemoryError as e:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"[OOM Probe] ✗ Failed at eval_batch={eval_batch_size}\n"
            f"  → Reduce per_device_eval_batch_size\n"
            f"  Original error: {e}"
        )


def load_model_and_tokenizer(model_path: str, use_lora: bool = False) -> Any:

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, config)
        model.enable_input_require_grads() # type: ignore
        model.print_trainable_parameters()

    return model, tokenizer


def train_sft(
    model_name: str,
    train_dataset_path: str,
    test_dataset_path: str,
    output_dir: str,
    use_lora: bool = True,
) -> None:


    train_dataset = load_streaming_dataset(train_dataset_path)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    test_dataset = load_streaming_dataset(test_dataset_path)

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_lora=use_lora)
    run_oom_probe(model, batch_size=cfg.BATCH_SIZE, eval_batch_size=cfg.EVAL_BATCH_SIZE, max_seq_len=cfg.MAX_SEQ_LEN)

    # 4. Define train args
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=cfg.MAX_STEPS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LR,
        bf16=True,
        gradient_checkpointing=cfg.GRAD_CHECKPOINT,
        logging_steps=cfg.LOG_STEPS,
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        # eval_accumulation_steps=16,
        save_steps=2000,
        save_total_limit=2,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # 6. Start training
    trainer.train()

    # 7. Save fine tuned model and tokenizer
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))


if __name__ == "__main__":
    train_sft(
        model_name=cfg.MODEL_DIR,
        train_dataset_path=cfg.TRAIN_SFT_SET,
        test_dataset_path=cfg.TEST_SFT_SET,
        output_dir=cfg.OUTPUT_DIR,
        use_lora=True,
    )