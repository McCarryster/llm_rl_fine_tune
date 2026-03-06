from datasets import load_from_disk, Dataset
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import sft_cfg as cfg
import os


def convert_and_tokenize(sample, tokenizer):
    """
    1. Convert \n\nHuman:/\n\nAssistant: format to Qwen's chat template
    2. Tokenize and mask everything before last assistant response with -100
    """
    # Step 1. Convert to Qwen format
    text = sample["text"]
    messages = []

    segments = text.split("\n\nHuman: ")
    segments = [s for s in segments if s.strip()]

    for segment in segments:
        if "\n\nAssistant: " in segment:
            human_part, assistant_part = segment.split("\n\nAssistant: ", 1)
            messages.append({"role": "user",      "content": human_part.strip()})
            messages.append({"role": "assistant", "content": assistant_part.strip()})
        else:
            messages.append({"role": "user", "content": segment.strip()})

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    tokenized = tokenizer(
        formatted,
        truncation=True,
        add_special_tokens=False,
        max_length=cfg.MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[0]
    labels = torch.full_like(input_ids, -100)

    # Step 2. Mask everything before last assistant response
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_header_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    input_ids_list = input_ids.tolist()
    last_im_start = len(input_ids_list) - 1 - input_ids_list[::-1].index(im_start_id)
    response_start = last_im_start + 1 + len(assistant_header_ids)
    labels[response_start:] = input_ids[response_start:]

    return {
        "input_ids":      input_ids.tolist(),   # store as lists, not tensors
        "attention_mask": tokenized.attention_mask[0].tolist(),
        "labels":         labels.tolist(),
    }


def load_model_and_tokenizer(model_path: str, use_lora: bool = False) -> Any:

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
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
        model.print_trainable_parameters()

    return model, tokenizer


def train_sft(
    model_name: str,
    train_dataset_path: str,
    test_dataset_path: str,
    output_dir: str,
    use_lora: bool = True,
) -> None:

    # 1. Load and check data
    train_dataset = load_from_disk(train_dataset_path)
    test_dataset = load_from_disk(test_dataset_path)
    print(f"SFT train size: {len(train_dataset)}")
    print(f"SFT test size:  {len(test_dataset)}")
    print("\nExample datapoint:")
    print(train_dataset[1]["text"])
    print('#'*100)

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_lora=use_lora)

    # 3. Tokenize lazily using .map() — processes one batch at a time, no RAM spike
    from functools import partial
    tokenize_fn = partial(convert_and_tokenize, tokenizer=tokenizer)

    tokenized_train = train_dataset.map(
        tokenize_fn,
        remove_columns=train_dataset.column_names,  # type: ignore
        batched=False,
        num_proc=4,         # keep at 1 to avoid multiprocess memory issues
    )
    tokenized_test = test_dataset.map(
        tokenize_fn,
        remove_columns=test_dataset.column_names, # type: ignore
        batched=False,
        num_proc=4,
    )

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # 4. Define train args
    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_checkpointing=cfg.GRAD_CHECKPOINT,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LR,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=600,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    # 5. Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train, # type: ignore
        eval_dataset=tokenized_test, # type: ignore
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