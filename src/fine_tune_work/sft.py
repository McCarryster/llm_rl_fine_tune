# sft_train.py
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# 1. Load and check dataset
sft_train = load_from_disk("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/sft/train/sft_train")
sft_test = load_from_disk("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/sft/train/sft_test")

print(f"SFT train size: {len(sft_train)}")
print(f"SFT test size:  {len(sft_test)}")
print("\nExample datapoint:")
print(sft_train[1]["text"])


# 2. Load model & tokenizer
def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    show_progress: bool = True,
) -> Dataset:

    texts = []
    for item in tqdm(dataset, disable=not show_progress):
        texts.append(item["text"])

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        result = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = Dataset.from_dict({"text": texts}).map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    return tokenized


def load_model(
    model_name: str,
    use_lora: bool = True,
) -> AutoModelForCausalLM:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

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

    return model


def train_sft(
    model_name: str,
    train_dataset_path: str,
    test_dataset_path: str,
    output_dir: str,
    use_lora: bool = True,
    show_progress: bool = True,
) -> None:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_from_disk(train_dataset_path)
    test_dataset = load_from_disk(test_dataset_path)

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
        show_progress=show_progress,
    )

    tokenized_test = tokenize_dataset(
        test_dataset,
        tokenizer,
        show_progress=show_progress,
    )

    model = load_model(
        model_name=model_name,
        use_lora=use_lora,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=16,

        num_train_epochs=2,

        learning_rate=2e-5,

        logging_steps=10,
        eval_steps=200,
        save_steps=200,

        evaluation_strategy="steps",
        save_strategy="steps",

        fp16=True,

        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))


if __name__ == "__main__":

    train_sft(
        model_name="",
        train_dataset_path="",
        test_dataset_path="",
        output_dir="",
        use_lora=True,
        show_progress=True,
    )