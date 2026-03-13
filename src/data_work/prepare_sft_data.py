from datasets import load_dataset
from typing import Dict, List
from datasets import Dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def split_into_turns(sample):
    """Split a full conversation into individual turn datapoints."""
    text = sample["chosen"]
    datapoints = []
    
    # Split on Human turns
    # Each datapoint = prior context (prompt) + one assistant response
    segments = text.split("\n\nHuman: ")
    segments = [s for s in segments if s.strip()]  # remove empty

    context = ""
    for segment in segments:
        if "\n\nAssistant: " in segment:
            human_part, assistant_part = segment.split("\n\nAssistant: ", 1)
            
            # Build the full datapoint
            full_turn = context + "\n\nHuman: " + human_part + "\n\nAssistant: " + assistant_part
            datapoints.append({"text": full_turn})
            
            # Update context for next turn
            context = full_turn
    
    return datapoints


def convert_and_tokenize(sample, max_length, tokenizer):
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
        max_length=max_length,
        padding=False,
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


def build_and_save(
    split: str,
    output_dir: str,
    tokenizer,
    data_path: str,
    max_length: int = 1024,
    shard_size: int = 50000,
):
    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=data_path)
 
    sft_dataset: List[Dict] = []
    for datapoint in dataset[split]:
        sft_dataset.extend(split_into_turns(datapoint))
 
    print(f"[{split}] Total turn-level samples: {len(sft_dataset)}")
 
    tokenized_buffer = []
    shard_id = 0
    os.makedirs(output_dir, exist_ok=True)
 
    for sample in tqdm(sft_dataset, desc=f"Tokenizing {split}"):
        tokenized_sample = convert_and_tokenize(sample, max_length, tokenizer)
 
        # Skip empty or fully-masked samples
        if len(tokenized_sample["input_ids"]) == 0:
            continue
        if all(l == -100 for l in tokenized_sample["labels"]):
            continue
 
        tokenized_buffer.append(tokenized_sample)
 
        if len(tokenized_buffer) >= shard_size:
            Dataset.from_list(tokenized_buffer).save_to_disk(
                os.path.join(output_dir, f"shard_{shard_id}")
            )
            print(f"  Saved shard_{shard_id} ({len(tokenized_buffer)} samples)")
            tokenized_buffer = []
            shard_id += 1
 
    if tokenized_buffer:
        Dataset.from_list(tokenized_buffer).save_to_disk(os.path.join(output_dir, f"shard_{shard_id}"))
        print(f"  Saved shard_{shard_id} ({len(tokenized_buffer)} samples)")


if __name__ == "__main__":
    DATA_PATH   = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data"
    MODEL_PATH  = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B_4bit"
    MAX_LENGTH  = 1024
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
 
    build_and_save(
        split="train",
        output_dir=os.path.join(DATA_PATH, "tokenized/train"),
        tokenizer=tokenizer,
        data_path=DATA_PATH,
        max_length=MAX_LENGTH,
    )
 
    build_and_save(
        split="test",
        output_dir=os.path.join(DATA_PATH, "tokenized/test"),
        tokenizer=tokenizer,
        data_path=DATA_PATH,
        max_length=MAX_LENGTH,
    )