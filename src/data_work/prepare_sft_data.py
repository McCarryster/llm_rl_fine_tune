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
        max_length=2200,
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

if __name__ == "__main__":
    data_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data"
    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=data_path)
    
    sft_dataset: List[Dict[str, str]] = []
    for datapoint in dataset['test']:
        processed_datapoints = split_into_turns(datapoint)
        sft_dataset.extend(processed_datapoints)
    
    tokenizer = AutoTokenizer.from_pretrained("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B_4bit")
    tokenizer.pad_token = tokenizer.eos_token

    sft_tokenized_dataset = []
    shard_size: int = 50000
    shard_id: int = 0
    show_progress = True
    output_dir = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/tokenized/test"

    for sample in tqdm(sft_dataset, desc="Tokenizing", disable=not show_progress):
        tokenized_sample = convert_and_tokenize(sample, tokenizer)
        sft_tokenized_dataset.append(tokenized_sample)

        if len(sft_tokenized_dataset) >= shard_size:
            dataset_obj = Dataset.from_list(sft_tokenized_dataset)
            shard_path: str = os.path.join(output_dir, f"shard_{shard_id}")
            dataset_obj.save_to_disk(shard_path)
            sft_tokenized_dataset = []
            shard_id += 1

    # save remaining samples
    if len(sft_tokenized_dataset) > 0:
        dataset_obj = Dataset.from_list(sft_tokenized_dataset)
        shard_path: str = os.path.join(output_dir, f"shard_{shard_id}")
        dataset_obj.save_to_disk(shard_path)

    # dataset_obj = Dataset.from_list(sft_tokenized_dataset)
    # dataset_obj.save_to_disk("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/tokenized/train/sft_train")