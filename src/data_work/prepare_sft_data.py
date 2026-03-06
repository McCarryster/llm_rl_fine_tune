from datasets import load_dataset
from typing import Dict, List
from datasets import Dataset

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


if __name__ == "__main__":
    data_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data"
    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=data_path)
    
    sft_dataset: List[Dict[str, str]] = []
    for datapoint in dataset['test']:
        processed_datapoints = split_into_turns(datapoint)
        sft_dataset.extend(processed_datapoints)
    
    dataset_obj = Dataset.from_list(sft_dataset)
    dataset_obj.save_to_disk("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/test/sft_test")