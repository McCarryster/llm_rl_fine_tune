from datasets import load_dataset
from typing import Dict, List
from datasets import Dataset

def prepare_dpo_data(sample):
    """
    Split chosen/rejected into (prompt, chosen_response, rejected_response).
    The prompt = full conversation up to (and including) the last 'Assistant: '
    """
    split_token = "\n\nAssistant: "
    
    # Find the last Assistant turn in chosen
    idx = sample["chosen"].rfind(split_token)
    
    # Everything up to and including "Assistant: " is the prompt
    prompt   = sample["chosen"][:idx + len(split_token)]
    
    # Everything after is the preferred response
    chosen   = sample["chosen"][idx + len(split_token):]
    
    # Same prompt, but take the rejected response
    rejected = sample["rejected"][idx + len(split_token):]
    
    return {
        "prompt":   prompt,
        "chosen":   chosen,
        "rejected": rejected,
    }


if __name__ == "__main__":
    data_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data"
    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=data_path)
    
    dpo_dataset: List[Dict[str, str]] = []
    for datapoint in dataset['test']:
        processed_datapoint = prepare_dpo_data(datapoint)
        dpo_dataset.append(processed_datapoint)

    dataset_obj = Dataset.from_list(dpo_dataset)
    dataset_obj.save_to_disk("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/dpo/test/dpo_test")