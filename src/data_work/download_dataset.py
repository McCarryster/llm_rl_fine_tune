import os
from datasets import load_dataset

# 1. Define your absolute project data path
project_data_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data"

# 2. Ensure the directory exists
os.makedirs(project_data_path, exist_ok=True)

# 3. Load the dataset using cache_dir
# This will download and process the data DIRECTLY into your specified folder
dataset = load_dataset(
    "Anthropic/hh-rlhf", 
    cache_dir=project_data_path
)

print(f"Dataset loaded and stored in: {project_data_path}")
print(dataset['train'][1400])