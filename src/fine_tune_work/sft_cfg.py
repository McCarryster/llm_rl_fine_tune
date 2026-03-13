import os
import math
from datasets import load_from_disk


# MODEL_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B"
MODEL_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B_4bit"
OUTPUT_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/sft_qwen2.5_1.5B"

TRAIN_SFT_SET = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/tokenized/train"
TEST_SFT_SET = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/tokenized/test"


def get_total_count_from_shards(root_dir: str) -> int:
    """
    Iterates through shard subdirectories and sums their row counts.
    Does not load the actual data into memory.
    """
    total_samples = 0
    # Get all subdirectories (shard_0, shard_1, etc.)
    shard_dirs = [
        os.path.join(root_dir, d) 
        for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    for shard_path in shard_dirs:
        # load_from_disk is lazy; it only reads the metadata/header
        ds = load_from_disk(shard_path)
        total_samples += ds.num_rows # type: ignore
    return total_samples

def compute_max_steps(
    dataset_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int = 1,
) -> int:
    total_samples = get_total_count_from_shards(dataset_dir)
    effective_batch = batch_size * gradient_accumulation_steps
    max_steps = math.ceil(total_samples * num_epochs / effective_batch)
    print(f"[Steps] total_samples={total_samples}, effective_batch={effective_batch}, max_steps={max_steps}")
    return max_steps


BASE_MODEL   = "Qwen/Qwen2.5-1.5B"
MAX_SEQ_LEN  = 1024
EPOCHS       = 1
BATCH_SIZE   = 1 # 3
EVAL_BATCH_SIZE   = 1
EVAL_STEPS = 5000
GRAD_ACCUM   = 16 # 6
MAX_STEPS    = compute_max_steps(TRAIN_SFT_SET, BATCH_SIZE, GRAD_ACCUM, EPOCHS)
GRAD_CHECKPOINT = False
LR           = 2e-5
LOG_STEPS    = 10