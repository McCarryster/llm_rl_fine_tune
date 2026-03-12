
BASE_MODEL   = "Qwen/Qwen2.5-1.5B"
MAX_SEQ_LEN  = 2200
EPOCHS       = 1
BATCH_SIZE   = 1
GRAD_ACCUM   = 16
GRAD_CHECKPOINT = False
LR           = 2e-5


# MODEL_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B"
MODEL_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B_4bit"
OUTPUT_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/sft_qwen2.5_1.5B"

TRAIN_SFT_SET = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/sft/train/sft_train"
TEST_SFT_SET = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/data/processed/sft/test/sft_test"