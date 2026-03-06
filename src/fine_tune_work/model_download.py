from huggingface_hub import snapshot_download

def download_qwen_embedding_model(
    local_dir: str,
    model_id: str,
) -> str:
    print(f"Attempting to download model '{model_id}' to: {local_dir}")
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir
        )
        print(f"Model downloaded successfully to: {local_path}")
        return local_path
    except Exception as e:
        print(f"An error occurred during download: {e}")
        raise

download_qwen_embedding_model("/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/llama3.2_1B", "meta-llama/Llama-3.2-1B")