from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sft_cfg as cfg

save_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/llm_rl_fine_tune/models/qwen2.5_1.5B_4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR)

# Check memory usage
mem = model.get_memory_footprint() / 1024**3
print(f"Model memory footprint: {mem:.2f} GB")

# Quick generation test to verify model still works
messages = [{"role": "user", "content": "What is 2+2?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print("\nModel output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")