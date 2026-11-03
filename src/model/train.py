import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# 1. Load model in 4-bit quantization to fit safely within 4GB VRAM
model_id = "unsloth/Llama-3.2-1B-Instruct"

print("Loading tokenizer and base model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 2. Prepare model for PEFT training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 3. Load dataset
print("Loading training dataset...")
dataset = load_dataset("json", data_files="data/processed/train.jsonl", split="train")

# 4. Use SFTConfig for SFT-specific arguments (compatible with newer trl versions)
training_args = SFTConfig(
    output_dir="experiments/outputs",
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=2,
    num_train_epochs=3,
    bf16=True,                       # Changed from fp16=True to bf16=True to fix the scaler error
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    dataset_text_field="text",       
    max_length=512                   
)

trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset, 
    processing_class=tokenizer,      # Replaces 'tokenizer' argument in newer trl
    args=training_args
)

print("Starting training...")
trainer.train()

# 5. Save the final adapter files
output_dir = os.path.abspath("experiments/final_adapter")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving PEFT adapter to: {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Training complete and adapter saved successfully!")