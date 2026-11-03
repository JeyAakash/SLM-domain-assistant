import os
import time
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Define held-out test dataset for evaluation (Milestone 7)[cite: 1]
test_data = [
    {
        "prompt": "Where is the main library?", 
        "expected": "Building A"
    },
    {
        "prompt": "What are the library hours on Sunday?", 
        "expected": "10:00 AM"
    }
]

model_id = "unsloth/Llama-3.2-1B-Instruct"

print("--- EVALUATION START ---")
print(f"Loading Tokenizer for: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Configure 4-bit quantization to prevent memory errors on 4GB VRAM / 8GB RAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Base Model in 4-bit quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 3. Safely resolve and normalize absolute path for Windows compatibility
raw_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experiments/final_adapter"))
normalized_adapter_path = raw_adapter_path.replace("\\", "/")

print(f"Loading Fine-Tuned PEFT Adapter from: {normalized_adapter_path}")

# Load model using is_local_path=True to correctly handle local Windows directories
model = PeftModel.from_pretrained(
    base_model, 
    normalized_adapter_path, 
    is_local_path=True
)
model.eval()

# Reset CUDA peak memory tracking
torch.cuda.reset_peak_memory_stats()

# 4. Run Evaluation on Test Cases (Milestone 7)[cite: 1]
for i, item in enumerate(test_data):
    inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7
        )
    latency = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Converted to MB

    print(f"\n--- Test Case {i+1} ---")
    print(f"Prompt: {item['prompt']}")
    print(f"Expected Context: {item['expected']}")
    print(f"Generated Output:\n{response}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")

print("\n--- EVALUATION COMPLETE ---")