import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(
    title="SLM Domain Assistant API", 
    description="Production API powered by your fine-tuned Llama-3.2-1B-Instruct model under hardware constraints.", 
    version="1.0"
)

# 1. Model configuration
model_id = "unsloth/Llama-3.2-1B-Instruct"

print("Loading tokenizer and base model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 2. Safely resolve and normalize absolute path for local adapter loading
raw_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experiments/final_adapter"))
normalized_adapter_path = raw_adapter_path.replace("\\", "/")

print(f"Loading local fine-tuned PEFT adapter from: {normalized_adapter_path}")
model = PeftModel.from_pretrained(
    base_model, 
    normalized_adapter_path, 
    is_local_path=True
)
model.eval()

# 3. Pydantic Schemas for API Validation
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 60

class QueryResponse(BaseModel):
    response: str

class AdvancedQueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.7  # Customize creativity dynamically

class AdvancedQueryResponse(BaseModel):
    prompt: str
    response: str
    temperature: float

# 4. API Endpoints
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to your SLM Domain Assistant API!",
        "docs_url": "http://127.0.0.1:8000/docs"
    }

@app.get("/model-info")
def get_model_info():
    return {
        "base_model": model_id,
        "adapter_path": normalized_adapter_path,
        "device": str(model.device),
        "status": "loaded and ready"
    }

@app.post("/generate", response_model=QueryResponse)
def generate_text(request: QueryRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=0.7
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-advanced", response_model=AdvancedQueryResponse)
def generate_advanced_text(request: AdvancedQueryRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "prompt": request.prompt,
            "response": result,
            "temperature": request.temperature
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))