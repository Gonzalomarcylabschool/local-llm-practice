from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import os

app = FastAPI()

# Use the smallest possible model for free tier
MODEL_NAME = 'sshleifer/tiny-gpt2'  # Much smaller than regular GPT-2
MODEL_DIR = 'models/tiny-gpt2'

# Memory optimization settings
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

def load_model_and_tokenizer():
    """Load model with aggressive memory optimizations"""
    # Set memory growth to prevent TensorFlow from allocating all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Load model with aggressive memory settings
    model = TFAutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=MODEL_DIR,
        low_cpu_mem_usage=True,
        tf_dtype=tf.float16,  # Use half precision
        device_map='auto' if tf.config.list_physical_devices('GPU') else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompts: Union[str, List[str]]
    max_tokens: int = 32  # Very small for memory constraints
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    results: List[str]

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
        # Force garbage collection after loading
        gc.collect()
        print("Model loaded successfully with memory optimizations")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue without model - will return error on requests

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    prompts = request.prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Very strict batch size limit
    if len(prompts) > 1:
        raise HTTPException(status_code=400, detail="Maximum 1 prompt per request for memory constraints.")
    
    results = []
    for prompt in prompts:
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="tf")
            
            # Use very small generation parameters
            output = model.generate(
                input_ids,
                max_new_tokens=min(request.max_tokens, 32),  # Cap at 32 tokens
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True,
                repetition_penalty=1.1
            )
            
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            results.append(generated)
            
            # Clear memory immediately
            del output
            gc.collect()
            
        except Exception as e:
            results.append(f"Error generating text: {str(e)}")
    
    return {"results": results}

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model": MODEL_NAME,
        "memory_usage": "ultra_lightweight"
    }

@app.get("/")
def root():
    return {
        "message": "Local LLM API - Lightweight Version",
        "model": MODEL_NAME,
        "endpoints": ["/generate", "/health"],
        "memory_optimized": True
    } 