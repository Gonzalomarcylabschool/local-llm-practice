from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import os
import sys

app = FastAPI()

# Use the smallest possible model for free tier
MODEL_NAME = 'sshleifer/tiny-gpt2'  # Much smaller than regular GPT-2
MODEL_DIR = 'models/tiny-gpt2'

# Memory optimization settings
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
except Exception as e:
    print(f"GPU configuration error (non-critical): {e}")

def load_model_and_tokenizer():
    """Load model with aggressive memory optimizations"""
    try:
        print(f"Starting model loading from {MODEL_DIR}...")
        
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            print(f"Model directory {MODEL_DIR} not found!")
            return None, None
        
        # Check if model files exist
        model_files = os.listdir(MODEL_DIR)
        print(f"Found {len(model_files)} files in model directory: {model_files[:5]}...")
        
        # Load model with aggressive memory settings
        print("Loading model...")
        model = TFAutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            cache_dir=MODEL_DIR,
            tf_dtype=tf.float16,  # Use half precision
            local_files_only=True  # Only use local files
        )
        print("Model loaded successfully.")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR, local_files_only=True)
        print("Tokenizer loaded successfully.")
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test the model
        print("Testing model with sample input...")
        test_input = tokenizer.encode("Hello", return_tensors="tf")
        test_output = model.generate(test_input, max_new_tokens=5, do_sample=False)
        print("Model test successful.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Model directory contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
        return None, None

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
    print("Starting up FastAPI application...")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Model directory: {MODEL_DIR}")
    
    try:
        model, tokenizer = load_model_and_tokenizer()
        if model is not None and tokenizer is not None:
            print("Model and tokenizer loaded successfully!")
            # Force garbage collection after loading
            gc.collect()
        else:
            print("Failed to load model and tokenizer!")
    except Exception as e:
        print(f"Startup error: {e}")

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check the build logs for model download issues."
        )
    
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
        "memory_usage": "ultra_lightweight",
        "model_directory": MODEL_DIR,
        "model_directory_exists": os.path.exists(MODEL_DIR)
    }

@app.get("/")
def root():
    return {
        "message": "Local LLM API - Lightweight Version",
        "model": MODEL_NAME,
        "endpoints": ["/generate", "/health"],
        "memory_optimized": True,
        "model_loaded": model is not None
    } 