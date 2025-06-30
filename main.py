from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

app = FastAPI()

MODEL_NAME = 'distilgpt2'
MODEL_DIR = 'models/distilgpt2'

def load_model_and_tokenizer():
    model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    return model, tokenizer

model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompts: Union[str, List[str]]
    max_tokens: int = 128
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    results: List[str]

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer()

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    prompts = request.prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="tf")
        output = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(generated)
    return {"results": results}

@app.get("/health")
def health():
    return {"status": "ok"} 