#!/bin/bash

echo "Setting up local LLM API..."

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models/gpt2

# Download GPT-2 model and tokenizer (smaller than DistilGPT2)
echo "Downloading GPT-2 model and tokenizer..."
python -c "
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import os

# Set cache directory
cache_dir = 'models/gpt2'

# Download model with memory optimizations
model = TFAutoModelForCausalLM.from_pretrained(
    'gpt2', 
    cache_dir=cache_dir
)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('GPT-2 model and tokenizer downloaded.')
"

echo "Setup complete." 