#!/bin/bash

echo "Setting up lightweight local LLM API..."

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models/tiny-gpt2

# Download tiny-gpt2 model and tokenizer (ultra-small for free tier)
echo "Downloading tiny-gpt2 model and tokenizer..."
python -c "
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import os

# Set cache directory
cache_dir = 'models/tiny-gpt2'

# Download model with aggressive memory optimizations
model = TFAutoModelForCausalLM.from_pretrained(
    'sshleifer/tiny-gpt2', 
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    tf_dtype='float16'
)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', cache_dir=cache_dir)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Tiny-GPT2 model and tokenizer downloaded.')
"

echo "Lightweight setup complete." 