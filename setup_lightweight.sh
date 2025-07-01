#!/bin/bash
set -e  # Exit on any error

echo "Setting up lightweight local LLM API..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create models directory
mkdir -p models/tiny-gpt2

# Download tiny-gpt2 model and tokenizer with better error handling
echo "Downloading tiny-gpt2 model and tokenizer..."
python -c "
import os
import sys
from transformers import TFAutoModelForCausalLM, AutoTokenizer

try:
    print('Starting model download...')
    
    # Set cache directory
    cache_dir = 'models/tiny-gpt2'
    print(f'Cache directory: {cache_dir}')
    
    # Download tokenizer first (smaller, less likely to fail)
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', cache_dir=cache_dir)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('Tokenizer downloaded successfully.')
    
    # Download model with memory optimizations (removed low_cpu_mem_usage)
    print('Downloading model (this may take a few minutes)...')
    model = TFAutoModelForCausalLM.from_pretrained(
        'sshleifer/tiny-gpt2', 
        cache_dir=cache_dir
    )
    print('Model downloaded successfully.')
    
    # Test the model
    print('Testing model...')
    test_input = tokenizer.encode('Hello', return_tensors='tf')
    test_output = model.generate(test_input, max_new_tokens=5, do_sample=False)
    print('Model test successful.')
    
    print('Tiny-GPT2 model and tokenizer downloaded and tested successfully.')
    
except Exception as e:
    print(f'Error during setup: {e}')
    print('Setup failed. Check the logs above for details.')
    sys.exit(1)
"

echo "Lightweight setup complete." 